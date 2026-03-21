import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List

from model.tensorflow.embedding import SparseEmbedding
from model.tensorflow.mlp import MLP


class AttentionSequencePoolingLayer(layers.Layer):
    """
    Fixed: Changed to DIEN paper's Bilinear Attention and Softmax normalization.
    """

    def __init__(
        self,
        return_score: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_score = return_score

    def build(self, input_shape):
        query_shape, key_shape, _ = input_shape
        # query_shape: (batch_size, dim_emb)
        # key_shape: (batch_size, seq_len, hidden_dim)
        self.W = self.add_weight(
            name="bilinear_W",
            shape=(key_shape[-1], query_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        query_emb, key_emb, seq_length = inputs

        # 修复 3: 使用论文中的 Bilinear Attention: h_t W e_a
        # kW shape: (batch_size, seq_len, dim_emb)
        kW = tf.tensordot(key_emb, self.W, axes=[[-1], [0]])

        query_emb_expand = tf.expand_dims(query_emb, axis=-1)
        # att_score shape: (batch_size, seq_len, 1) -> (batch_size, seq_len)
        att_score = tf.matmul(kW, query_emb_expand)
        att_score = tf.squeeze(att_score, axis=-1)

        # 获取 Mask 屏蔽 padding 部分
        seq_len = tf.shape(key_emb)[1]
        mask = tf.sequence_mask(
            tf.squeeze(seq_length, axis=-1), maxlen=seq_len, dtype=tf.float32
        )

        # 将无效位置的分数设为一个极小值，以防止 softmax 后获得权重
        paddings = tf.ones_like(att_score) * (-(2**32) + 1)
        att_score = tf.where(mask > 0, att_score, paddings)

        # 修复 4: 强制使用 Softmax 沿时间序列维度进行归一化
        att_score = tf.nn.softmax(att_score, axis=1)  # (batch_size, seq_len)
        att_score_expand = tf.expand_dims(att_score, axis=-1)

        if self.return_score:
            return att_score_expand

        weighted_output = tf.reduce_sum(att_score_expand * key_emb, axis=1)
        return weighted_output


class DynamicGRU(layers.Layer):
    def __init__(
        self,
        units: int,
        gru_type: str = "GRU",
        return_sequence: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.gru_type = gru_type.upper()
        self.return_sequence = return_sequence
        self.bias = bias

        self.update_gate = layers.Dense(units, use_bias=bias, activation="sigmoid")
        self.reset_gate = layers.Dense(units, use_bias=bias, activation="sigmoid")
        self.candidate_gate = layers.Dense(units, use_bias=bias, activation="tanh")

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        if self.gru_type in ["AGRU", "AUGRU"]:
            seq_emb, seq_length, att_score = inputs
            att_score = tf.squeeze(att_score, axis=-1)
        else:
            seq_emb, seq_length = inputs
            att_score = None

        batch_size = tf.shape(seq_emb)[0]
        seq_len = tf.shape(seq_emb)[1]
        seq_length_squeeze = tf.squeeze(seq_length, axis=-1)

        h_t = tf.zeros((batch_size, self.units), dtype=seq_emb.dtype)
        outputs_ta = tf.TensorArray(dtype=seq_emb.dtype, size=seq_len)

        def loop_body(t, h_t, outputs_ta):
            x_t = seq_emb[:, t, :]
            mask_t = tf.cast(t < seq_length_squeeze, dtype=seq_emb.dtype)
            mask_t = tf.expand_dims(mask_t, -1)

            if self.gru_type == "AIGRU" and att_score is not None:
                x_t = x_t * att_score[:, t : t + 1]

            gate_input = tf.concat([x_t, h_t], axis=-1)
            u_t = self.update_gate(gate_input)
            r_t = self.reset_gate(gate_input)
            candidate_input = tf.concat([x_t, r_t * h_t], axis=-1)
            h_tilde = self.candidate_gate(candidate_input)

            if self.gru_type == "GRU" or self.gru_type == "AIGRU":
                h_t_new = (1 - u_t) * h_t + u_t * h_tilde
            elif self.gru_type == "AGRU":
                a_t = att_score[:, t : t + 1]
                h_t_new = (1 - a_t) * h_t + a_t * h_tilde
            elif self.gru_type == "AUGRU":
                a_t = att_score[:, t : t + 1]
                u_tilde = u_t * a_t
                h_t_new = (1 - u_tilde) * h_t + u_tilde * h_tilde

            h_t = mask_t * h_t_new + (1 - mask_t) * h_t
            outputs_ta = outputs_ta.write(t, h_t)

            return t + 1, h_t, outputs_ta

        def loop_cond(t, h_t, outputs_ta):
            return t < seq_len

        t_final, h_final, outputs_ta_final = tf.while_loop(
            cond=loop_cond, body=loop_body, loop_vars=(0, h_t, outputs_ta)
        )

        if self.return_sequence:
            outputs = outputs_ta_final.stack()
            outputs = tf.transpose(outputs, [1, 0, 2])
            return outputs
        else:
            return h_final


def auxiliary_loss(
    rnn_states: tf.Tensor,
    click_seq: tf.Tensor,
    noclick_seq: tf.Tensor,
    seq_length: tf.Tensor,
) -> tf.Tensor:
    """
    Fixed: Removed MLP bug, implemented inner product similarity as per paper.
    """
    batch_size = tf.shape(click_seq)[0]
    seq_len = tf.shape(click_seq)[1]
    seq_length_squeeze = tf.squeeze(seq_length, axis=-1)
    mask = tf.sequence_mask(seq_length_squeeze, maxlen=seq_len, dtype=tf.float32)

    rnn_states_aligned = rnn_states[:, :-1, :]
    click_seq_aligned = click_seq[:, 1:, :]
    noclick_seq_aligned = noclick_seq[:, 1:, :]
    mask_aligned = mask[:, 1:]

    # 修复 1 & 2: 移除失效的 MLP，使用论文中定义的内积公式
    # 假设 rnn_states_aligned 和 click_seq_aligned 维度对齐 (hidden_dim == dim_emb)
    click_pred = tf.sigmoid(
        tf.reduce_sum(rnn_states_aligned * click_seq_aligned, axis=-1)
    )
    noclick_pred = tf.sigmoid(
        tf.reduce_sum(rnn_states_aligned * noclick_seq_aligned, axis=-1)
    )

    click_loss = -tf.math.log(tf.clip_by_value(click_pred, 1e-8, 1.0)) * mask_aligned
    noclick_loss = (
        -tf.math.log(tf.clip_by_value(1.0 - noclick_pred, 1e-8, 1.0)) * mask_aligned
    )

    total_loss = tf.reduce_sum(click_loss + noclick_loss)
    num_valid_samples = tf.reduce_sum(mask_aligned) + 1e-8
    avg_loss = total_loss / num_valid_samples

    return avg_loss


class InterestExtractorLayer(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        use_auxiliary_loss: bool = True,
        aux_loss_alpha: float = 1.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.use_auxiliary_loss = use_auxiliary_loss
        self.aux_loss_alpha = aux_loss_alpha
        self.bias = bias
        self.gru = DynamicGRU(
            units=hidden_dim, gru_type="GRU", return_sequence=True, bias=bias
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        if self.use_auxiliary_loss:
            behavior_seq_emb, seq_length, neg_behavior_seq_emb = inputs
        else:
            behavior_seq_emb, seq_length = inputs
            neg_behavior_seq_emb = None

        interest_states = self.gru([behavior_seq_emb, seq_length], training=training)

        if self.use_auxiliary_loss and training and neg_behavior_seq_emb is not None:
            aux_loss_val = auxiliary_loss(
                rnn_states=interest_states,
                click_seq=behavior_seq_emb,
                noclick_seq=neg_behavior_seq_emb,
                seq_length=seq_length,
            )
            # 修复 5: 实际应用 aux_loss_alpha 乘数
            self.add_loss(self.aux_loss_alpha * aux_loss_val)

        return interest_states


class InterestEvolutionLayer(layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        gru_type: str = "AUGRU",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.gru_type = gru_type.upper()
        self.bias = bias

        self.attention_layer = AttentionSequencePoolingLayer(
            return_score=True,
        )

        self.evolution_gru = DynamicGRU(
            units=hidden_dim, gru_type=gru_type, return_sequence=False, bias=bias
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        interest_seq, target_emb, seq_length = inputs

        att_scores = self.attention_layer(
            [target_emb, interest_seq, seq_length], training=training
        )

        if self.gru_type in ["AGRU", "AUGRU"]:
            final_interest = self.evolution_gru(
                [interest_seq, seq_length, att_scores], training=training
            )
        else:
            if self.gru_type == "AIGRU":
                interest_seq = interest_seq * att_scores
            final_interest = self.evolution_gru(
                [interest_seq, seq_length], training=training
            )

        return final_interest


class DIEN(Model):
    def __init__(
        self,
        num_sparse_embs: List[int],
        num_seq_embs: int,
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        max_seq_len: int,
        extractor_hidden_dim: int,
        evolution_hidden_dim: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        use_auxiliary_loss: bool = True,
        aux_loss_alpha: float = 1.0,
        gru_type: str = "AUGRU",
        dim_output: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_emb = dim_emb
        self.use_auxiliary_loss = use_auxiliary_loss
        self.max_seq_len = max_seq_len
        self.dim_input_dense = dim_input_dense

        # Embedding Layers
        self.static_sparse_embedding = SparseEmbedding(num_sparse_embs, dim_emb)
        self.dense_embedding = tf.keras.layers.Dense(
            units=dim_input_dense * dim_emb, use_bias=bias
        )
        self.seq_embedding = tf.keras.layers.Embedding(
            input_dim=num_seq_embs, output_dim=dim_emb, mask_zero=True
        )
        if self.use_auxiliary_loss:
            self.neg_seq_embedding = tf.keras.layers.Embedding(
                input_dim=num_seq_embs, output_dim=dim_emb, mask_zero=True
            )

        # Core DIEN Layers
        self.interest_extractor = InterestExtractorLayer(
            hidden_dim=extractor_hidden_dim,
            use_auxiliary_loss=use_auxiliary_loss,
            aux_loss_alpha=aux_loss_alpha,
            bias=bias,
        )
        self.interest_evolution = InterestEvolutionLayer(
            hidden_dim=evolution_hidden_dim,
            gru_type=gru_type,
            bias=bias,
        )

        # Prediction Head
        self.prediction_head = MLP(
            dim_in=(dim_input_sparse + dim_input_dense) * dim_emb
            + evolution_hidden_dim,
            num_hidden=num_hidden_head,
            dim_hidden=dim_hidden_head,
            dim_out=dim_output,
            dropout=dropout,
            bias=bias,
        )

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        if self.use_auxiliary_loss:
            (
                static_sparse_inputs,
                dense_inputs,
                seq_inputs,
                seq_length,
                neg_seq_inputs,
            ) = inputs
        else:
            static_sparse_inputs, dense_inputs, seq_inputs, seq_length = inputs
            neg_seq_inputs = None

        static_sparse_emb = self.static_sparse_embedding(static_sparse_inputs)
        dense_emb = self.dense_embedding(dense_inputs)
        dense_emb = tf.reshape(dense_emb, [-1, self.dim_input_dense, self.dim_emb])
        seq_emb = self.seq_embedding(seq_inputs)
        neg_seq_emb = None
        if self.use_auxiliary_loss and neg_seq_inputs is not None:
            neg_seq_emb = self.neg_seq_embedding(neg_seq_inputs)

        target_item_emb = static_sparse_emb[:, 0, :]

        if self.use_auxiliary_loss:
            extractor_inputs = [seq_emb, seq_length, neg_seq_emb]
        else:
            extractor_inputs = [seq_emb, seq_length]
        interest_states = self.interest_extractor(extractor_inputs, training=training)

        evolution_inputs = [interest_states, target_item_emb, seq_length]
        final_interest_emb = self.interest_evolution(
            evolution_inputs, training=training
        )

        static_sparse_flat = tf.reshape(
            static_sparse_emb, [-1, static_sparse_emb.shape[1] * self.dim_emb]
        )
        dense_flat = tf.reshape(dense_emb, [-1, dense_emb.shape[1] * self.dim_emb])
        final_features = tf.concat(
            [static_sparse_flat, dense_flat, final_interest_emb], axis=-1
        )

        logits = self.prediction_head(final_features, training=training)
        return logits


if __name__ == "__main__":
    # Example usage
    # Note for Issue 6: When calling this model in an actual training pipeline,
    # the dataset must have genuine variable-length sequence data.
    model = DIEN(
        num_sparse_embs=[1000, 1000],
        num_seq_embs=1000,
        dim_emb=64,
        dim_input_sparse=2,
        dim_input_dense=1,
        max_seq_len=50,
        extractor_hidden_dim=64,  # Important: should equal dim_emb to allow dot product
        evolution_hidden_dim=64,
        num_hidden_head=2,
        dim_hidden_head=32,
        use_auxiliary_loss=True,
        aux_loss_alpha=1.0,
        gru_type="AUGRU",
        dim_output=1,
        dropout=0.2,
        bias=True,
    )

    batch_size = 4
    static_sparse_inputs = tf.random.uniform(
        (batch_size, 2), maxval=1000, dtype=tf.int32
    )
    dense_inputs = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    seq_inputs = tf.random.uniform((batch_size, 50), maxval=1000, dtype=tf.int32)
    seq_length = tf.constant([[50], [45], [30], [20]], dtype=tf.int32)
    neg_seq_inputs = tf.random.uniform((batch_size, 50), maxval=1000, dtype=tf.int32)

    inputs = (
        static_sparse_inputs,
        dense_inputs,
        seq_inputs,
        seq_length,
        neg_seq_inputs,
    )
    outputs = model(inputs, training=True)
    print("Model output shape:", outputs.shape)

    # Check layer loss length to confirm aux loss applies
    print("Number of auxiliary losses applied:", len(model.losses))

    model = DIEN(
        num_sparse_embs=[1000, 1000],
        num_seq_embs=1000,
        dim_emb=64,
        dim_input_sparse=2,
        dim_input_dense=1,
        max_seq_len=50,
        extractor_hidden_dim=64,  # Important: should equal dim_emb to allow dot product
        evolution_hidden_dim=64,
        num_hidden_head=2,
        dim_hidden_head=32,
        use_auxiliary_loss=False,
        aux_loss_alpha=1.0,
        gru_type="AUGRU",
        dim_output=1,
        dropout=0.2,
        bias=True,
    )

    batch_size = 4
    static_sparse_inputs = tf.random.uniform(
        (batch_size, 2), maxval=1000, dtype=tf.int32
    )
    dense_inputs = tf.random.uniform((batch_size, 1), dtype=tf.float32)
    seq_inputs = tf.random.uniform((batch_size, 50), maxval=1000, dtype=tf.int32)
    seq_length = tf.constant([[50], [45], [30], [20]], dtype=tf.int32)

    inputs = (
        static_sparse_inputs,
        dense_inputs,
        seq_inputs,
        seq_length,
    )
    outputs = model(inputs, training=True)
    print("Model output shape:", outputs.shape)

    # Check layer loss length to confirm aux loss applies
    print("Number of auxiliary losses applied:", len(model.losses))
