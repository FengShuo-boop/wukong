import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Optional
from model.tensorflow.embedding import SparseEmbedding
from model.tensorflow.mlp import MLP


class Dice(layers.Layer):
    def __init__(self, epsilon: float = 1e-9, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.bn = layers.BatchNormalization(
            center=False, scale=False, epsilon=self.epsilon
        )
        self.alpha = self.add_weight(
            name="dice_alpha",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_norm = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_norm)
        return self.alpha * (1.0 - x_p) * inputs + x_p * inputs


class AttentionSequencePoolingLayer(layers.Layer):
    def __init__(
        self,
        att_hidden_units: List[int] = (64, 16),
        att_activation: str = "dice",
        weight_normalization: bool = False,
        return_score: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.bias = bias

        self.att_layers = []
        for hidden_dim in att_hidden_units:
            self.att_layers.append(layers.Dense(hidden_dim, use_bias=bias))
            if att_activation.lower() == "dice":
                self.att_layers.append(Dice())
            elif att_activation.lower() == "relu":
                self.att_layers.append(layers.ReLU())
            elif att_activation.lower() == "sigmoid":
                self.att_layers.append(layers.Activation("sigmoid"))
        self.att_layers.append(layers.Dense(1, use_bias=bias))

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of attention layer
        Args:
            inputs: List of [query_emb, key_emb, seq_length]
                query_emb: Target item embedding, shape (batch_size, dim_emb)
                key_emb: History sequence embedding, shape (batch_size, seq_len, dim_emb)
                seq_length: Valid length of each sequence, shape (batch_size, 1)
            training: Boolean indicating training mode
        Returns:
            Weighted pooling result or attention scores
        """
        query_emb, key_emb, seq_length = inputs
        batch_size = tf.shape(key_emb)[0]
        seq_len = tf.shape(key_emb)[1]

        query_expand = tf.tile(tf.expand_dims(query_emb, 1), [1, seq_len, 1])
        att_input = tf.concat(
            [query_expand, key_emb, query_expand - key_emb, query_expand * key_emb],
            axis=-1,
        )

        for layer in self.att_layers:
            if isinstance(layer, (Dice, layers.BatchNormalization, layers.Dropout)):
                att_input = layer(att_input, training=training)
            else:
                att_input = layer(att_input)
        att_score = att_input

        mask = tf.sequence_mask(
            tf.squeeze(seq_length, axis=-1), maxlen=seq_len, dtype=tf.float32
        )
        mask = tf.expand_dims(mask, -1)
        att_score = att_score * mask + (1.0 - mask) * (-(2**32) + 1)

        if self.weight_normalization:
            att_score = tf.nn.softmax(att_score, axis=1)
        else:
            att_score = tf.sigmoid(att_score)

        if self.return_score:
            return att_score

        weighted_output = tf.reduce_sum(att_score * key_emb, axis=1)
        return weighted_output


class DynamicGRU(layers.Layer):
    """
    Dynamic GRU Layer with support for multiple GRU variants: GRU, AIGRU, AGRU, AUGRU
    Handles variable-length sequences and integrates attention mechanism for DIEN
    """

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

        if self.gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
            raise ValueError(
                f"Unsupported GRU type: {self.gru_type}, must be one of [GRU, AIGRU, AGRU, AUGRU]"
            )

        self.update_gate = layers.Dense(units, use_bias=bias, activation="sigmoid")
        self.reset_gate = layers.Dense(units, use_bias=bias, activation="sigmoid")
        self.candidate_gate = layers.Dense(units, use_bias=bias, activation="tanh")

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of Dynamic GRU using tf.while_loop for graph compatibility
        Args:
            inputs: List of [seq_emb, seq_length] or [seq_emb, seq_length, att_score]
                seq_emb: Input sequence embedding, shape (batch_size, seq_len, dim_emb)
                seq_length: Valid sequence length per sample, shape (batch_size, 1)
                att_score: Attention scores for AUGRU/AGRU, shape (batch_size, seq_len, 1)
            training: Boolean indicating training mode
        Returns:
            GRU output sequence or final hidden state
        """
        if self.gru_type in ["AGRU", "AUGRU"]:
            seq_emb, seq_length, att_score = inputs
            att_score = tf.squeeze(att_score, axis=-1)
        else:
            seq_emb, seq_length = inputs
            att_score = None

        batch_size = tf.shape(seq_emb)[0]
        seq_len = tf.shape(seq_emb)[1]
        seq_length_squeeze = tf.squeeze(seq_length, axis=-1)

        # Initialize hidden state
        h_t = tf.zeros((batch_size, self.units), dtype=seq_emb.dtype)

        # Initialize output tensor array (always create it to avoid None issues)
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

            # Write to output tensor array
            outputs_ta = outputs_ta.write(t, h_t)

            return t + 1, h_t, outputs_ta

        def loop_cond(t, h_t, outputs_ta):
            return t < seq_len

        # Run while loop
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
    Auxiliary loss for interest extractor layer in DIEN
    Supervises hidden state learning using next click/non-click behaviors
    Args:
        rnn_states: Hidden states from GRU, shape (batch_size, seq_len, hidden_dim)
        click_seq: Embedding of next clicked items, shape (batch_size, seq_len, dim_emb)
        noclick_seq: Embedding of next non-clicked items, shape (batch_size, seq_len, dim_emb)
        seq_length: Valid sequence length, shape (batch_size, 1)
    Returns:
        Computed auxiliary loss scalar
    """
    batch_size = tf.shape(click_seq)[0]
    seq_len = tf.shape(click_seq)[1]
    seq_length_squeeze = tf.squeeze(seq_length, axis=-1)
    mask = tf.sequence_mask(seq_length_squeeze, maxlen=seq_len, dtype=tf.float32)

    rnn_states_aligned = rnn_states[:, :-1, :]
    click_seq_aligned = click_seq[:, 1:, :]
    noclick_seq_aligned = noclick_seq[:, 1:, :]
    mask_aligned = mask[:, 1:]

    click_input = tf.concat([rnn_states_aligned, click_seq_aligned], axis=-1)
    noclick_input = tf.concat([rnn_states_aligned, noclick_seq_aligned], axis=-1)

    aux_mlp = tf.keras.Sequential(
        [
            layers.Dense(100, activation="sigmoid"),
            layers.Dense(50, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    click_pred = tf.squeeze(aux_mlp(click_input), axis=-1)
    noclick_pred = tf.squeeze(aux_mlp(noclick_input), axis=-1)

    click_loss = -tf.math.log(tf.clip_by_value(click_pred, 1e-8, 1.0)) * mask_aligned
    noclick_loss = (
        -tf.math.log(tf.clip_by_value(1.0 - noclick_pred, 1e-8, 1.0)) * mask_aligned
    )

    total_loss = tf.reduce_sum(click_loss + noclick_loss)
    num_valid_samples = tf.reduce_sum(mask_aligned) + 1e-8
    avg_loss = total_loss / num_valid_samples

    return avg_loss


class InterestExtractorLayer(layers.Layer):
    """
    Interest Extractor Layer from DIEN
    Extracts temporal interest states from user behavior sequence using GRU
    Supports auxiliary loss for better interest representation learning
    """

    def __init__(
        self,
        hidden_dim: int,
        use_auxiliary_loss: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.use_auxiliary_loss = use_auxiliary_loss
        self.bias = bias
        self.gru = DynamicGRU(
            units=hidden_dim, gru_type="GRU", return_sequence=True, bias=bias
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of interest extractor
        Args:
            inputs: List of [behavior_seq_emb, seq_length] or [behavior_seq_emb, seq_length, neg_behavior_seq_emb]
                behavior_seq_emb: Embedding of user click behavior sequence, shape (batch_size, seq_len, dim_emb)
                seq_length: Valid sequence length, shape (batch_size, 1)
                neg_behavior_seq_emb: Embedding of non-click behavior sequence, shape (batch_size, seq_len, dim_emb)
            training: Boolean indicating training mode
        Returns:
            Extracted interest sequence, shape (batch_size, seq_len, hidden_dim)
        """
        if self.use_auxiliary_loss:
            behavior_seq_emb, seq_length, neg_behavior_seq_emb = inputs
        else:
            behavior_seq_emb, seq_length = inputs
            neg_behavior_seq_emb = None

        interest_states = self.gru([behavior_seq_emb, seq_length], training=training)

        if self.use_auxiliary_loss and training and neg_behavior_seq_emb is not None:
            aux_loss = auxiliary_loss(
                rnn_states=interest_states,
                click_seq=behavior_seq_emb,
                noclick_seq=neg_behavior_seq_emb,
                seq_length=seq_length,
            )
            self.add_loss(aux_loss)

        return interest_states


class InterestEvolutionLayer(layers.Layer):
    """
    Interest Evolution Layer from DIEN
    Models interest evolving process relative to the target item
    Supports multiple GRU variants with attention mechanism
    """

    def __init__(
        self,
        hidden_dim: int,
        gru_type: str = "AUGRU",
        att_hidden_units: List[int] = (64, 16),
        att_activation: str = "dice",
        att_weight_normalization: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.gru_type = gru_type.upper()
        self.bias = bias

        self.attention_layer = AttentionSequencePoolingLayer(
            att_hidden_units=att_hidden_units,
            att_activation=att_activation,
            weight_normalization=att_weight_normalization,
            return_score=True,
            bias=bias,
        )

        self.evolution_gru = DynamicGRU(
            units=hidden_dim, gru_type=gru_type, return_sequence=False, bias=bias
        )

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of interest evolution layer
        Args:
            inputs: List of [interest_seq, target_emb, seq_length]
                interest_seq: Interest states from extractor layer, shape (batch_size, seq_len, hidden_dim)
                target_emb: Embedding of target item, shape (batch_size, dim_emb)
                seq_length: Valid sequence length, shape (batch_size, 1)
            training: Boolean indicating training mode
        Returns:
            Final evolved interest representation, shape (batch_size, hidden_dim)
        """
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
        gru_type: str = "AUGRU",
        att_hidden_units: List[int] = (64, 16),
        att_activation: str = "dice",
        att_weight_normalization: bool = False,
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
            bias=bias,
        )
        self.interest_evolution = InterestEvolutionLayer(
            hidden_dim=evolution_hidden_dim,
            gru_type=gru_type,
            att_hidden_units=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization,
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

        self.output_names = ["output"]

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        """
        Forward pass of DIEN model
        Args:
            inputs: Tuple of input tensors
                (static_sparse_inputs, dense_inputs, seq_inputs, seq_length) if no auxiliary loss
                (static_sparse_inputs, dense_inputs, seq_inputs, seq_length, neg_seq_inputs) with auxiliary loss
            training: Boolean indicating training mode
        Returns:
            Model output logits (before sigmoid for binary classification)
        """
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

        # Embedding Lookup
        static_sparse_emb = self.static_sparse_embedding(static_sparse_inputs)
        dense_emb = self.dense_embedding(dense_inputs)
        dense_emb = tf.reshape(dense_emb, [-1, self.dim_input_dense, self.dim_emb])
        seq_emb = self.seq_embedding(seq_inputs)
        neg_seq_emb = None
        if self.use_auxiliary_loss and neg_seq_inputs is not None:
            neg_seq_emb = self.neg_seq_embedding(neg_seq_inputs)

        # Target Item Embedding (first field of static sparse features)
        target_item_emb = static_sparse_emb[:, 0, :]

        # Interest Extraction
        if self.use_auxiliary_loss:
            extractor_inputs = [seq_emb, seq_length, neg_seq_emb]
        else:
            extractor_inputs = [seq_emb, seq_length]
        interest_states = self.interest_extractor(extractor_inputs, training=training)

        # Interest Evolution
        evolution_inputs = [interest_states, target_item_emb, seq_length]
        final_interest_emb = self.interest_evolution(
            evolution_inputs, training=training
        )

        # Feature Concatenation
        static_sparse_flat = tf.reshape(
            static_sparse_emb, [-1, static_sparse_emb.shape[1] * self.dim_emb]
        )
        dense_flat = tf.reshape(dense_emb, [-1, dense_emb.shape[1] * self.dim_emb])
        final_features = tf.concat(
            [static_sparse_flat, dense_flat, final_interest_emb], axis=-1
        )

        # Final Prediction
        logits = self.prediction_head(final_features, training=training)
        return logits
