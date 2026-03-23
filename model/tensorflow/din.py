# -*- coding: utf-8 -*-
"""
Deep Interest Network (DIN) for Click-Through Rate Prediction.

Reference:
    Zhou G, et al. Deep interest network for click-through rate prediction.
    KDD 2018. https://arxiv.org/abs/1706.06978
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Tuple

from model.tensorflow.embedding import Embedding


# ──────────────────────────────────────────────────────────────────────────────
#  Dice Activation
# ──────────────────────────────────────────────────────────────────────────────


class Dice(layers.Layer):
    """
    Data Adaptive Activation Function.

    Generalizes PReLU by adaptively adjusting the rectified point
    according to the distribution of inputs:

        f(s) = p(s) * s + (1 - p(s)) * alpha * s
        p(s) = sigmoid((s - E[s]) / sqrt(Var[s] + eps))

    During training:  E[s], Var[s] from current mini-batch.
    During inference: exponential moving averages of E[s], Var[s].

    Supports both 2-D inputs (bs, units) and 3-D inputs (bs, seq, units).
    """

    def __init__(self, epsilon: float = 1e-8, momentum: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        # Learnable negative-slope parameter (initialised to 0 → identity at start)
        self.alpha = self.add_weight(
            name="alpha", shape=(feat_dim,), initializer="zeros", trainable=True
        )
        # Moving-average statistics (non-trainable, updated in call)
        self.moving_mean = self.add_weight(
            name="moving_mean", shape=(feat_dim,), initializer="zeros", trainable=False
        )
        self.moving_var = self.add_weight(
            name="moving_var", shape=(feat_dim,), initializer="ones", trainable=False
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Flatten to 2-D to compute per-feature statistics
        feat_dim = tf.shape(inputs)[-1]
        flat = tf.reshape(inputs, [-1, feat_dim])  # (N, feat_dim)

        if training:
            batch_mean = tf.reduce_mean(flat, axis=0)
            batch_var = tf.math.reduce_variance(flat, axis=0)
            # Exponential moving average update
            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1.0 - self.momentum) * batch_mean
            )
            self.moving_var.assign(
                self.momentum * self.moving_var + (1.0 - self.momentum) * batch_var
            )
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.moving_mean, self.moving_var

        s_norm = (flat - mean) / tf.sqrt(var + self.epsilon)
        p = tf.sigmoid(s_norm)  # control function
        out = p * flat + (1.0 - p) * self.alpha * flat  # Dice output

        return tf.reshape(out, tf.shape(inputs))  # restore original shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon, "momentum": self.momentum})
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
#  Local Activation Unit
# ──────────────────────────────────────────────────────────────────────────────


class LocalActivationUnit(layers.Layer):
    """
    Computes per-behavior attention weights given the target item embedding.

    Attention MLP input (per paper, Eq. 3 and Figure 2):
        concat([key_j, query, key_j * query])   shape: (bs, seq_len, 3 * dim_emb)

    Key design choice: NO softmax on outputs — raw weights preserve the
    *intensity* of user interest (sum_i w_i approximates total interest
    level), unlike standard attention that always sums to 1.
    """

    def __init__(
        self,
        att_hidden_units: Tuple[int, ...] = (80, 40),
        att_activation: str = "dice",
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Build attention MLP as a tf.keras.Sequential for clean weight tracking
        att_layers: List[layers.Layer] = []
        for i, units in enumerate(att_hidden_units):
            att_layers.append(
                layers.Dense(units, use_bias=use_bias, name=f"att_fc_{i}")
            )
            if att_activation == "dice":
                att_layers.append(Dice(name=f"att_dice_{i}"))
            elif att_activation == "prelu":
                att_layers.append(layers.PReLU(name=f"att_prelu_{i}"))
            else:
                att_layers.append(
                    layers.Activation(att_activation, name=f"att_act_{i}")
                )

        # Final linear layer → scalar weight per behaviour
        att_layers.append(layers.Dense(1, use_bias=use_bias, name="att_out"))
        self.att_net = tf.keras.Sequential(att_layers, name="att_net")

    def call(
        self,
        query: tf.Tensor,  # (bs, dim_emb)
        keys: tf.Tensor,  # (bs, seq_len, dim_emb)
        training: bool = False,
    ) -> tf.Tensor:  # (bs, seq_len, 1)
        seq_len = tf.shape(keys)[1]

        # Broadcast query over the sequence dimension
        query_exp = tf.tile(tf.expand_dims(query, axis=1), [1, seq_len, 1])
        # (bs, seq_len, dim_emb)

        # Element-wise interaction (outer product approximation)
        interaction = query_exp * keys  # (bs, seq_len, dim_emb)

        # Concatenate: [key, query, key*query]  → (bs, seq_len, 3*dim_emb)
        att_input = tf.concat([keys, query_exp, interaction], axis=-1)

        # Attention MLP: (bs, seq_len, 3*dim_emb) → (bs, seq_len, 1)
        weights = self.att_net(att_input, training=training)

        return weights  # un-normalised (no softmax, per paper Section 4.3)


# ──────────────────────────────────────────────────────────────────────────────
#  DIN Model
# ──────────────────────────────────────────────────────────────────────────────


class DIN(Model):
    """
    Deep Interest Network for CTR prediction on Criteo Kaggle dataset.

    Args:
        num_sparse_embs:    List of vocabulary sizes for each sparse feature.
        dim_emb:            Embedding dimension D for all features.
        dim_input_sparse:   Number of sparse input features (26 for Criteo).
        dim_input_dense:    Number of dense input features (13 for Criteo).
        num_target_features:
            Number of leading sparse features treated as "target item"
            (analogous to goods_id / shop_id / cate_id in the paper).
            The remaining (dim_input_sparse - num_target_features) features
            are treated as "user behavior history".
        att_hidden_units:   Hidden layer sizes for the attention MLP.
        att_activation:     Activation in attention MLP ('dice'|'prelu'|'relu').
        dnn_hidden_units:   Hidden layer sizes for the final DNN.
        dnn_activation:     Activation in final DNN ('dice'|'prelu'|'relu').
        dnn_use_bn:         Whether to add BatchNorm before each activation in DNN.
        dropout:            Dropout rate in DNN.
        bias:               Whether to use bias in Dense layers.
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_target_features: int = 3,
        att_hidden_units: Tuple[int, ...] = (80, 40),
        att_activation: str = "dice",
        dnn_hidden_units: Tuple[int, ...] = (256, 128, 64),
        dnn_activation: str = "dice",
        dnn_use_bn: bool = False,
        dropout: float = 0.5,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert 1 <= num_target_features < dim_input_sparse, (
            f"num_target_features ({num_target_features}) must be in "
            f"[1, dim_input_sparse-1] = [1, {dim_input_sparse - 1}]"
        )

        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.num_target_features = num_target_features
        self.num_behavior_features = dim_input_sparse - num_target_features

        # ── Shared embedding layer (identical interface to Wukong) ──────────
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)

        # ── Local activation unit ────────────────────────────────────────────
        self.attention = LocalActivationUnit(
            att_hidden_units, att_activation, bias, name="local_activation_unit"
        )

        # ── Final DNN ────────────────────────────────────────────────────────
        # DNN input dimension:
        #   target_flat   : num_target_features * dim_emb
        #   behavior_pool : dim_emb   (attention-weighted sum)
        #   dense_flat    : dim_input_dense * dim_emb
        dnn_layers: List[layers.Layer] = []
        for i, units in enumerate(dnn_hidden_units):
            dnn_layers.append(layers.Dense(units, use_bias=bias, name=f"dnn_fc_{i}"))
            if dnn_use_bn:
                dnn_layers.append(layers.BatchNormalization(name=f"dnn_bn_{i}"))
            if dnn_activation == "dice":
                dnn_layers.append(Dice(name=f"dnn_dice_{i}"))
            elif dnn_activation == "prelu":
                dnn_layers.append(layers.PReLU(name=f"dnn_prelu_{i}"))
            else:
                dnn_layers.append(
                    layers.Activation(dnn_activation, name=f"dnn_act_{i}")
                )
            if dropout > 0.0:
                dnn_layers.append(layers.Dropout(dropout, name=f"dnn_drop_{i}"))

        self.dnn = tf.keras.Sequential(dnn_layers, name="dnn")

        # Final linear projection → logit (no activation; BCE(from_logits=True))
        self.output_layer = layers.Dense(1, use_bias=False, name="logit")

        # Compatibility marker for ONNX export (matches Wukong convention)
        self.output_names = ["output"]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        """
        inputs : (sparse_inputs, dense_inputs)
            sparse_inputs : (bs, dim_input_sparse)  int32
            dense_inputs  : (bs, dim_input_dense)   float32
        Returns: (bs, 1)  logit (before sigmoid)
        """
        sparse_inputs, dense_inputs = inputs

        # ── 1. Embedding ──────────────────────────────────────────────────
        # all_embs: (bs, dim_input_sparse + dim_input_dense, dim_emb)
        all_embs = self.embedding(sparse_inputs, dense_inputs)

        sparse_embs = all_embs[:, : self.dim_input_sparse, :]  # (bs, 26, D)
        dense_embs = all_embs[:, self.dim_input_sparse :, :]  # (bs, 13, D)

        # ── 2. Split target vs. behavior ──────────────────────────────────
        target_embs = sparse_embs[:, : self.num_target_features, :]  # (bs, Nt, D)
        behavior_embs = sparse_embs[:, self.num_target_features :, :]  # (bs, Nb, D)

        # ── 3. Local activation unit ──────────────────────────────────────
        # query: mean of target feature embeddings  → (bs, D)
        # (aggregates multi-field "ad" representation into one vector)
        query = tf.reduce_mean(target_embs, axis=1)  # (bs, D)

        # att_weights: (bs, Nb, 1)  — un-normalised interest weights
        att_weights = self.attention(query, behavior_embs, training=training)

        # Weighted sum pooling: vU(A) = sum_j w_j * e_j  (Eq. 3)
        behavior_pooled = tf.reduce_sum(behavior_embs * att_weights, axis=1)
        # (bs, D)

        # ── 4. Concatenate DNN input ──────────────────────────────────────
        target_flat = tf.reshape(
            target_embs, [-1, self.num_target_features * self.dim_emb]
        )
        dense_flat = tf.reshape(dense_embs, [-1, self.dim_input_dense * self.dim_emb])
        dnn_input = tf.concat([target_flat, behavior_pooled, dense_flat], axis=-1)
        # shape: (bs, (Nt + 1 + 13) * D)

        # ── 5. DNN → logit ────────────────────────────────────────────────
        dnn_out = self.dnn(dnn_input, training=training)  # (bs, 64)
        logit = self.output_layer(dnn_out)  # (bs, 1)

        return logit


if __name__ == "__main__":
    # Example usage for DIN
    batch_size = 4
    num_sparse_embs = [100, 200, 150, 300, 50, 80]
    dim_emb = 16
    dim_input_sparse = len(num_sparse_embs)  # 6
    dim_input_dense = 5
    num_target_features = 2  # first 2 sparse → "target ad"

    model = DIN(
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        num_target_features=num_target_features,
        att_hidden_units=(80, 40),
        att_activation="dice",
        dnn_hidden_units=(256, 128, 64),
        dnn_activation="dice",
        dnn_use_bn=False,
        dropout=0.5,
        bias=False,
    )

    # Dummy input data
    sparse_inputs = tf.random.uniform(
        (batch_size, dim_input_sparse), minval=0, maxval=50, dtype=tf.int32
    )
    dense_inputs = tf.random.normal((batch_size, dim_input_dense))

    logits = model((sparse_inputs, dense_inputs), training=False)
    print("DIN logits shape:", logits.shape)  # (4, 1)
