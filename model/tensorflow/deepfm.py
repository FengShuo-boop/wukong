import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List


class SparseEmbeddingWithL2(tf.keras.layers.Layer):
    def __init__(self, num_sparse_embs: List[int], dim_emb: int, l2_reg: float):
        super().__init__()
        self.embeddings = [
            tf.keras.layers.Embedding(
                input_dim=num_emb,
                output_dim=dim_emb,
                embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
            )
            for num_emb in num_sparse_embs
        ]

    def call(self, sparse_inputs: tf.Tensor) -> tf.Tensor:
        sparse_outputs = [
            embedding(sparse_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return tf.stack(sparse_outputs, axis=1)


class DeepFM(Model):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    Reference: https://arxiv.org/abs/1703.04247
    Strictly follow the paper's implementation and experimental settings
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int = 10,
        dim_input_sparse: int = 26,
        dim_input_dense: int = 13,
        dnn_hidden_units: tuple = (400, 400, 400),
        l2_reg_linear: float = 1e-5,
        l2_reg_embedding: float = 1e-5,
        l2_reg_dnn: float = 0.0,
        dropout: float = 0.5,
        use_bias: bool = True,
        dim_output: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Core configs from paper
        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.num_fields = dim_input_sparse + dim_input_dense

        # 1. Linear (First-Order) Component (Wide Part)
        self.sparse_linear_emb = SparseEmbeddingWithL2(
            num_sparse_embs=num_sparse_embs, dim_emb=1, l2_reg=l2_reg_linear
        )
        self.dense_linear = layers.Dense(
            units=dim_input_dense * 1,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_linear),
        )
        self.linear_bias = (
            self.add_weight(
                name="linear_bias", shape=(1,), initializer="zeros", trainable=True
            )
            if use_bias
            else 0.0
        )

        # 2. Shared Embedding Layer for FM Second-Order and DNN (Core of DeepFM)
        self.shared_embedding = SparseEmbeddingWithL2(
            num_sparse_embs=num_sparse_embs, dim_emb=dim_emb, l2_reg=l2_reg_embedding
        )
        self.dense_embedding = layers.Dense(
            units=dim_input_dense * dim_emb,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_embedding),
        )

        # 3. DNN Component for High-Order Feature Interactions
        self.dnn_layers = []
        for units in dnn_hidden_units:
            self.dnn_layers.append(
                layers.Dense(
                    units=units,
                    use_bias=use_bias,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
                )
            )
            self.dnn_layers.append(layers.BatchNormalization())
            self.dnn_layers.append(layers.ReLU())
            self.dnn_layers.append(layers.Dropout(dropout))
        self.dnn_output_layer = layers.Dense(
            units=dim_output,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
        )

        # For ONNX export compatibility
        self.output_names = ["output"]

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        """
        Forward pass of DeepFM
        Args:
            inputs: Tuple of (sparse_inputs, dense_inputs)
                sparse_inputs: (batch_size, dim_input_sparse), int32, categorical features
                dense_inputs: (batch_size, dim_input_dense), float32, continuous features
            training: Boolean for dropout and batch norm training mode
        Returns:
            logits: (batch_size, dim_output), raw logits for binary cross entropy loss
        """
        sparse_inputs, dense_inputs = inputs

        # -------------------------- First-Order Linear Logit --------------------------
        sparse_linear = self.sparse_linear_emb(sparse_inputs)  # (bs, num_sparse, 1)
        dense_linear = self.dense_linear(dense_inputs)  # (bs, num_dense * 1)
        dense_linear = tf.reshape(
            dense_linear, [-1, self.dim_input_dense, 1]
        )  # (bs, num_dense, 1)
        linear_out = tf.concat(
            [sparse_linear, dense_linear], axis=1
        )  # (bs, num_fields, 1)
        linear_logit = tf.reduce_sum(linear_out, axis=1) + self.linear_bias  # (bs, 1)

        # -------------------------- Shared Feature Embedding --------------------------
        # Sparse features embedding
        sparse_shared_emb = self.shared_embedding(
            sparse_inputs
        )  # (bs, num_sparse, dim_emb)
        # Dense features embedding
        dense_shared_emb = self.dense_embedding(
            dense_inputs
        )  # (bs, num_dense * dim_emb)
        dense_shared_emb = tf.reshape(
            dense_shared_emb, [-1, self.dim_input_dense, self.dim_emb]
        )  # (bs, num_dense, dim_emb)
        # Concat all field embeddings
        shared_emb = tf.concat(
            [sparse_shared_emb, dense_shared_emb], axis=1
        )  # (bs, num_fields, dim_emb)

        # -------------------------- FM Second-Order Interaction Logit --------------------------
        sum_of_emb = tf.reduce_sum(shared_emb, axis=1)  # (bs, dim_emb)
        square_of_sum = tf.square(sum_of_emb)  # (bs, dim_emb)
        sum_of_square = tf.reduce_sum(tf.square(shared_emb), axis=1)  # (bs, dim_emb)
        fm_second_order = 0.5 * (square_of_sum - sum_of_square)  # (bs, dim_emb)
        fm_logit = tf.reduce_sum(fm_second_order, axis=1, keepdims=True)  # (bs, 1)

        # -------------------------- DNN High-Order Interaction Logit --------------------------
        dnn_input = tf.reshape(
            shared_emb, [-1, self.num_fields * self.dim_emb]
        )  # (bs, num_fields * dim_emb)
        x = dnn_input
        for layer in self.dnn_layers:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        dnn_logit = self.dnn_output_layer(x)  # (bs, 1)

        # -------------------------- Final Combined Logit --------------------------
        final_logit = linear_logit + fm_logit + dnn_logit
        return final_logit
