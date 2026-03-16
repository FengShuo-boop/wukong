import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List
from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP

class CIN(layers.Layer):
    """
    Compressed Interaction Network for xDeepFM
    Models explicit high-order feature interactions
    """
    def __init__(
        self,
        cin_layer_size: list,
        cin_split_half: bool = True,
        cin_activation: str = None,
        l2_reg: float = 0.0,
        seed: int = 42,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.cin_layer_size = cin_layer_size
        self.cin_split_half = cin_split_half
        self.cin_activation = tf.keras.activations.get(cin_activation) if cin_activation else None
        self.l2_reg = l2_reg
        self.seed = seed
        self.conv_layers = []
        self.split_len = None

    def build(self, input_shape):
        num_fields = input_shape[1]
        emb_dim = input_shape[2]
        prev_fm = num_fields

        for idx, fm_size in enumerate(self.cin_layer_size):
            if self.cin_split_half and idx != len(self.cin_layer_size) - 1:
                cur_fm = fm_size // 2
                self.split_len = cur_fm
            else:
                cur_fm = fm_size
                self.split_len = fm_size

            conv = layers.Conv1D(
                filters=fm_size,
                kernel_size=1,
                strides=1,
                padding='valid',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + idx)
            )
            self.conv_layers.append(conv)
            prev_fm = cur_fm
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        emb_dim = inputs.shape[2]
        cin_outputs = []
        x0 = inputs
        xk = x0

        for idx, conv in enumerate(self.conv_layers):
            # Step 1: Outer product via einsum (batch, H, D) @ (batch, M, D) -> (batch, H, M, D)
            outer_product = tf.einsum('bhd,bmd->bhmd', xk, x0)
            
            # Step 2: Rearrange dimensions to (batch, D, H*M)
            # We want to apply convolution along the interaction axis, keeping D as channels
            outer_product = tf.transpose(outer_product, [0, 3, 1, 2])  # (batch, D, H, M)
            outer_product = tf.reshape(outer_product, [batch_size, emb_dim, -1])  # (batch, D, H*M)
            
            # Step 3: Apply Conv1D (filters act on the last dimension)
            conv_output = conv(outer_product)  # Output shape: (batch, D, fm_size)
            
            # Step 4: Transpose back to (batch, fm_size, D) for next iteration
            conv_output = tf.transpose(conv_output, [0, 2, 1])
            
            if self.cin_activation is not None:
                conv_output = self.cin_activation(conv_output)
            
            if self.cin_split_half and idx != len(self.cin_layer_size) - 1:
                xk = conv_output[:, :self.split_len, :]
                cin_outputs.append(conv_output[:, self.split_len:, :])
            else:
                xk = conv_output
                cin_outputs.append(conv_output)

        cin_outputs = tf.concat(cin_outputs, axis=1)
        cin_outputs = tf.reduce_sum(cin_outputs, axis=-1)
        return cin_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "cin_layer_size": self.cin_layer_size,
            "cin_split_half": self.cin_split_half,
            "cin_activation": tf.keras.activations.serialize(self.cin_activation) if self.cin_activation else None,
            "l2_reg": self.l2_reg,
            "seed": self.seed
        })
        return config

class XDeepFM(Model):
    """
    xDeepFM Model: Linear + CIN + DNN
    Input: (sparse_inputs, dense_inputs)
    Output: binary classification logits
    """
    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        cin_layer_size: list,
        cin_split_half: bool = True,
        cin_activation: str = None,
        l2_reg_linear: float = 0.0,
        l2_reg_cin: float = 0.0,
        l2_reg_dnn: float = 0.0,
        dnn_hidden_units: list = (256, 128, 64),
        dnn_dropout: float = 0.0,
        dim_output: int = 1,
        bias: bool = False,
        seed: int = 42,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.seed = seed

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.num_fields = dim_input_sparse + dim_input_dense

        self.linear = layers.Dense(
            units=1,
            use_bias=bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_linear),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.flatten = layers.Flatten()

        self.cin = CIN(
            cin_layer_size=cin_layer_size,
            cin_split_half=cin_split_half,
            cin_activation=cin_activation,
            l2_reg=l2_reg_cin,
            seed=seed
        )
        
        # Added: Separate projection layer for CIN output
        self.cin_proj = layers.Dense(
            units=1,
            use_bias=bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_cin),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+2)
        )

        self.dnn = MLP(
            dim_in=self.num_fields * dim_emb,
            num_hidden=len(dnn_hidden_units),
            dim_hidden=dnn_hidden_units[0] if dnn_hidden_units else 0,
            dim_out=dnn_hidden_units[-1] if dnn_hidden_units else dim_output,
            dropout=dnn_dropout,
            bias=bias
        )
        self.dnn_out = layers.Dense(
            units=1,
            use_bias=bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1)
        )

    def call(self, inputs) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs
        emb_output = self.embedding(sparse_inputs, dense_inputs)
        emb_flatten = self.flatten(emb_output)

        linear_logit = self.linear(emb_flatten)
        
        # CIN path
        cin_logit = 0.0
        if self.cin.cin_layer_size:
            cin_output = self.cin(emb_output)
            cin_logit = self.cin_proj(cin_output)
        
        # DNN path
        dnn_output = self.dnn(emb_flatten)
        dnn_logit = self.dnn_out(dnn_output)

        final_logit = linear_logit + cin_logit + dnn_logit
        return final_logit