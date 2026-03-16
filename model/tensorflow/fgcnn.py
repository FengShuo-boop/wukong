from typing import List
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, Dense, Flatten
from tensorflow.keras import backend as K
from tensorflow.python.layers import utils

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


class FGCNNLayer(Layer):
    """Feature Generation Layer used in FGCNN,including Convolution,MaxPooling and Recombination.

    Input shape
      - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size,new_feture_num,embedding_size)``.

    References
      - [Liu B, Tang R, Chen Y, et al. Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1904.04447, 2019.](https://arxiv.org/pdf/1904.04447)

    """

    def __init__(
        self,
        filters=(
            14,
            16,
        ),
        kernel_width=(
            7,
            7,
        ),
        new_maps=(
            3,
            3,
        ),
        pooling_width=(2, 2),
        **kwargs,
    ):
        if not (
            len(filters) == len(kernel_width) == len(new_maps) == len(pooling_width)
        ):
            raise ValueError("length of argument must be equal")
        self.filters = filters
        self.kernel_width = kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width

        super(FGCNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(input_shape))
            )
        self.conv_layers = []
        self.pooling_layers = []
        self.dense_layers = []
        pooling_shape = input_shape.as_list() + [
            1,
        ]
        embedding_size = int(input_shape[-1])
        for i in range(1, len(self.filters) + 1):
            filters = self.filters[i - 1]
            width = self.kernel_width[i - 1]
            new_filters = self.new_maps[i - 1]
            pooling_width = self.pooling_width[i - 1]
            conv_output_shape = self._conv_output_shape(pooling_shape, (width, 1))
            pooling_shape = self._pooling_output_shape(
                conv_output_shape, (pooling_width, 1)
            )
            self.conv_layers.append(
                Conv2D(
                    filters=filters,
                    kernel_size=(width, 1),
                    strides=(1, 1),
                    padding="same",
                    activation="tanh",
                    use_bias=True,
                )
            )
            self.pooling_layers.append(MaxPooling2D(pool_size=(pooling_width, 1)))
            self.dense_layers.append(
                Dense(
                    pooling_shape[1] * embedding_size * new_filters,
                    activation="tanh",
                    use_bias=True,
                )
            )

        self.flatten = Flatten()

        super(FGCNNLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs))
            )

        embedding_size = int(inputs.shape[-1])
        pooling_result = tf.expand_dims(inputs, axis=3)

        new_feature_list = []

        for i in range(1, len(self.filters) + 1):
            new_filters = self.new_maps[i - 1]

            conv_result = self.conv_layers[i - 1](pooling_result)

            pooling_result = self.pooling_layers[i - 1](conv_result)

            flatten_result = self.flatten(pooling_result)

            new_result = self.dense_layers[i - 1](flatten_result)

            new_feature_list.append(
                tf.reshape(
                    new_result,
                    (-1, int(pooling_result.shape[1]) * new_filters, embedding_size),
                )
            )

        new_features = tf.concat(new_feature_list, axis=1)
        return new_features

    def _conv_output_shape(self, input_shape, kernel_size):
        # channels_last
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i], kernel_size[i], padding="same", stride=1, dilation=1
            )
            new_space.append(new_dim)
        return [input_shape[0]] + new_space + [self.filters]

    def _pooling_output_shape(self, input_shape, pool_size):
        # channels_last

        rows = input_shape[1]
        cols = input_shape[2]
        rows = utils.conv_output_length(rows, pool_size[0], "valid", pool_size[0])
        cols = utils.conv_output_length(cols, pool_size[1], "valid", pool_size[1])
        return [input_shape[0], rows, cols, input_shape[3]]


class FGCNN(tf.keras.Model):
    def __init__(
        self,
        num_layers: int,
        num_sparse_embs: List[int],
        dim_input_dense: int,
        dim_emb: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.dim_emb = dim_emb

        self.blocks = [FGCNNLayer() for _ in range(num_layers)]
        self.projection_head = MLP(
            dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs
        x = self.embedding(sparse_inputs, dense_inputs)
        for block in self.blocks:
            x = block(x)
        x = tf.reduce_mean(x, axis=1)
        x = self.projection_head(x)
        return x


if __name__ == "__main__":
    import numpy as np

    BATCH_SIZE = 2
    NUM_SPARSE_EMBS = [
        1460,
        583,
        10131227,
        2202608,
        305,
        24,
        12517,
        633,
        3,
        93145,
        5683,
        8351593,
        3194,
        27,
        14992,
        5461306,
        10,
        5652,
        2173,
        4,
        7046547,
        18,
        15,
        286181,
        105,
        142572,
    ]
    DIM_INPUT_SPARSE = 26
    DIM_INPUT_DENSE = 13
    # Example usage
    model = FGCNN(
        num_layers=2,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_dense=13,
        dim_emb=128,
        num_hidden_head=2,
        dim_hidden_head=128,
        dim_output=1,
    )

    sparse_inputs = tf.constant(
        np.column_stack(
            [
                np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
                for i in range(DIM_INPUT_SPARSE)
            ]
        ).astype(np.int32)
    )
    dense_inputs = tf.constant(
        np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    )
    outputs = model((sparse_inputs, dense_inputs))
    print("Model output shape:", outputs.shape)
