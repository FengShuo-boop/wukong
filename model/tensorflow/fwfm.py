from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import backend as K

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


class FwFMLayer(Layer):
    """Field-weighted Factorization Machines

    Optimized Vectorized Implementation.

    Input shape
      - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, 1)``.
    """

    def __init__(self, num_fields=4, regularizer=0.000001, **kwargs):
        self.num_fields = num_fields
        self.regularizer = regularizer
        super(FwFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions"
                % (len(input_shape))
            )

        if input_shape[1] != self.num_fields:
            raise ValueError(
                "Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(
                    self.num_fields, input_shape[1]
                )
            )

        self.field_strengths = self.add_weight(
            name="field_pair_strengths",
            shape=(self.num_fields, self.num_fields),
            initializer=TruncatedNormal(),
            regularizer=l2(self.regularizer),
            trainable=True,
        )

        super(FwFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs))
            )

        if inputs.shape[1] != self.num_fields:
            raise ValueError(
                "Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(
                    self.num_fields, inputs.shape[1]
                )
            )

        # 1. 批量计算所有域特征两两之间的点积 (Batch Matrix Multiplication)
        # inputs shape: (batch_size, num_fields, embedding_size)
        # inner_products shape: (batch_size, num_fields, num_fields)
        inner_products = tf.matmul(inputs, inputs, transpose_b=True)

        # 2. 创建一个严格的"上三角掩码" (Upper triangular mask)
        # 这是为了等效替代 itertools.combinations (仅保留 i < j 的组合)
        ones = tf.ones((self.num_fields, self.num_fields), dtype=inputs.dtype)
        mask = tf.linalg.band_part(ones, 0, -1) - tf.linalg.band_part(ones, 0, 0)

        # 3. 将权重矩阵应用掩码，只保留右上角的权重 (i < j 的 r_ij)
        # r_ij shape: (num_fields, num_fields)
        r_ij = self.field_strengths * mask

        # 4. 用两两点积结果乘上对应的域权重 (利用 Broadcasting 机制)
        weighted_products = inner_products * r_ij

        # 5. 将所有交叉结果求和
        # 对 num_fields 所在的维度(axis=1, 2)进行累加，保持 keepdims=True 使得输出为 (batch_size, 1)
        sum_ = tf.reduce_sum(weighted_products, axis=[1, 2])
        out = tf.expand_dims(sum_, axis=-1)

        return out


class FwFM(Layer):
    def __init__(
        self,
        num_layers: int,
        num_sparse_embs: List[int],
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                MLP(
                    dim_emb,
                    2,
                    dim_emb,
                    dim_emb,
                    dropout,
                    bias,
                )
            )
        self.projection_head = FwFMLayer(dim_input_dense + dim_input_sparse, 0.00001)

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs
        x = self.embedding(sparse_inputs, dense_inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.projection_head(x)
        return x


if __name__ == "__main__":
    import numpy as np

    BATCH_SIZE = 4096
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
    model = FwFM(
        num_layers=2,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=DIM_INPUT_SPARSE,
        dim_input_dense=DIM_INPUT_DENSE,
        dim_emb=128,
        dropout=0.2,
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
    inputs = (sparse_inputs, dense_inputs)
    outputs = model(inputs)
    print(outputs.shape)
