from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import backend as K

from model.tensorflow.embedding import Embedding, SparseEmbedding


class FwFMLayer(Layer):
    """Field-weighted Factorization Machines (Interaction Terms only)

    计算 FwFM 的纯二阶交叉项。

    Input shape
      - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.

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
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(input_shape))
            )

        if input_shape[1] != self.num_fields:
            raise ValueError(
                "Mismatch in number of fields {} and concatenated embeddings dims {}".format(
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

        # 1. 批量计算所有域特征两两之间的点积
        inner_products = tf.matmul(inputs, inputs, transpose_b=True)

        # 2. 创建上三角掩码 (等效于 itertools.combinations i < j)
        ones = tf.ones((self.num_fields, self.num_fields), dtype=inputs.dtype)
        mask = tf.linalg.band_part(ones, 0, -1) - tf.linalg.band_part(ones, 0, 0)

        # 3. 将权重矩阵应用掩码
        r_ij = self.field_strengths * mask

        # 4. 点积结果乘以对应的域对权重 r_{F(i), F(j)}
        weighted_products = inner_products * r_ij

        # 5. 求和，输出维度降至 (batch_size, 1)
        sum_ = tf.reduce_sum(weighted_products, axis=[1, 2])
        out = tf.expand_dims(sum_, axis=-1)

        return out


class FwFM(Layer):
    """
    完整的 FwFM 模型结构，严格对齐论文：
    Prediction = Bias + Linear_Terms + FwFM_Interaction_Terms
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense

        # 二阶交叉项组件
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense)
        self.interaction_layer = FwFMLayer(
            num_fields=dim_input_dense + dim_input_sparse
        )

        # 一阶线性项组件 (FwFMs_LW 形式)
        self.linear_sparse = SparseEmbedding(num_sparse_embs, dim_emb=1)
        self.linear_dense = tf.keras.layers.Dense(units=1, use_bias=False)

        # 全局偏置组件
        self.global_bias = self.add_weight(
            name="global_bias", shape=(1,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs

        # 1. 计算二阶交叉项 (Interaction terms)
        # emb_x shape: (batch_size, num_fields, dim_emb)
        emb_x = self.embedding(sparse_inputs, dense_inputs)
        # interaction_term shape: (batch_size, 1)
        interaction_term = self.interaction_layer(emb_x)

        # 2. 计算一阶线性项 (Linear terms)
        # 稀疏特征的一阶输出 shape: (batch_size, num_sparse_fields, 1) -> sum 为 (batch_size, 1)
        sparse_linear = tf.reduce_sum(self.linear_sparse(sparse_inputs), axis=1)
        # 稠密特征的一阶输出 shape: (batch_size, 1)
        dense_linear = self.linear_dense(dense_inputs)
        linear_term = sparse_linear + dense_linear

        # 3. 汇总合并输出
        out = self.global_bias + linear_term + interaction_term
        return out


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
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=DIM_INPUT_SPARSE,
        dim_input_dense=DIM_INPUT_DENSE,
        dim_emb=128,
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
