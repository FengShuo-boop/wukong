import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)

from model.tensorflow.embedding import Embedding


class InnerProductLayer(Layer):
    """实现 IPNN 的两两特征内积交互"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch_size, num_fields, dim_emb)
        num_fields = inputs.shape[1]
        inner_products = []
        for i, j in itertools.combinations(range(num_fields), 2):
            # dot product: (batch, dim_emb) * (batch, dim_emb) -> (batch, 1)
            dot = tf.reduce_sum(
                inputs[:, i, :] * inputs[:, j, :], axis=1, keepdims=True
            )
            inner_products.append(dot)
        return tf.concat(inner_products, axis=1)


class FGCNNLayer(Layer):
    def __init__(self, filters, kernel_size, pooling_size, recombination_dim):
        super().__init__()
        self.conv = Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            padding="same",
            activation="tanh",
        )
        self.pool = MaxPooling2D(pool_size=(pooling_size, 1))
        self.recombination_dim = recombination_dim

    def build(self, input_shape):
        # input_shape: (batch, h, w, c)
        self.recombine_fc = Dense(self.recombination_dim, activation="tanh")
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        # Recombination: 拉平后生成新特征
        new_feat = Flatten()(x)
        new_feat = self.recombine_fc(new_feat)
        return x, new_feat


class FGCNN(tf.keras.Model):
    def __init__(
        self,
        num_sparse_embs,
        dim_input_dense,
        dim_emb,
        num_layers=2,
        filters=8,
        kernel_size=7,
        pooling_size=2,
    ):
        super().__init__()
        # 1. 双 Embedding 机制
        self.emb_generation = Embedding(num_sparse_embs, dim_emb, dim_input_dense)
        self.emb_classifier = Embedding(num_sparse_embs, dim_emb, dim_input_dense)

        # 2. Feature Generation 模块
        self.fg_blocks = []
        for _ in range(num_layers):
            self.fg_blocks.append(
                FGCNNLayer(
                    filters, kernel_size, pooling_size, recombination_dim=dim_emb * 3
                )
            )

        # 3. Deep Classifier (IPNN)
        self.inner_product = InnerProductLayer()
        self.fc_layers = tf.keras.Sequential(
            [
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs
        embs_gen = self.emb_generation(sparse_inputs, dense_inputs)
        embs_cls = self.emb_classifier(sparse_inputs, dense_inputs)

        # CNN Feature Generation
        curr_x = tf.expand_dims(embs_gen, axis=-1)  # (batch, num_fields, dim_emb, 1)
        all_new_features = []
        for block in self.fg_blocks:
            curr_x, new_feat = block(curr_x)
            all_new_features.append(new_feat)

        # 拼接所有生成的新特征
        generated_features = tf.concat(all_new_features, axis=1)
        # 将生成特征 reshape 回 (batch, N, dim_emb) 以便进行交互
        batch_size = tf.shape(generated_features)[0]
        generated_features = tf.reshape(
            generated_features, (batch_size, -1, embs_cls.shape[-1])
        )

        # 4. 拼接原始特征与生成特征
        augmented_space = tf.concat([embs_cls, generated_features], axis=1)

        # 5. Classifier: 内积 + MLP
        ip_out = self.inner_product(augmented_space)
        flat_aug = Flatten()(augmented_space)
        classifier_input = tf.concat([ip_out, flat_aug], axis=1)

        return self.fc_layers(classifier_input)


# --- 测试运行 ---
if __name__ == "__main__":
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
    DIM_INPUT_DENSE = 13
    DIM_INPUT_SPARSE = 26
    SESS_NUM = 3
    SESS_LEN = 13
    DIM = 128
    model = FGCNN(
        num_sparse_embs=NUM_SPARSE_EMBS, dim_input_dense=DIM_INPUT_DENSE, dim_emb=DIM
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
    output = model((sparse_inputs, dense_inputs))
    print("Output shape:", output.shape)  # 应为 (2, 1)
    model.summary()
