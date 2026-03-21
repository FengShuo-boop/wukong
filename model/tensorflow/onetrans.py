from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np

from model.tensorflow.mlp import MLP


# ---------------- Criteo Tokenizer ----------------
class CriteoTokenizer(Layer):
    """专门为 Criteo 39个特征设计的 Tokenizer"""

    def __init__(
        self, num_sparse_embs: List[int], dim_emb: int, bias: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_emb = dim_emb
        self.sparse_embeddings = [
            tf.keras.layers.Embedding(input_dim=num, output_dim=dim_emb)
            for num in num_sparse_embs
        ]
        # Criteo 的 13 个稠密特征，通过 Dense 映射到 13 * dim_emb
        self.dense_embedding = Dense(units=13 * dim_emb, use_bias=bias)

    def call(self, sparse_inputs, dense_inputs):
        # 1. Sparse 特征 -> 26 个 tokens
        sparse_outs = [
            emb(sparse_inputs[:, i]) for i, emb in enumerate(self.sparse_embeddings)
        ]
        sparse_tokens = tf.stack(sparse_outs, axis=1)  # (B, 26, D)

        # 2. Dense 特征 -> 13 个 tokens
        dense_outs = self.dense_embedding(dense_inputs)  # (B, 13 * D)
        dense_tokens = tf.reshape(dense_outs, [-1, 13, self.dim_emb])  # (B, 13, D)

        # 3. 拼接得到 39 个 tokens
        return tf.concat([sparse_tokens, dense_tokens], axis=1)  # (B, 39, D)


class PositionalEmbedding(Layer):
    """位置编码（为伪序列特征提供字段位置信息）"""

    def __init__(self, max_len: int, dim_emb: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.dim_emb = dim_emb

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.max_len, self.dim_emb),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        T = tf.shape(x)[1]
        return x + self.pos_emb[:, :T, :]


class RMSLayerNorm(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            "scale", shape=(input_shape[-1],), initializer="ones"
        )

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale


# ---------------- OneTrans Core Components ----------------
class MixedFFN(Layer):
    def __init__(self, dim_emb, d_ff, LNS, activation="gelu", bias=False, **kwargs):
        super().__init__(**kwargs)
        self.LNS = LNS
        self.W1S, self.W2S = Dense(d_ff, use_bias=bias), Dense(dim_emb, use_bias=bias)
        init = tf.keras.initializers.GlorotUniform()
        self.W1NS = self.add_weight("W1NS", (LNS, dim_emb, d_ff), initializer=init)
        self.W2NS = self.add_weight("W2NS", (LNS, d_ff, dim_emb), initializer=init)
        self.act = tf.keras.activations.get(activation)

    def call(self, x):
        T = tf.shape(x)[1]
        t = tf.minimum(T, self.LNS)
        s = T - t
        yS = self.W2S(self.act(self.W1S(x[:, :s])))
        xT = x[:, s:]
        W1, W2 = self.W1NS[-t:], self.W2NS[-t:]
        h = self.act(tf.einsum("btd,tde->bte", xT, W1))
        yT = tf.einsum("btd,tde->bte", h, W2)
        return tf.concat([yS, yT], axis=1)


class PyramidMixedCausalAttention(Layer):
    def __init__(self, dim_emb, num_heads, LNS, **kwargs):
        super().__init__(**kwargs)
        self.D, self.H, self.dh = dim_emb, num_heads, dim_emb // num_heads
        self.LNS = LNS
        self.WqS, self.WkS, self.WvS = (
            Dense(dim_emb, use_bias=False),
            Dense(dim_emb, use_bias=False),
            Dense(dim_emb, use_bias=False),
        )
        init = tf.keras.initializers.GlorotUniform()
        self.WqNS = self.add_weight("WqNS", (LNS, dim_emb, dim_emb), initializer=init)
        self.WkNS = self.add_weight("WkNS", (LNS, dim_emb, dim_emb), initializer=init)
        self.WvNS = self.add_weight("WvNS", (LNS, dim_emb, dim_emb), initializer=init)
        self.Wo = Dense(dim_emb, use_bias=False)

    def _mh(self, x):
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, [b, t, self.H, self.dh]), [0, 2, 1, 3])

    def _unmh(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.D])

    def call(self, x, Lq):
        L = tf.shape(x)[1]
        t = tf.minimum(L, self.LNS)
        s = L - t
        xS, xT = x[:, :s], x[:, s:]

        Q = tf.concat([self.WqS(xS), tf.einsum("btd,tde->bte", xT, self.WqNS[-t:])], 1)
        K = tf.concat([self.WkS(xS), tf.einsum("btd,tde->bte", xT, self.WkNS[-t:])], 1)
        V = tf.concat([self.WvS(xS), tf.einsum("btd,tde->bte", xT, self.WvNS[-t:])], 1)

        Q = Q[:, -Lq:]  # Query 仅保留尾部 Lq 个 tokens

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)
        logits = tf.matmul(Qh, Kh, transpose_b=True) * (
            tf.cast(self.dh, tf.float32) ** -0.5
        )

        # Causal mask alignment
        q_idx = tf.range(L - Lq, L)[:, None]
        k_idx = tf.range(L)[None, :]
        logits += tf.cast(k_idx > q_idx, tf.float32)[None, None] * (-1e9)

        out = tf.matmul(tf.nn.softmax(logits, -1), Vh)
        return self.Wo(self._unmh(out))


class OneTransBlock(Layer):
    def __init__(
        self, dim_emb, num_heads, d_ff, LNS, ln_eps=1e-5, bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.ln1, self.ln2 = RMSLayerNorm(ln_eps), RMSLayerNorm(ln_eps)
        self.mha = PyramidMixedCausalAttention(dim_emb, num_heads, LNS)
        self.ffn = MixedFFN(dim_emb, d_ff, LNS, bias=bias)

    def call(self, x, Lq):
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]
        return self.ffn(self.ln2(z)) + z


# ---------------- Model Level ----------------
class OneTrans(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        LS,
        LNS,
        dim_emb,
        num_heads,
        d_ff,
        num_sparse_embs: List[int],
        num_hidden_head: int,
        dim_hidden_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.LNS = LNS
        self.LS = LS
        self.tokenizer = CriteoTokenizer(num_sparse_embs, dim_emb, bias)
        self.pos_embedding = PositionalEmbedding(max_len=LS + LNS, dim_emb=dim_emb)

        # 修复: 按照论文进行线性衰减生成 Pyramid Lq_list (适应 Criteo 的较短长度)
        # 从 LS 线性衰减到 0，然后加上 LNS
        targets = np.linspace(LS, 0, num_layers + 1)[1:]
        self.Lq_list = []
        for t in targets[:-1]:
            # Criteo 长度极短，不做 32 倍数对齐，直接使用 round 后的整数
            self.Lq_list.append(int(np.round(t)) + LNS)
        self.Lq_list.append(LNS)  # 顶层只保留 NS-token

        self.blocks = [
            OneTransBlock(dim_emb, num_heads, d_ff, LNS=LNS, bias=bias)
            for _ in range(num_layers)
        ]

        # 仅保留单 CTR 预测头
        self.ctr_head = MLP(-1, num_hidden_head, dim_hidden_head, 1, dropout, bias)

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs

        # 1. 提取所有 Criteo 特征并对齐到 39 个 tokens
        x = self.tokenizer(sparse_inputs, dense_inputs)

        # 2. 加入位置编码
        x = self.pos_embedding(x)

        # 3. 逐层进行金字塔截断压缩
        for blk, Lq_py in zip(self.blocks, self.Lq_list):
            x = blk(x, Lq_py)

        # 4. Pooling 层 (x 的长度此时已被压缩至 self.LNS)
        x_pooled = tf.reduce_mean(x, axis=1)

        # 5. 单 CTR 预测输出
        p_ctr = tf.nn.sigmoid(self.ctr_head(x_pooled))
        return p_ctr


# ---------------- Test Run for Criteo ----------------
if __name__ == "__main__":
    BATCH_SIZE = 4

    # Criteo 数据格式定义
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

    # 由于 Criteo 总特征数为 26(sparse) + 13(dense) = 39
    # 我们人为将其切分为 LS = 23 (伪S-token) 和 LNS = 16 (NS-token)
    model = OneTrans(
        num_layers=3,  # Criteo特征长度较短，层数建议设小一点
        LS=23,
        LNS=16,
        dim_emb=128,
        num_heads=4,
        d_ff=256,
        num_sparse_embs=NUM_SPARSE_EMBS,
        num_hidden_head=2,
        dim_hidden_head=128,
    )

    # 构造 Mock Data
    sparse_inputs = tf.constant(
        np.column_stack(
            [
                np.random.randint(0, min(1000, NUM_SPARSE_EMBS[i]), size=BATCH_SIZE)
                for i in range(DIM_INPUT_SPARSE)
            ]
        ).astype(np.int32)
    )
    dense_inputs = tf.constant(
        np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    )

    print(f"Sparse Input Shape: {sparse_inputs.shape}")
    print(f"Dense Input Shape: {dense_inputs.shape}")
    print(f"Pyramid Schedule (Lq_list): {model.Lq_list}\n")

    # 执行图模型
    outputs = model((sparse_inputs, dense_inputs))
    print(f"CTR Task Output Shape: {outputs.shape}")
