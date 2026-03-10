from typing import List
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


# ---------------- RMSNorm ----------------
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


# ---------------- Mixed FFN (tail LNS token-specific) ----------------
class MixedFFN(Layer):
    # 尾部 min(LNS, T) token 用 token-specific，其余 shared
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
        t = tf.minimum(T, self.LNS)  # tail token-specific count
        s = T - t  # shared count

        yS = self.W2S(self.act(self.W1S(x[:, :s])))  # [B,s,D]

        xT = x[:, s:]  # [B,t,D]
        W1 = self.W1NS[-t:]
        W2 = self.W2NS[-t:]
        h = self.act(tf.einsum("btd,tde->bte", xT, W1))
        yT = tf.einsum("btd,tde->bte", h, W2)  # [B,t,D]

        return tf.concat([yS, yT], axis=1)


# ---------------- Pyramid Mixed Causal Attention (Eq.14 strict) ----------------
class PyramidMixedCausalAttention(Layer):
    """
    Eq.(14) strict:
      - Q from tail set (length Lq)
      - K/V from full sequence (length L)
      - output keeps only tail (length Lq)
    Mixed parameterization:
      - tail min(L, LNS) tokens use token-specific
      - earlier tokens use shared
    """

    def __init__(self, dim_emb, num_heads, LNS, **kwargs):
        super().__init__(**kwargs)
        assert dim_emb % num_heads == 0
        self.D, self.H, self.dh = dim_emb, num_heads, dim_emb // num_heads
        self.LNS = LNS

        self.WqS = Dense(dim_emb, use_bias=False)
        self.WkS = Dense(dim_emb, use_bias=False)
        self.WvS = Dense(dim_emb, use_bias=False)

        init = tf.keras.initializers.GlorotUniform()
        self.WqNS = self.add_weight("WqNS", (LNS, dim_emb, dim_emb), initializer=init)
        self.WkNS = self.add_weight("WkNS", (LNS, dim_emb, dim_emb), initializer=init)
        self.WvNS = self.add_weight("WvNS", (LNS, dim_emb, dim_emb), initializer=init)

        self.Wo = Dense(dim_emb, use_bias=False)

    def _mh(self, x):  # [B,T,D] -> [B,H,T,dh]
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        return tf.transpose(tf.reshape(x, [b, t, self.H, self.dh]), [0, 2, 1, 3])

    def _unmh(self, x):  # [B,H,T,dh] -> [B,T,D]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.D])

    def call(self, x, Lq):
        L = tf.shape(x)[1]
        t = tf.minimum(L, self.LNS)  # tail token-specific
        s = L - t  # head shared

        xS, xT = x[:, :s], x[:, s:]

        Q = tf.concat([self.WqS(xS), tf.einsum("btd,tde->bte", xT, self.WqNS[-t:])], 1)
        K = tf.concat([self.WkS(xS), tf.einsum("btd,tde->bte", xT, self.WkNS[-t:])], 1)
        V = tf.concat([self.WvS(xS), tf.einsum("btd,tde->bte", xT, self.WvNS[-t:])], 1)

        Q = Q[:, -Lq:]  # only tail queries (all tokens in Q will be updated)

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)
        logits = tf.matmul(Qh, Kh, transpose_b=True) * (
            tf.cast(self.dh, tf.float32) ** -0.5
        )

        # causal mask for tail queries: absolute indices [L-Lq .. L-1]
        q = tf.range(L - Lq, L)[:, None]
        k = tf.range(L)[None, :]
        logits += tf.cast(k > q, tf.float32)[None, None] * (-1e9)

        out = tf.matmul(tf.nn.softmax(logits, -1), Vh)  # [B,H,Lq,dh]
        return self.Wo(self._unmh(out))  # [B,Lq,D]


# ---------------- OneTrans Block (auto Lq=L-1) ----------------
class OneTransBlock(Layer):
    def __init__(
        self, dim_emb, num_heads, d_ff, LNS, ln_eps=1e-5, bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.ln1, self.ln2 = RMSLayerNorm(ln_eps), RMSLayerNorm(ln_eps)
        self.mha = PyramidMixedCausalAttention(dim_emb, num_heads, LNS)
        self.ffn = MixedFFN(dim_emb, d_ff, LNS, bias=bias)

    def call(self, x, Lq):
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]  # residual对齐尾部
        return self.ffn(self.ln2(z)) + z


# ---------------- Stack: compress S for LS layers ----------------


class OneTrans(Layer):
    def __init__(
        self,
        LS,
        LNS,
        dim_emb,
        num_heads,
        d_ff,
        num_sparse_embs: List[int],
        dim_input_dense: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.Lq_list = list(range(LS + LNS, LNS, -4))  # LS+LNS .. LNS+1
        self.Lq_list.append(LNS)
        self.blocks = [
            OneTransBlock(dim_emb, num_heads, d_ff, LNS=LNS, bias=bias)
            for _ in range(len(self.Lq_list))
        ]  # finally LNS.

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

        for blk, Lq_py in zip(self.blocks, self.Lq_list):
            x = blk(x, Lq_py)
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
    model = OneTrans(
        LS=16,
        LNS=16,
        dim_emb=128,
        num_heads=4,
        d_ff=256,
        num_hidden_head=256,
        dim_hidden_head=128,
        dim_output=1,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_dense=13,
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
    print("Sparse input shape:", sparse_inputs.shape)
    print("Dense input shape:", dense_inputs.shape)
    outputs = model((sparse_inputs, dense_inputs))
    print("Model output shape:", outputs.shape)
