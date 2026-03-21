from typing import List
import tensorflow as tf
from tensorflow.keras import layers

from model.tensorflow.mlp import MLP
from model.tensorflow.embedding import Embedding


class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.dim,), initializer="ones", trainable=True
        )

    def call(self, x):
        # norm = mean(x^2)
        norm = tf.reduce_mean(tf.pow(x, 2), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(norm + self.eps)
        return self.scale * x


class PertokenSwiGLU(layers.Layer):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01, bias=False, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = layers.Dense(hidden_dim, use_bias=bias)
        self.fc_gate = layers.Dense(hidden_dim, use_bias=bias)
        self.fc_down = layers.Dense(
            dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=down_scale, mode="fan_avg", distribution="uniform"
            ),
            use_bias=bias,
        )

    def call(self, x):
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        # Swish = x * sigmoid(x)
        gate = tf.nn.sigmoid(gate_logits) * gate_logits
        return self.fc_down(up * gate)


class SparsePertokenMoE(layers.Layer):
    def __init__(
        self,
        dim,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        alpha=2.0,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        self.router = layers.Dense(num_experts, use_bias=bias)
        # In TF 2.4, we just use a list of layers
        self.experts = [
            PertokenSwiGLU(dim, hidden_mult, bias=bias) for _ in range(num_experts - 1)
        ]
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult, bias=bias)

    def call(self, x):
        # Shape: (B, T, D)
        logits = self.router(x)
        probs = tf.nn.softmax(logits, axis=-1)

        # Get top-k
        topk_vals, topk_idx = tf.math.top_k(probs, k=self.top_k)

        # We initialize output as zeros
        output = tf.zeros_like(x)

        # Loop over top-k (excluding the last slot if strictly following PyTorch logic)
        for i in range(self.top_k - 1):
            expert_prob = tf.expand_dims(topk_vals[..., i], axis=-1)
            indices = topk_idx[..., i]

            expert_outputs_sum = tf.zeros_like(x)
            for j, expert in enumerate(self.experts):
                # Create mask for which tokens go to which expert
                mask = tf.cast(tf.equal(indices, j), dtype=x.dtype)
                mask = tf.expand_dims(mask, axis=-1)

                # Apply expert to all then mask (more TF friendly than boolean indexing)
                exp_out = expert(x)
                expert_outputs_sum += exp_out * mask

            output += self.alpha * expert_prob * expert_outputs_sum

        # Shared expert always activated
        output += self.shared_expert(x)
        return output


class MixingReverting(layers.Layer):
    def __init__(self, dim, num_heads, num_tokens, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim
        self.num_tokens = num_tokens

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        d = dim // num_heads
        mix_dim = num_tokens * d

        self.mixing = PertokenSwiGLU(mix_dim, bias=bias)
        self.reverting = PertokenSwiGLU(dim, bias=bias)

    def call(self, x):
        # B, T, D
        batch_size = tf.shape(x)[0]
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_norm = self.norm1(x)

        # Reshape for multi-head mixing: [B, T, H, d]
        x_split = tf.reshape(x_norm, (batch_size, T, H, d))
        # Permute to: [H, B, T, d]
        x_split = tf.transpose(x_split, perm=[2, 0, 1, 3])
        # Flatten T and d: [H, B, T*d]
        x_split = tf.reshape(x_split, (H, batch_size, T * d))

        x_mixed = self.mixing(x_split)

        # Reverse: [H, B, T, d]
        x_rev = tf.reshape(x_mixed, (H, batch_size, T, d))
        # [B, T, H, d]
        x_rev = tf.transpose(x_rev, perm=[1, 2, 0, 3])
        # [B, T, D]
        x_rev = tf.reshape(x_rev, (batch_size, T, self.dim))

        return x + self.norm2(self.reverting(x_rev))


class TokenMixerLargeBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mr = MixingReverting(dim, num_heads, num_tokens, bias=bias)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult, bias=bias)

    def call(self, x):
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(layers.Layer):
    def __init__(self, group_dims, model_dim, bias=False, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlps = []
        for dims in group_dims:
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        layers.Dense(model_dim, activation="relu", use_bias=bias),
                        layers.Dropout(dropout),
                        layers.Dense(model_dim, use_bias=bias),
                    ]
                )
            )

        self.global_mlp = tf.keras.Sequential(
            [
                layers.Dense(model_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(model_dim, use_bias=bias),
            ]
        )

    def call(self, groups):
        # groups is a list of lists of tensors
        tokens = []
        for group_tensors, mlp in zip(groups, self.mlps):
            # Concatenate tensors in each group
            concat = tf.concat(group_tensors, axis=-1)
            tokens.append(mlp(concat))

        # Stack into [B, T-1, D]
        stacked = tf.stack(tokens, axis=1)

        # Global token: Flatten stacked and pass through global_mlp
        batch_size = tf.shape(stacked)[0]
        flattened = tf.reshape(stacked, (batch_size, -1))
        global_token = self.global_mlp(flattened)
        global_token = tf.expand_dims(global_token, axis=1)

        # Concatenate global token with group tokens
        return tf.concat([global_token, stacked], axis=1)


class TokenMixerLarge(tf.keras.Model):
    def __init__(
        self,
        group_dims: List[List[int]],
        num_layers: int,
        num_sparse_embs: List[int],
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
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
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse
        self.num_layers = num_layers
        self.tokenizer = SemanticTokenizer(group_dims, dim_emb, bias, dropout)
        num_tokens = len(group_dims) + 1
        self.blocks = [
            TokenMixerLargeBlock(
                dim_emb, num_heads, num_tokens, num_experts, top_k, bias=bias
            )
            for _ in range(num_layers)
        ]
        self.projection_head = MLP(
            dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )
        self.aux_head = layers.Dense(dim_output, use_bias=bias)

    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs
        x = self.embedding(sparse_inputs, dense_inputs)
        x = self.tokenizer(
            [
                [x[:, i] for i in range(self.dim_input_sparse)],
                [
                    x[:, i]
                    for i in range(
                        self.dim_input_sparse,
                        self.dim_input_sparse + self.dim_input_dense,
                    )
                ],
            ]
        )
        is_last_layer = self.num_layers - 1
        residual_cache = []
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i % 2 == 1 and i != is_last_layer:
                x = x + residual_cache[-1]
                residual_cache = []
            residual_cache.append(x)
            if training and i < is_last_layer and i % 2 == 1:
                aux_pooled = tf.reduce_mean(x, axis=1)
                aux_logit = self.aux_head(aux_pooled)
                self.add_loss(tf.reduce_mean(tf.square(aux_logit)))
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
    model = TokenMixerLarge(
        group_dims=[[128] * 26, [128] * 13],
        num_layers=2,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=26,
        dim_input_dense=13,
        dim_emb=128,
        num_heads=4,
        num_experts=4,
        top_k=2,
        num_hidden_head=256,
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

    # 根据是否有辅助输出处理打印逻辑
    if isinstance(outputs, tuple):
        main_out, aux_outs = outputs
        print("Model main output shape:", main_out.shape)
        print(f"Number of auxiliary outputs: {len(aux_outs)}")
    else:
        print("Model output shape:", outputs.shape)
