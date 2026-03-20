import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # norm = mean(x^2)
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x


class PertokenSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        down_scale: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc_down = nn.Linear(hidden_dim, dim, bias=bias)

        # Equivalent to VarianceScaling(scale=down_scale, mode="fan_avg", distribution="uniform")
        fan_in = hidden_dim
        fan_out = dim
        fan_avg = (fan_in + fan_out) / 2.0
        limit = math.sqrt(3.0 * down_scale / fan_avg)

        nn.init.uniform_(self.fc_down.weight, -limit, limit)
        if bias:
            nn.init.zeros_(self.fc_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        # Swish = x * sigmoid(x), natively available as SiLU in PyTorch
        gate = F.silu(gate_logits)
        return self.fc_down(up * gate)


class SparsePertokenMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        alpha: float = 2.0,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        self.router = nn.Linear(dim, num_experts, bias=bias)
        self.experts = nn.ModuleList(
            [
                PertokenSwiGLU(dim, hidden_mult, bias=bias)
                for _ in range(num_experts - 1)
            ]
        )
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (B, T, D)
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        # Get top-k
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        # We initialize output as zeros
        output = torch.zeros_like(x)

        # Loop over top-k (excluding the last slot if strictly following PyTorch logic)
        for i in range(self.top_k - 1):
            expert_prob = topk_vals[..., i].unsqueeze(-1)
            indices = topk_idx[..., i]

            expert_outputs_sum = torch.zeros_like(x)
            for j, expert in enumerate(self.experts):
                # Create mask for which tokens go to which expert
                mask = (indices == j).to(x.dtype).unsqueeze(-1)

                # Apply expert to all then mask
                exp_out = expert(x)
                expert_outputs_sum += exp_out * mask

            output += self.alpha * expert_prob * expert_outputs_sum

        # Shared expert always activated
        output += self.shared_expert(x)
        return output


class MixingReverting(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_tokens: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_tokens = num_tokens

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        d = dim // num_heads
        mix_dim = num_tokens * d

        self.mixing = PertokenSwiGLU(mix_dim, bias=bias)
        self.reverting = PertokenSwiGLU(dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, T, D
        batch_size = x.size(0)
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_norm = self.norm1(x)

        # Reshape for multi-head mixing: [B, T, H, d]
        x_split = x_norm.view(batch_size, T, H, d)
        # Permute to: [H, B, T, d]
        x_split = x_split.permute(2, 0, 1, 3).contiguous()
        # Flatten T and d: [H, B, T*d]
        x_split = x_split.view(H, batch_size, T * d)

        x_mixed = self.mixing(x_split)

        # Reverse: [H, B, T, d]
        x_rev = x_mixed.view(H, batch_size, T, d)
        # [B, T, H, d]
        x_rev = x_rev.permute(1, 2, 0, 3).contiguous()
        # [B, T, D]
        x_rev = x_rev.view(batch_size, T, self.dim)

        return x + self.norm2(self.reverting(x_rev))


class TokenMixerLargeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tokens: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.mr = MixingReverting(dim, num_heads, num_tokens, bias=bias)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(nn.Module):
    def __init__(
        self,
        group_dims: List[List[int]],
        model_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()

        for dims in group_dims:
            in_dim = sum(dims)  # Calculate input dimension since PyTorch requires it
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_dim, model_dim, bias=bias),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim, model_dim, bias=bias),
                )
            )

        global_in_dim = len(group_dims) * model_dim
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in_dim, model_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim, bias=bias),
        )

    def forward(self, groups: List[List[torch.Tensor]]) -> torch.Tensor:
        tokens = []
        for group_tensors, mlp in zip(groups, self.mlps):
            # Concatenate tensors in each group
            concat = torch.cat(group_tensors, dim=-1)
            tokens.append(mlp(concat))

        # Stack into [B, T-1, D]
        stacked = torch.stack(tokens, dim=1)

        # Global token: Flatten stacked and pass through global_mlp
        batch_size = stacked.size(0)
        flattened = stacked.view(batch_size, -1)
        global_token = self.global_mlp(flattened)
        global_token = global_token.unsqueeze(1)

        # Concatenate global token with group tokens
        return torch.cat([global_token, stacked], dim=1)


class TokenMixerLarge(nn.Module):
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
    ):
        super().__init__()
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse

        self.tokenizer = SemanticTokenizer(group_dims, dim_emb, bias, dropout)
        num_tokens = len(group_dims) + 1

        self.blocks = nn.ModuleList(
            [
                TokenMixerLargeBlock(
                    dim_emb,
                    num_heads,
                    num_tokens,
                    num_experts,
                    top_k,
                    hidden_mult=4,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.projection_head = MLP(
            dim_in=dim_emb,
            num_hidden=num_hidden_head,
            dim_hidden=dim_hidden_head,
            dim_out=dim_output,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, sparse_inputs, dense_inputs) -> torch.Tensor:
        x = self.embedding(sparse_inputs, dense_inputs)

        group1 = [x[:, i] for i in range(self.dim_input_sparse)]
        group2 = [
            x[:, i]
            for i in range(
                self.dim_input_sparse, self.dim_input_sparse + self.dim_input_dense
            )
        ]

        x = self.tokenizer([group1, group2])

        residual_cache = []
        for i, layer in enumerate(self.blocks):
            x = layer(x)

            if i % 2 == 1:
                x = x + residual_cache[-1]
                residual_cache = []

            residual_cache.append(x)

        x = torch.mean(x, dim=1)
        x = self.projection_head(x)
        return x


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
        num_hidden_head=256,  # Represents layers
        dim_hidden_head=128,
        dim_output=1,
    )

    sparse_inputs_np = np.column_stack(
        [
            np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
            for i in range(DIM_INPUT_SPARSE)
        ]
    ).astype(np.int64)

    dense_inputs_np = np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)

    # Convert to PyTorch tensors
    sparse_inputs = torch.tensor(sparse_inputs_np)
    dense_inputs = torch.tensor(dense_inputs_np)

    # Disable gradient calculation for a forward pass test
    with torch.no_grad():
        outputs = model(sparse_inputs, dense_inputs)

    print("Model output shape:", outputs.shape)
