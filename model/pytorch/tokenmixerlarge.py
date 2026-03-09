import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        norm = x_fp32.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps).to(x.dtype)
        return self.scale * x


class PertokenSwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01, bias=False):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc_down = nn.Linear(hidden_dim, dim, bias=bias)

        # 初始化下投影层，模拟 TF 的 VarianceScaling
        nn.init.trunc_normal_(self.fc_down.weight, std=down_scale)

    def forward(self, x):
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        # Swish (SiLU) = x * sigmoid(x)
        gate = F.silu(gate_logits)
        return self.fc_down(up * gate)


class SparsePertokenMoE(nn.Module):
    def __init__(
        self, dim, num_experts=4, top_k=2, hidden_mult=4, alpha=2.0, bias=False
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

    def forward(self, x):
        # Shape: (B, T, D)
        logits = self.router(x)
        probs = F.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)

        # Get top-k
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        output = torch.zeros_like(x)

        # 这里的逻辑遵循原代码：遍历 top-k 的前 (k-1) 个槽位
        # 注意：TF 原代码逻辑是 top_k-1，这意味着如果 top_k=2，只循环一次
        for i in range(self.top_k - 1):
            expert_prob = topk_vals[..., i : i + 1]  # (B, T, 1)
            indices = topk_idx[..., i]  # (B, T)

            expert_outputs_sum = torch.zeros_like(x)
            for j, expert in enumerate(self.experts):
                # 创建掩码
                mask = (indices == j).unsqueeze(-1)

                # 计算该 expert 的输出并累加
                exp_out = expert(x)
                safe_out = torch.where(mask, exp_out, torch.zeros_like(exp_out))
                expert_outputs_sum += safe_out

            output += self.alpha * expert_prob * expert_outputs_sum

        # 共享专家始终激活
        output += self.shared_expert(x)
        return output


class MixingReverting(nn.Module):
    def __init__(self, dim, num_heads, num_tokens, bias=False):
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

    def forward(self, x):
        # B, T, D
        B = x.shape[0]
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_norm = self.norm1(x)

        # Reshape & Permute: [B, T, H, d] -> [H, B, T, d]
        x_split = x_norm.view(B, T, H, d).permute(2, 0, 1, 3)
        # Flatten: [H, B, T*d]
        x_split = x_split.reshape(H, B, T * d)

        x_mixed = self.mixing(x_split)

        # Reverse: [H, B, T, d]
        x_rev = x_mixed.view(H, B, T, d)
        # [B, T, H, d] -> [B, T, D]
        x_rev = x_rev.permute(1, 2, 0, 3).reshape(B, T, self.dim)

        return x + self.norm2(self.reverting(x_rev))


class TokenMixerLargeBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        bias=False,
    ):
        super().__init__()
        self.mr = MixingReverting(dim, num_heads, num_tokens, bias=bias)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult, bias=bias)

    def forward(self, x):
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

        # 遍历各个组，显式计算每组的输入特征维度
        for group in group_dims:
            in_features = sum(group)  # 组内各特征维度的总和
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_features, model_dim, bias=bias),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim, model_dim, bias=bias),
                )
            )

        # 全局输入维度：一共有 len(group_dims) 个 token，每个 token 的维度为 model_dim
        num_groups = len(group_dims)
        global_in_features = num_groups * model_dim

        self.global_mlp = nn.Sequential(
            nn.Linear(global_in_features, model_dim, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim, bias=bias),
        )

    def forward(self, groups: List[List[torch.Tensor]]):
        tokens = []
        for group_tensors, mlp in zip(groups, self.mlps):
            # group_tensors 是一组 tensor 的列表
            concat = torch.cat(group_tensors, dim=-1)
            tokens.append(mlp(concat))

        # Stack into [B, T-1, D]
        stacked = torch.stack(tokens, dim=1)

        # Global token
        B = stacked.shape[0]
        flattened = stacked.view(B, -1)
        global_token = self.global_mlp(flattened).unsqueeze(1)

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
        # 假设接口一致
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
            dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )

    def forward(self, sparse_inputs, dense_inputs):
        x = self.embedding(sparse_inputs, dense_inputs)
        # 这里假设 x 的 shape 是 [B, Total_Inputs, D]
        # 下面模拟原代码的切片逻辑

        # 模拟 tokenizer 的输入构造
        sparse_list = [x[:, i] for i in range(self.dim_input_sparse)]
        dense_list = [
            x[:, i]
            for i in range(
                self.dim_input_sparse, self.dim_input_sparse + self.dim_input_dense
            )
        ]

        x = self.tokenizer([sparse_list, dense_list])

        residual_cache = []
        for i, layer in enumerate(self.blocks):
            x = layer(x)

            if i % 2 == 1:
                x = x + residual_cache[-1]
                residual_cache = []

            residual_cache.append(x)

        x = x.mean(dim=1)
        x = self.projection_head(x)
        return x


if __name__ == "__main__":
    import numpy as np

    # 1. 基础参数设置
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
    DIM_EMB = 128

    # 2. 实例化模型
    # 注意：这里的 group_dims 需要匹配你的 Embedding 输出
    # 原代码逻辑是将 26 个稀疏特征作为一组，13 个稠密特征作为一组
    model = TokenMixerLarge(
        group_dims=[[DIM_EMB] * DIM_INPUT_SPARSE, [DIM_EMB] * DIM_INPUT_DENSE],
        num_layers=2,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=DIM_INPUT_SPARSE,
        dim_input_dense=DIM_INPUT_DENSE,
        dim_emb=DIM_EMB,
        num_heads=4,
        num_experts=4,
        top_k=2,
        num_hidden_head=256,
        dim_hidden_head=128,
        dim_output=1,
    )

    # 3. 准备模拟输入数据
    # Sparse Inputs: [BATCH_SIZE, 26]
    sparse_data = np.column_stack(
        [
            np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
            for i in range(DIM_INPUT_SPARSE)
        ]
    ).astype(np.int64)

    sparse_inputs = torch.from_numpy(sparse_data)

    # Dense Inputs: [BATCH_SIZE, 13]
    dense_data = np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    dense_inputs = torch.from_numpy(dense_data)

    # 4. 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(sparse_inputs, dense_inputs)

    print("Model output shape:", outputs.shape)
