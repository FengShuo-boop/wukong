import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


class SemanticTokenization(nn.Module):
    def __init__(self, dim_in: int, num_T: int, num_D: int):
        super().__init__()
        self.num_T = num_T
        self.num_D = num_D

        # 计算每个 token 块的输入维度
        self.chunk_size = dim_in // num_T

        self.dense_layers = nn.ModuleList(
            [nn.Linear(self.chunk_size, num_D) for _ in range(num_T)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim_in) -> (B, num_T, chunk_size)
        x_chunks = torch.split(x, self.chunk_size, dim=-1)

        outputs = [layer(x_chunks[i]) for i, layer in enumerate(self.dense_layers)]
        return torch.stack(outputs, dim=1)  # (B, num_T, num_D)


class TokenMixer(nn.Module):
    def __init__(self, num_T: int, num_D: int, num_H: int):
        super().__init__()
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D // num_H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # (B, T, D) -> (B, T, H, D/H)
        x = x.view(B, self.num_T, self.num_H, self.d_k)
        # -> (B, H, T, D/H)
        x = x.permute(0, 2, 1, 3).contiguous()
        # -> (B, H, T * D/H)
        x = x.view(B, self.num_H, self.num_T * self.d_k)
        return x


class PerTokenFFN(nn.Module):
    def __init__(
        self, num_T: int, num_D: int, expansion_ratio: int = 4, dropout: float = 0.0
    ):
        super().__init__()

        # 每个 expert 的 FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_D, num_D * expansion_ratio),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(dropout),
                    nn.Linear(num_D * expansion_ratio, num_D),
                )
                for _ in range(num_T)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, expert in enumerate(self.experts):
            h = x[:, i, :]
            h = expert(h)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


class PerTokenSparseMoE(nn.Module):
    """
    Per-token Sparse MoE with ReLU routing + optional DTSI.
    """

    def __init__(
        self,
        num_T: int,
        num_D: int,
        expansion_ratio: int = 4,
        num_experts: int = 4,
        dropout: float = 0.0,
        l1_coef: float = 0.0,
        sparsity_ratio: float = 1.0,
        use_dtsi: bool = True,
        routing_type: str = "relu_dtsi",
    ):
        super().__init__()
        self.num_T = int(num_T)
        self.num_D = int(num_D)
        self.expansion_ratio = int(expansion_ratio)
        self.num_experts = int(num_experts)
        self.dropout = nn.Dropout(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

        hidden_dim = self.num_D * self.expansion_ratio

        # 可学习参数
        self.W1 = nn.Parameter(
            torch.empty(self.num_T, self.num_experts, self.num_D, hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_T, self.num_experts, hidden_dim))

        self.W2 = nn.Parameter(
            torch.empty(self.num_T, self.num_experts, hidden_dim, self.num_D)
        )
        self.b2 = nn.Parameter(torch.zeros(self.num_T, self.num_experts, self.num_D))

        self.gate_w_train = nn.Parameter(
            torch.empty(self.num_T, self.num_D, self.num_experts)
        )
        self.gate_b_train = nn.Parameter(torch.zeros(self.num_T, self.num_experts))

        if self.use_dtsi:
            self.gate_w_infer = nn.Parameter(
                torch.empty(self.num_T, self.num_D, self.num_experts)
            )
            self.gate_b_infer = nn.Parameter(torch.zeros(self.num_T, self.num_experts))

        self._reset_parameters()

    def _reset_parameters(self):
        # 等效于 TF 的 VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")
        std_w1 = math.sqrt(2.0 / self.num_D)
        nn.init.trunc_normal_(self.W1, std=std_w1)

        std_w2 = (
            math.sqrt(2.0 / hidden_dim)
            if (hidden_dim := self.num_D * self.expansion_ratio)
            else 0
        )
        nn.init.trunc_normal_(self.W2, std=std_w2)

        std_gate = math.sqrt(2.0 / self.num_D)
        nn.init.trunc_normal_(self.gate_w_train, std=std_gate)
        if self.use_dtsi:
            nn.init.trunc_normal_(self.gate_w_infer, std=std_gate)

    def _router_logits(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        # 每个 token 的路由 logits
        return torch.einsum("btd,tde->bte", x, w) + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算每个 token 的专家输出
        h = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = F.gelu(h, approximate="tanh")
        h = self.dropout(h)

        expert_out = torch.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        expert_out = self.dropout(expert_out)

        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)

        if self.routing_type == "relu_dtsi":
            gate_train = F.softmax(gate_train_logits, dim=-1)
        elif self.routing_type == "relu":
            gate_train = F.relu(gate_train_logits)
        else:
            raise ValueError(f"Unsupported routing_type: {self.routing_type}")

        if self.use_dtsi:
            gate_infer_logits = self._router_logits(
                x, self.gate_w_infer, self.gate_b_infer
            )
            gate_infer = F.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # 根据 self.training 自动选择 gate
        gate = gate_train if self.training else gate_infer

        y = torch.sum(expert_out * gate.unsqueeze(-1), dim=2)
        return y


class RankMixerLayer(nn.Module):
    def __init__(
        self,
        num_T: int,
        num_D: int,
        num_H: int,
        expansion_ratio: int,
        use_moe: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_mixer = TokenMixer(num_T, num_D, num_H)

        if use_moe:
            self.per_token_ffn = PerTokenSparseMoE(
                num_T, num_D, expansion_ratio, dropout=dropout
            )
        else:
            self.per_token_ffn = PerTokenFFN(
                num_T, num_D, expansion_ratio, dropout=dropout
            )

        self.norm1 = nn.LayerNorm(num_D, eps=1e-6)
        self.norm2 = nn.LayerNorm(num_D, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed_x = self.token_mixer(x)
        x = self.norm1(x + mixed_x)
        x = self.norm2(x + self.per_token_ffn(x))
        return x


class RankMixer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_sparse_embs: List[int],
        num_tokens: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        num_heads: int,
        expansion_ratio: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if num_tokens != num_heads:
            raise ValueError(
                f"num_tokens (T) must be equal to num_heads (H) for RankMixerLayer, but got T={num_tokens}, H={num_heads}"
            )

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse
        self.num_tokens = num_tokens

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)

        # 计算 SemanticTokenization 需要的输入总维度
        dim_in_semantic = (dim_input_dense + dim_input_sparse) * dim_emb
        self.semantic_tokenization = SemanticTokenization(
            dim_in_semantic, num_tokens, dim_emb
        )

        self.layers_list = nn.ModuleList(
            [
                RankMixerLayer(
                    num_T=num_tokens,
                    num_D=dim_emb,
                    num_H=num_heads,
                    expansion_ratio=expansion_ratio,
                    use_moe=(i % 2 == 0),
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.projection_head = MLP(
            dim_in=num_tokens * dim_emb,
            num_hidden=num_hidden_head,
            dim_hidden=dim_hidden_head,
            dim_out=dim_output,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, sparse_inputs, dense_inputs) -> torch.Tensor:
        x = self.embedding(sparse_inputs, dense_inputs)
        x = x.view(-1, (self.dim_input_dense + self.dim_input_sparse) * self.dim_emb)
        x = self.semantic_tokenization(x)

        for layer in self.layers_list:
            x = layer(x)

        x = x.view(-1, self.num_tokens * self.dim_emb)
        x = self.projection_head(x)
        return x
