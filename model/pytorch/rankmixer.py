import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


def custom_gelu(x):
    """Matches the exact approximation used in the TF implementation."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SemanticTokenization(nn.Module):
    def __init__(self, num_T, in_features_per_token, num_D):
        super().__init__()
        self.num_T = num_T
        self.num_D = num_D
        # PyTorch requires explicit input dimensions
        self.dense_layers = nn.ModuleList([
            nn.Linear(in_features_per_token, num_D) for _ in range(num_T)
        ])

    def forward(self, x):
        # x: (B, num_T * in_features_per_token)
        x_chunks = torch.chunk(x, self.num_T, dim=-1) # Tuple of (B, in_features_per_token)
        outputs = [layer(chunk) for chunk, layer in zip(x_chunks, self.dense_layers)]
        return torch.stack(outputs, dim=1)  # (B, num_T, num_D)


class TokenMixer(nn.Module):
    def __init__(self, num_T, num_D, num_H):
        super().__init__()
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D // num_H

    def forward(self, x):
        B = x.size(0)
        # (B, T, D) -> (B, T, H, D/H)
        x = x.view(B, self.num_T, self.num_H, self.d_k)
        # (B, T, H, D/H) -> (B, H, T, D/H)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, H, T, D/H) -> (B, H, T * D/H)
        x = x.view(B, self.num_H, self.num_T * self.d_k)
        return x


class PerTokenFFN(nn.Module):
    def __init__(self, num_T, num_D, expansion_ratio=4, dropout=0.0):
        super().__init__()
        self.experts = nn.ModuleList()
        for i in range(num_T):
            self.experts.append(nn.Sequential(
                nn.Linear(num_D, num_D * expansion_ratio),
                nn.GELU(approximate='tanh'), # You can swap this with custom_gelu if preferred
                nn.Dropout(dropout),
                nn.Linear(num_D * expansion_ratio, num_D)
            ))

    def forward(self, x):
        outputs = []
        for i, expert in enumerate(self.experts):
            h = x[:, i, :]
            outputs.append(expert(h))
        return torch.stack(outputs, dim=1)


class PerTokenSparseMoE(nn.Module):
    """
    Per-token Sparse MoE with ReLU routing + optional DTSI.
    """
    def __init__(
        self,
        num_T,
        num_D,
        expansion_ratio=4,
        num_experts=4,
        dropout=0.0,
        l1_coef=0.0,  # Unused in TF forward, keeping for signature parity
        sparsity_ratio=1.0, # Unused in TF forward, keeping for signature parity
        use_dtsi=True,
        routing_type="relu_dtsi",
    ):
        super().__init__()
        self.num_T = int(num_T)
        self.num_D = int(num_D)
        self.expansion_ratio = int(expansion_ratio)
        self.num_experts = int(num_experts)
        self.dropout = nn.Dropout(dropout)
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

        hidden_dim = self.num_D * self.expansion_ratio

        # Define custom weights
        self.W1 = nn.Parameter(torch.empty(self.num_T, self.num_experts, self.num_D, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_T, self.num_experts, hidden_dim))
        
        self.W2 = nn.Parameter(torch.empty(self.num_T, self.num_experts, hidden_dim, self.num_D))
        self.b2 = nn.Parameter(torch.zeros(self.num_T, self.num_experts, self.num_D))
        
        self.gate_w_train = nn.Parameter(torch.empty(self.num_T, self.num_D, self.num_experts))
        self.gate_b_train = nn.Parameter(torch.zeros(self.num_T, self.num_experts))
        
        if self.use_dtsi:
            self.gate_w_infer = nn.Parameter(torch.empty(self.num_T, self.num_D, self.num_experts))
            self.gate_b_infer = nn.Parameter(torch.zeros(self.num_T, self.num_experts))

        self._reset_parameters()

    def _trunc_normal(self, tensor, fan_in):
        # Approximates TF's VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")
        std = math.sqrt(2.0 / fan_in)
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)

    def _reset_parameters(self):
        self._trunc_normal(self.W1, fan_in=self.num_D)
        self._trunc_normal(self.W2, fan_in=self.num_D * self.expansion_ratio)
        self._trunc_normal(self.gate_w_train, fan_in=self.num_D)
        if self.use_dtsi:
            self._trunc_normal(self.gate_w_infer, fan_in=self.num_D)

    def _router_logits(self, x, w, b):
        return torch.einsum("btd,tde->bte", x, w) + b

    def forward(self, x):
        # x shape: [B, T, D]
        h = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = custom_gelu(h)
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
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = F.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # Use self.training built-in PyTorch flag
        gate = gate_train if self.training else gate_infer
        y = torch.sum(expert_out * gate.unsqueeze(-1), dim=2)
        return y


class RankMixerLayer(nn.Module):
    def __init__(self, num_T, num_D, num_H, expansion_ratio, use_moe=False, dropout=0.0):
        super().__init__()
        self.token_mixer = TokenMixer(num_T, num_D, num_H)
        
        if use_moe:
            self.per_token_ffn = PerTokenSparseMoE(num_T, num_D, expansion_ratio, dropout=dropout)
        else:
            self.per_token_ffn = PerTokenFFN(num_T, num_D, expansion_ratio, dropout=dropout)
            
        self.norm1 = nn.LayerNorm(num_D, eps=1e-6)
        self.norm2 = nn.LayerNorm(num_D, eps=1e-6)

    def forward(self, x):
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
                f"num_tokens (T) must be equal to num_heads (H) for RankMixerLayer, "
                f"but got T={num_tokens}, H={num_heads}"
            )
            
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        
        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse
        self.num_tokens = num_tokens

        # In PyTorch, we pre-calculate the flat input dimension for SemanticTokenization
        in_features_per_token = ((dim_input_dense + dim_input_sparse) * dim_emb) // num_tokens
        self.semantic_tokenization = SemanticTokenization(num_tokens, in_features_per_token, dim_emb)
        
        self.layers_list = nn.ModuleList([
            RankMixerLayer(
                num_tokens,
                dim_emb,
                num_heads,
                expansion_ratio,
                use_moe=(i % 2 == 0),
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        self.projection_head = MLP(
            num_tokens * dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )

    def forward(self, sparse_inputs: Tensor, dense_inputs) -> Tensor:
        x = self.embedding(sparse_inputs, dense_inputs)
        
        B = x.size(0)
        # Reshape to (B, total_features)
        x = x.view(B, (self.dim_input_dense + self.dim_input_sparse) * self.dim_emb)
        
        x = self.semantic_tokenization(x)
        
        for layer in self.layers_list:
            x = layer(x)
            
        x = x.view(B, self.num_tokens * self.dim_emb)
        x = self.projection_head(x)
        
        return x