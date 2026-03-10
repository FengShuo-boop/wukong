import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


# ---------------- RMSNorm ----------------
class RMSLayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.eps = epsilon
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


# ---------------- Mixed FFN (tail LNS token-specific) ----------------
class MixedFFN(nn.Module):
    def __init__(
        self,
        dim_emb: int,
        d_ff: int,
        LNS: int,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.LNS = LNS
        self.W1S = nn.Linear(dim_emb, d_ff, bias=bias)
        self.W2S = nn.Linear(d_ff, dim_emb, bias=bias)

        self.W1NS = nn.Parameter(torch.empty(LNS, dim_emb, d_ff))
        self.W2NS = nn.Parameter(torch.empty(LNS, d_ff, dim_emb))

        # Glorot Uniform translates to Xavier Uniform in PyTorch
        nn.init.xavier_uniform_(self.W1NS)
        nn.init.xavier_uniform_(self.W2NS)

        self.act = F.gelu if activation == "gelu" else getattr(F, activation)

    def forward(self, x):
        T = x.size(1)
        t = min(T, self.LNS)  # tail token-specific count
        s = T - t  # shared count

        # Shared processing for earlier tokens
        xS = x[:, :s]
        yS = self.W2S(self.act(self.W1S(xS)))  # [B, s, D]

        # Token-specific processing for tail tokens
        xT = x[:, s:]  # [B, t, D]
        W1 = self.W1NS[-t:]  # [t, D, d_ff]
        W2 = self.W2NS[-t:]  # [t, d_ff, D]

        h = self.act(torch.einsum("btd,tde->bte", xT, W1))
        yT = torch.einsum("btd,tde->bte", h, W2)  # [B, t, D]

        return torch.cat([yS, yT], dim=1)


# ---------------- Pyramid Mixed Causal Attention (Eq.14 strict) ----------------
class PyramidMixedCausalAttention(nn.Module):
    def __init__(self, dim_emb: int, num_heads: int, LNS: int):
        super().__init__()
        assert (
            dim_emb % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."
        self.D, self.H, self.dh = dim_emb, num_heads, dim_emb // num_heads
        self.LNS = LNS

        self.WqS = nn.Linear(dim_emb, dim_emb, bias=False)
        self.WkS = nn.Linear(dim_emb, dim_emb, bias=False)
        self.WvS = nn.Linear(dim_emb, dim_emb, bias=False)

        self.WqNS = nn.Parameter(torch.empty(LNS, dim_emb, dim_emb))
        self.WkNS = nn.Parameter(torch.empty(LNS, dim_emb, dim_emb))
        self.WvNS = nn.Parameter(torch.empty(LNS, dim_emb, dim_emb))

        nn.init.xavier_uniform_(self.WqNS)
        nn.init.xavier_uniform_(self.WkNS)
        nn.init.xavier_uniform_(self.WvNS)

        self.Wo = nn.Linear(dim_emb, dim_emb, bias=False)

    def _mh(self, x):  # [B,T,D] -> [B,H,T,dh]
        b, t = x.size(0), x.size(1)
        return x.view(b, t, self.H, self.dh).transpose(1, 2)

    def _unmh(self, x):  # [B,H,T,dh] -> [B,T,D]
        x = x.transpose(1, 2).contiguous()
        return x.view(x.size(0), x.size(1), self.D)

    def forward(self, x, Lq):
        L = x.size(1)
        t = min(L, self.LNS)
        s = L - t

        xS, xT = x[:, :s], x[:, s:]

        # Shared representations
        qS, kS, vS = self.WqS(xS), self.WkS(xS), self.WvS(xS)

        # Token-specific representations
        qT = torch.einsum("btd,tde->bte", xT, self.WqNS[-t:])
        kT = torch.einsum("btd,tde->bte", xT, self.WkNS[-t:])
        vT = torch.einsum("btd,tde->bte", xT, self.WvNS[-t:])

        Q = torch.cat([qS, qT], dim=1)
        K = torch.cat([kS, kT], dim=1)
        V = torch.cat([vS, vT], dim=1)

        # Only tail queries
        Q = Q[:, -Lq:]

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)

        # Scaled Dot-Product
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.dh)

        # Causal mask for tail queries
        q = torch.arange(L - Lq, L, device=x.device).unsqueeze(1)
        k = torch.arange(L, device=x.device).unsqueeze(0)
        mask = k > q  # [Lq, L]
        logits = logits.masked_fill(
            mask.unsqueeze(0).unsqueeze(0), torch.finfo(logits.dtype).min
        )

        out = torch.matmul(F.softmax(logits, dim=-1), Vh)  # [B,H,Lq,dh]
        return self.Wo(self._unmh(out))  # [B,Lq,D]


# ---------------- OneTrans Block (auto Lq=L-1) ----------------
class OneTransBlock(nn.Module):
    def __init__(
        self,
        dim_emb: int,
        num_heads: int,
        d_ff: int,
        LNS: int,
        ln_eps: float = 1e-5,
        bias: bool = False,
    ):
        super().__init__()
        # In PyTorch, we need to specify the dimension for our RMS Norm manually
        self.ln1 = RMSLayerNorm(dim_emb, ln_eps)
        self.ln2 = RMSLayerNorm(dim_emb, ln_eps)
        self.mha = PyramidMixedCausalAttention(dim_emb, num_heads, LNS)
        self.ffn = MixedFFN(dim_emb, d_ff, LNS, bias=bias)

    def forward(self, x, Lq):
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]
        return self.ffn(self.ln2(z)) + z


# ---------------- Stack: compress S for LS layers ----------------
class OneTrans(nn.Module):
    def __init__(
        self,
        LS: int,
        LNS: int,
        dim_emb: int,
        num_heads: int,
        d_ff: int,
        num_sparse_embs: List[int],
        dim_input_dense: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)

        self.Lq_list = list(range(LS + LNS, LNS, -4))  # LS+LNS .. LNS+1
        self.Lq_list.append(LNS)

        # ModuleList is required in PyTorch for lists of layers to properly register parameters
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(dim_emb, num_heads, d_ff, LNS=LNS, bias=bias)
                for _ in range(len(self.Lq_list))
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

        for blk, Lq_py in zip(self.blocks, self.Lq_list):
            x = blk(x, Lq_py)

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

    # Ensure reproducibility for test
    torch.manual_seed(42)

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
        dim_input_dense=DIM_INPUT_DENSE,
    )

    # Creating random sparse inputs cleanly with list comprehension and torch.stack
    sparse_inputs = torch.stack(
        [torch.randint(0, high, (BATCH_SIZE,)) for high in NUM_SPARSE_EMBS], dim=1
    ).long()

    # Dense float inputs
    dense_inputs = torch.rand(BATCH_SIZE, DIM_INPUT_DENSE, dtype=torch.float32)

    print("Sparse input shape:", sparse_inputs.shape)
    print("Dense input shape:", dense_inputs.shape)

    outputs = model(sparse_inputs, dense_inputs)

    print("Model output shape:", outputs.shape)
