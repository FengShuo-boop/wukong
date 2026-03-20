import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.pytorch.mlp import MLP
from model.pytorch.embedding import Embedding


# ---------------- RMSNorm ----------------
class RMSLayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.eps = epsilon
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


# ---------------- Mixed FFN (tail LNS token-specific) ----------------
class MixedFFN(nn.Module):
    def __init__(self, dim_emb, d_ff, LNS, activation="gelu", bias=False):
        super().__init__()
        self.LNS = LNS
        self.W1S = nn.Linear(dim_emb, d_ff, bias=bias)
        self.W2S = nn.Linear(d_ff, dim_emb, bias=bias)

        # Token-specific weights initialized with Glorot Uniform (Xavier)
        self.W1NS = nn.Parameter(torch.empty(LNS, dim_emb, d_ff))
        self.W2NS = nn.Parameter(torch.empty(LNS, d_ff, dim_emb))
        nn.init.xavier_uniform_(self.W1NS)
        nn.init.xavier_uniform_(self.W2NS)

        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        B, T, D = x.shape
        t = min(T, self.LNS)  # tail token-specific count
        s = T - t  # shared count

        # Shared processing
        xS = x[:, :s]
        yS = (
            self.W2S(self.act(self.W1S(xS)))
            if s > 0
            else torch.empty(B, 0, D, device=x.device)
        )

        # Token-specific processing
        xT = x[:, s:]
        if t > 0:
            W1 = self.W1NS[-t:]
            W2 = self.W2NS[-t:]
            h = self.act(torch.einsum("btd,tde->bte", xT, W1))
            yT = torch.einsum("btd,tde->bte", h, W2)
        else:
            yT = torch.empty(B, 0, D, device=x.device)

        return torch.cat([yS, yT], dim=1)


# ---------------- Pyramid Mixed Causal Attention (Eq.14 strict) ----------------
class PyramidMixedCausalAttention(nn.Module):
    def __init__(self, dim_emb, num_heads, LNS):
        super().__init__()
        assert dim_emb % num_heads == 0
        self.D = dim_emb
        self.H = num_heads
        self.dh = dim_emb // num_heads
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
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.dh).transpose(1, 2)

    def _unmh(self, x):  # [B,H,T,dh] -> [B,T,D]
        B, H, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.D)

    def forward(self, x, Lq):
        B, L, _ = x.shape
        t = min(L, self.LNS)
        s = L - t

        xS, xT = x[:, :s], x[:, s:]

        if s > 0:
            qS, kS, vS = self.WqS(xS), self.WkS(xS), self.WvS(xS)
        else:
            qS = kS = vS = torch.empty(B, 0, self.D, device=x.device)

        if t > 0:
            qT = torch.einsum("btd,tde->bte", xT, self.WqNS[-t:])
            kT = torch.einsum("btd,tde->bte", xT, self.WkNS[-t:])
            vT = torch.einsum("btd,tde->bte", xT, self.WvNS[-t:])
        else:
            qT = kT = vT = torch.empty(B, 0, self.D, device=x.device)

        Q = torch.cat([qS, qT], dim=1)
        K = torch.cat([kS, kT], dim=1)
        V = torch.cat([vS, vT], dim=1)

        Q = Q[:, -Lq:]  # only tail queries

        Qh, Kh, Vh = self._mh(Q), self._mh(K), self._mh(V)

        # Scaling factor
        scale = self.dh**-0.5
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale

        # Causal mask for tail queries
        q = torch.arange(L - Lq, L, device=x.device)[:, None]
        k = torch.arange(L, device=x.device)[None, :]
        mask = k > q

        # Apply mask
        logits = logits.masked_fill(mask[None, None, :, :], -1e9)

        attn_weights = F.softmax(logits, dim=-1)
        out = torch.matmul(attn_weights, Vh)  # [B,H,Lq,dh]
        return self.Wo(self._unmh(out))  # [B,Lq,D]


# ---------------- OneTrans Block (auto Lq=L-1) ----------------
class OneTransBlock(nn.Module):
    def __init__(self, dim_emb, num_heads, d_ff, LNS, ln_eps=1e-5, bias=False):
        super().__init__()
        # Pass dim_emb so PyTorch knows the normalization parameter sizes
        self.ln1 = RMSLayerNorm(dim_emb, epsilon=ln_eps)
        self.ln2 = RMSLayerNorm(dim_emb, epsilon=ln_eps)
        self.mha = PyramidMixedCausalAttention(dim_emb, num_heads, LNS)
        self.ffn = MixedFFN(dim_emb, d_ff, LNS, bias=bias)

    def forward(self, x, Lq):
        z = self.mha(self.ln1(x), Lq) + x[:, -Lq:]  # residual aligns to tail
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
        self.Lq_list = list(range(LS + LNS, LNS, -4))
        self.Lq_list.append(LNS)

        self.blocks = nn.ModuleList(
            [
                OneTransBlock(dim_emb, num_heads, d_ff, LNS=LNS, bias=bias)
                for _ in range(len(self.Lq_list))
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

    def forward(
        self, sparse_inputs: torch.Tensor, dense_inputs: torch.Tensor
    ) -> torch.Tensor:
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
    DIM_INPUT_SPARSE = len(NUM_SPARSE_EMBS)
    DIM_INPUT_DENSE = 13

    # Initialize the model
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

    # Generate dummy input data
    sparse_inputs = torch.stack(
        [
            torch.randint(0, high=NUM_SPARSE_EMBS[i], size=(BATCH_SIZE,))
            for i in range(DIM_INPUT_SPARSE)
        ],
        dim=1,
    ).to(torch.int32)

    dense_inputs = torch.rand((BATCH_SIZE, DIM_INPUT_DENSE), dtype=torch.float32)

    print("Sparse input shape:", sparse_inputs.shape)
    print("Dense input shape:", dense_inputs.shape)

    # Run a forward pass
    outputs = model(sparse_inputs, dense_inputs)
    print("Model output shape:", outputs.shape)
