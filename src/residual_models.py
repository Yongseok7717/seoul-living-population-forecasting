"""Residual forecaster with pluggable relational branches (Task 4.3 ablation).

    x_hist [B,k,N,2] (+ calendar, static) --GRU--> h [B,N,d]
    h* = h + sum_o  g_o * MLP_o(P_o h)          o in relational operators (spatial / func_knn / func_hard / func_soft /
                                                 func_dynamic (weekday|weekend by target daytype) / facility)
    head(h*, future calendar, future baseline per node) -> [B,H,N] normalised relative residual
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import sparse_to_torch


class HyperProp(nn.Module):
    """P v = Dv^-1/2 H W De^-1 H^T Dv^-1/2 v, matrix-free."""
    def __init__(self, H, w):
        super().__init__()
        Hc = H.tocsr(); w = np.asarray(w, np.float32)
        de = np.asarray(Hc.sum(0)).ravel(); dv = np.asarray((Hc @ np.diag(w) if Hc.shape[1] < 2000 else Hc.multiply(w[None, :])).sum(1)).ravel()
        self.register_buffer("H", sparse_to_torch(Hc)); self.register_buffer("Ht", sparse_to_torch(Hc.T.tocsr()))
        self.register_buffer("dvis", torch.tensor(np.where(dv > 0, 1 / np.sqrt(np.maximum(dv, 1e-9)), 0.0), dtype=torch.float32)[:, None])
        self.register_buffer("wde", torch.tensor(w * np.where(de > 0, 1 / np.maximum(de, 1e-9), 0.0), dtype=torch.float32)[:, None])

    def forward(self, v):                                              # v [N, M]
        return self.dvis * torch.sparse.mm(self.H, self.wde * torch.sparse.mm(self.Ht, self.dvis * v))


class SparseProp(nn.Module):
    def __init__(self, P):
        super().__init__(); self.register_buffer("P", sparse_to_torch(P))

    def forward(self, v):
        return torch.sparse.mm(self.P, v)


class RelationalBranch(nn.Module):
    def __init__(self, hidden, op):
        super().__init__()
        self.kind = op["kind"]
        if self.kind == "sparse":
            self.props = nn.ModuleList([SparseProp(op["P"])])
        elif self.kind == "hyper":
            self.props = nn.ModuleList([HyperProp(op["H"], op["w"])])
        elif self.kind == "hyper_dynamic":
            self.props = nn.ModuleList([HyperProp(h, w) for h, w in zip(op["H"], op["w"])])
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())

    def forward(self, h, weekend_mask=None):                          # h [B,N,d]
        B, N, d = h.shape
        v = h.permute(1, 0, 2).reshape(N, B * d)
        if self.kind == "hyper_dynamic":
            m0 = self.props[0](v).view(N, B, d); m1 = self.props[1](v).view(N, B, d)
            sel = weekend_mask.view(1, B, 1).float()
            msg = (1 - sel) * m0 + sel * m1
        else:
            msg = self.props[0](v).view(N, B, d)
        msg = msg.permute(1, 0, 2)
        msg = self.mlp(msg)
        g = self.gate(torch.cat([h, msg], -1))
        return g * msg


class TransformerBackbone(nn.Module):
    def __init__(self, hidden, k, layers=2, heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, k, hidden)); nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, seq):                                            # [B*N, k, hidden]
        z = self.enc(seq + self.pos[:, -seq.shape[1]:])
        return self.norm(z[:, -1])


class LinearBackbone(nn.Module):
    """shared linear map from the flattened per-node history (after the input projection) to a hidden vector."""
    def __init__(self, hidden, k):
        super().__init__(); self.lin = nn.Linear(k * hidden, hidden)

    def forward(self, seq):
        return self.lin(seq.reshape(seq.shape[0], -1))


class MLPBackbone(nn.Module):
    """nonlinear map from the flattened per-node history: Linear(k*hidden -> 2*hidden) -> GELU -> Linear(-> hidden)."""
    def __init__(self, hidden, k):
        super().__init__(); self.net = nn.Sequential(nn.Linear(k * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, seq):
        return self.net(seq.reshape(seq.shape[0], -1))


class ResidualForecaster(nn.Module):
    def __init__(self, n_static, H, ops, hidden=64, n_cal=11, backbone="gru", k=24, n_cond=0):
        super().__init__()
        self.H = H; self.backbone = backbone
        # Task 14-H: FiLM conditioning of the temporal code by a per-node vector (cluster one-hot, calendar), applied before the relational branches
        self.film = nn.Sequential(nn.Linear(n_cond, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden)) if n_cond else None
        self.inp = nn.Linear(2 + n_cal + n_static, hidden)
        if backbone == "gru":
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        elif backbone == "transformer":
            self.tfm = TransformerBackbone(hidden, k)
        elif backbone == "linear":
            self.linb = LinearBackbone(hidden, k)
        elif backbone == "mlp":
            self.linb = MLPBackbone(hidden, k)
        else:
            raise ValueError(backbone)
        self.branches = nn.ModuleList([RelationalBranch(hidden, op) for op in ops])
        self.tf_proj = nn.Linear(H * n_cal, hidden)
        self.bf_proj = nn.Linear(H, hidden)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, H))
        self.decay = nn.Sequential(nn.Linear(hidden, H), nn.Sigmoid())   # per-node, per-horizon persistence weight

    def forward(self, x_hist, tf_hist, tf_fut, b_fut, static, weekend_mask=None, cond=None):
        B, k, N, _ = x_hist.shape
        inp = torch.cat([x_hist, tf_hist[:, :, None, :].expand(B, k, N, -1), static[None, None].expand(B, k, N, -1)], -1)
        inp = self.inp(inp).permute(0, 2, 1, 3).reshape(B * N, k, -1)
        if self.backbone == "gru":
            _, h = self.gru(inp); h = h[-1].view(B, N, -1)
        elif self.backbone == "transformer":
            h = self.tfm(inp).view(B, N, -1)
        else:
            h = self.linb(inp).view(B, N, -1)
        if self.film is not None and cond is not None:
            gb = self.film(cond); h = (1 + gb[..., :h.shape[-1]]) * h + gb[..., h.shape[-1]:]
        for br in self.branches:
            h = h + br(h, weekend_mask)
        t = self.tf_proj(tf_fut.reshape(B, -1))[:, None, :].expand(B, N, -1)
        bfe = self.bf_proj(b_fut.permute(0, 2, 1))                    # [B,N,hidden] from [B,H,N]
        out = self.head(torch.cat([h, t, bfe], -1))                    # [B,N,H]
        # persistence skip: predict the change from the last observed residual (decayed towards 0 by a learned gate)
        last = x_hist[:, -1, :, 0]                                     # [B,N]
        return out.permute(0, 2, 1) + self.decay(h).permute(0, 2, 1) * last[:, None, :]
