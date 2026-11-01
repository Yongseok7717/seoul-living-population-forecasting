"""Forecasting models. All take normalised inputs:
    x_hist [B, k, N], tf_hist [B, k, 8], tf_fut [B, H, 8], static [N, S]
and return [B, H, N].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_to_torch(m):
    m = m.tocoo()
    idx = torch.tensor([m.row, m.col], dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape).coalesce()


class TemporalEncoder(nn.Module):
    """Shared GRU over each cell's history; input per step = [z, time feats, static]."""
    def __init__(self, n_static, hidden):
        super().__init__()
        self.inp = nn.Linear(1 + 8 + n_static, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x_hist, tf_hist, static):
        B, k, N = x_hist.shape
        tfe = tf_hist[:, :, None, :].expand(B, k, N, 8)
        st = static[None, None].expand(B, k, N, -1)
        inp = torch.cat([x_hist[..., None], tfe, st], dim=-1)          # [B,k,N,F]
        inp = self.inp(inp).permute(0, 2, 1, 3).reshape(B * N, k, -1)   # [B*N,k,hid]
        _, h = self.gru(inp)
        return h[-1].view(B, N, -1)                                     # [B,N,hid]


class Head(nn.Module):
    """Node hidden + future time features -> H outputs (residual on last observed value)."""
    def __init__(self, hidden, H):
        super().__init__()
        self.tf_proj = nn.Linear(H * 8, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, H))

    def forward(self, h, tf_fut, last):
        B, N, d = h.shape
        t = self.tf_proj(tf_fut.reshape(B, -1))[:, None, :].expand(B, N, d)
        out = self.mlp(torch.cat([h, t], dim=-1))                       # [B,N,H]
        return out.permute(0, 2, 1) + last[:, None, :]                  # [B,H,N]


class GraphBranch(nn.Module):
    def __init__(self, hidden, A, layers=2):
        super().__init__()
        self.register_buffer("A", A)
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    def forward(self, h):
        B, N, d = h.shape
        for lin in self.lins:
            hp = torch.sparse.mm(self.A, h.permute(1, 0, 2).reshape(N, -1)).view(N, B, d).permute(1, 0, 2)
            h = h + F.gelu(lin(hp))
        return h


class HyperBranch(nn.Module):
    """node -> hyperedge (mean) -> node (mean) message passing over facility service regions."""
    def __init__(self, hidden, H_inc):
        super().__init__()
        Hc = H_inc.tocsr()
        dv = torch.tensor(1.0 / (Hc.sum(1).A.ravel() + 1e-6), dtype=torch.float32)
        de = torch.tensor(1.0 / (Hc.sum(0).A.ravel() + 1e-6), dtype=torch.float32)
        self.register_buffer("H", sparse_to_torch(Hc))
        self.register_buffer("Ht", sparse_to_torch(Hc.T))
        self.register_buffer("dv", dv); self.register_buffer("de", de)
        self.edge = nn.Linear(hidden, hidden)
        self.node = nn.Linear(hidden, hidden)

    def forward(self, h):
        B, N, d = h.shape
        hn = h.permute(1, 0, 2).reshape(N, -1)
        he = torch.sparse.mm(self.Ht, hn) * self.de[:, None]           # [E, B*d]
        he = F.gelu(self.edge(he.view(-1, B, d)))
        hv = torch.sparse.mm(self.H, he.reshape(-1, B * d)) * self.dv[:, None]
        return F.gelu(self.node(hv.view(N, B, d).permute(1, 0, 2)))


class Forecaster(nn.Module):
    """variant in {'gru', 'gru_graph', 'gru_hyper', 'gru_graph_hyper'}"""
    def __init__(self, variant, n_static, H, hidden=64, A=None, H_inc=None):
        super().__init__()
        self.variant = variant
        self.enc = TemporalEncoder(n_static, hidden)
        self.graph = GraphBranch(hidden, A) if "graph" in variant else None
        self.hyper = HyperBranch(hidden, H_inc) if "hyper" in variant else None
        if self.hyper is not None:
            self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
        self.head = Head(hidden, H)

    def forward(self, x_hist, tf_hist, tf_fut, static):
        h = self.enc(x_hist, tf_hist, static)
        hg = self.graph(h) if self.graph is not None else h
        if self.hyper is not None:
            hh = self.hyper(h)
            g = self.gate(torch.cat([hg, hh], -1))
            hg = hg + g * hh
        return self.head(hg, tf_fut, x_hist[:, -1])
