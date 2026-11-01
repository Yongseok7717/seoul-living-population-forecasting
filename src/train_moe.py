"""Task 13: extreme-aware Mixture-of-Experts on the rolling-baseline residual (= two-stage hybrid).

    python train_moe.py --name moe_k24_mlp --k 24 --normal linear --extreme mlp --extreme_rel none
    python train_moe.py --name moe_k24_mlp_func --k 24 --extreme mlp --extreme_rel func_knn      # reuses cached normal expert + gates

Stage 1  normal expert f_N (linear AR by default), masked loss on non-event targets (1 - e); optionally full loss.
Stage 2  extreme expert f_E (mlp | gru | linear, relations none | spatial | func_knn), masked Huber loss on event targets.
Stage A  oracle mixture  z_or = (1 - e) z_N + e z_E  with the true future event indicator (upper bound).
Stage B  learned gate p = sigma(g(history, event state, calendar, static)): linear backbone; weighted BCE or focal (gamma 1, 2).
         soft mix (1 - p) z_N + p z_E, hard mix with tau in {0.3, 0.5, 0.7} chosen on validation; calibration (Brier, ECE,
         reliability diagram), Platt and isotonic calibration fitted on validation and applied to test.
Events   --event_def zev: |z_ev| > 3 and |X - B| >= 30 (z_ev = (X - B)/(sigma_MAD + sqrt(B + 1)));  q95 / q99: |R| > Q_q(|R|_train).
Metrics  train_residual.metrics (overall + extreme) and regime MAE_N (e = 0), MAE_E (e = 1) in people.
Experts and gates are cached in results/base/results/moe_cache/ by (kind, config, k, event_def) so ablations reuse them.
"""
import argparse, gc, json, time, sys, hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_data import ResidualData, ML_DIR, RES_DIR, SPLITS, months_between   # noqa: E402
import config as _cfg  # noqa: E402
C_OUT = _cfg.OUT
from train_residual import metrics                                                # noqa: E402

import os as _os
RES = Path(_os.environ["LP_RESULTS"]) if _os.environ.get("LP_RESULTS") else ML_DIR / "results"
CACHE = Path(_os.environ["LP_CACHE"]) if _os.environ.get("LP_CACHE") else ML_DIR / "results" / "moe_cache"


class MoEData:
    """ResidualData + event labels + event-state channels for the gate."""
    def __init__(self, k, H, event_def, prior="", rel=""):
        self.d = ResidualData(k=k, horizon=H); d = self.d; self.k, self.H = k, H; self.RAIN24 = {}; self.rel = rel; self.ZP, self.EP = {}, {}
        self.E, self.G = {}, {}
        # refined definitions (T2): q95c = q95 cell-hour that is spatially clustered (>= 3 q95 cells in its rook neighbourhood incl. itself,
        # same hour) OR persists (q95 in the previous or next hour of the same cell); q95s = both conditions (strict)
        base_def, refine = event_def, ""
        if event_def.endswith(("c", "s", "k")) and event_def[:-1].startswith("q") and event_def[1:-1].isdigit():
            base_def, refine = event_def[:-1], event_def[-1]
        self.event_def, self.refine = event_def, refine
        if base_def.startswith("q"):
            q = float(base_def[1:]) / 100; tr = np.abs(d.RP["train"]); self.thr = float(np.nanquantile(tr[~np.isnan(tr[:, 0])], q))
        if refine:
            from loader import load_adjacency
            A = load_adjacency("rook")[d.cells][:, d.cells].tocsr(); A.data[:] = 1.0; A = A + sp.eye(d.N, format="csr")
        for split, (a, b) in SPLITS.items():
            X, B, RP = d.X[split], d.B[split], d.RP[split]
            if event_def == "zev":
                SC = np.concatenate([np.load(RES_DIR / f"scale_{ym}.npy") for ym in months_between(a, b)])[:, d.cells].astype(np.float32)
                zev = (X - B) / (SC + np.sqrt(np.maximum(B, 0) + 1)); E = (np.abs(zev) > 3) & (np.abs(RP) >= 30)
            else:
                E = np.abs(RP) > self.thr
            E = np.where(np.isnan(RP), False, E)
            if refine:
                Ef = E.astype(np.float32); n_sp = np.asarray((A @ Ef.T).T)                      # number of flagged cells in the rook neighbourhood (incl. self)
                spatial = E & (n_sp >= 3)
                prev = np.zeros_like(E); prev[1:] = E[:-1]; nxt = np.zeros_like(E); nxt[:-1] = E[1:]; temporal = E & (prev | nxt)
                E = spatial if refine == "k" else ((spatial | temporal) if refine == "c" else (spatial & temporal))
            self.E[split] = E
            cs = np.concatenate([np.zeros((1, d.N), np.int32), np.cumsum(E, 0, dtype=np.int32)]); T = len(E); ar = np.arange(T)
            n24 = (cs[ar + 1] - cs[np.maximum(ar - 23, 0)]) / 24.0; n168 = (cs[ar + 1] - cs[np.maximum(ar - 167, 0)]) / 168.0
            self.G[split] = np.stack([E.astype(np.float32), n24.astype(np.float32), n168.astype(np.float32)], -1)   # [T, N, 3]
        self.rate = {s: float(self.E[s][~np.isnan(self.d.RP[s][:, 0])].mean()) for s in self.E}
        # external priors (build_priors.py): calendar [T,9] (all cells), rain [T,N] causal + 'forecast', venue window [T,N]
        self.prior = prior; PR = C_OUT / "priors"; self.CAL, self.RAIN, self.RAINOK, self.VEN, self.DMSG = {}, {}, {}, {}, {}
        if prior:
            for split, (a, b) in SPLITS.items():
                ms = months_between(a, b)
                self.CAL[split] = np.concatenate([np.load(PR / f"prior_cal_{m}.npy") for m in ms]).astype(np.float32)
                self.RAIN[split] = np.concatenate([np.load(PR / f"prior_rain_{m}.npy") for m in ms])[:, d.cells].astype(np.float32)
                self.RAINOK[split] = np.concatenate([np.load(PR / f"prior_rainok_{m}.npy") for m in ms]).astype(np.float32)
                self.VEN[split] = np.concatenate([np.load(PR / f"prior_venue_{m}.npy") for m in ms])[:, d.cells].astype(np.float32)
                if "dmsg" in prior:                                                                     # disaster-message windows (all / weather / crowd), [T, N, 3]
                    self.DMSG[split] = np.concatenate([np.load(PR / f"prior_dmsg_{m}.npy") for m in ms])[:, d.cells].astype(np.float16)
                R = self.RAIN[split]; cs = np.concatenate([np.zeros((1, d.N), np.float32), np.cumsum(R, 0)]); ar = np.arange(len(R))
                self.RAIN24[split] = (cs[ar + 1] - cs[np.maximum(ar - 23, 0)]).astype(np.float32) if hasattr(self, "RAIN24") else None
        # relational context for the gate (Task 14-G): neighbours' residual z and event share propagated by P (row-stochastic), per operator
        if rel:
            for name in rel.split(","):
                P = prop_matrix(d, name); self.ZP[name], self.EP[name] = {}, {}
                for split in SPLITS:
                    z = np.nan_to_num(np.clip(d.RP[split] / d.r_std, -20, 20)).astype(np.float32)
                    self.ZP[name][split] = np.asarray((P @ z.T).T, np.float16); self.EP[name][split] = np.asarray((P @ self.E[split].astype(np.float32).T).T, np.float16)
                print("relational context", name, "nnz/row", round(P.nnz / d.N, 1), flush=True)

    def batch(self, split, ts):
        xh, th, tfu, bf, y, dtf = self.d.batch(split, ts)
        ye = np.stack([self.E[split][t + 1:t + 1 + self.H] for t in ts]).astype(np.float32)
        xg = np.stack([self.G[split][t - self.k + 1:t + 1] for t in ts]).astype(np.float32)          # [B,k,N,3]
        if self.rel:
            for name in self.rel.split(","):
                zp = np.stack([self.ZP[name][split][t - self.k + 1:t + 1] for t in ts]).astype(np.float32); ep = np.stack([self.EP[name][split][t - self.k + 1:t + 1] for t in ts]).astype(np.float32)
                xg = np.concatenate([xg, zp[..., None], ep[..., None]], -1)
        if self.prior:
            pr = self.prior.split(",")
            if "cal" in pr:                                                                          # per-hour calendar priors appended to the future calendar
                tfu = np.concatenate([tfu, np.stack([self.CAL[split][t + 1:t + 1 + self.H] for t in ts])], -1)
                th = np.concatenate([th, np.stack([self.CAL[split][t - self.k + 1:t + 1] for t in ts])], -1)
            if "rain" in pr or "rainfc" in pr:                                                       # causal rain: hourly rain and 24-h sum up to t (history channels)
                rh = np.stack([self.RAIN[split][t - self.k + 1:t + 1] for t in ts]); r24 = np.stack([self.RAIN24[split][t - self.k + 1:t + 1] for t in ts])
                xg = np.concatenate([xg, np.log1p(rh)[..., None], np.log1p(r24)[..., None]], -1)
            xf = []
            if "rainfc" in pr:                                                                       # 'perfect forecast' upper bound: rain at the target hour
                xf.append(np.log1p(np.stack([self.RAIN[split][t + 1:t + 1 + self.H] for t in ts])))
            if "venue" in pr:
                xf.append(np.stack([self.VEN[split][t + 1:t + 1 + self.H] for t in ts]))
            if "dmsg" in pr:
                D = np.stack([self.DMSG[split][t + 1:t + 1 + self.H] for t in ts]).astype(np.float32)
                xf += [D[..., c] for c in range(D.shape[-1])]
            xf = np.stack(xf, -1).astype(np.float32) if xf else np.zeros((len(ts), self.H, self.d.N, 0), np.float32)
            return xh, th, tfu, bf, y, ye, xg, dtf, xf
        return xh, th, tfu, bf, y, ye, xg, dtf, np.zeros((len(ts), self.H, self.d.N, 0), np.float32)


def prop_matrix(d, name):
    """Row-stochastic propagation matrix on the analysis subset for the gate's relational context (Task 14-D/E).
    geo = rook adjacency; mob_lag = inferred lag dependency (context of j = its leaders i, W^T); mob_ext = extreme transition (W^T);
    trans = subway-connectivity graph; trans_hyper = station catchments + line areas (H W H^T); func_knn = functional kNN (Task 3)."""
    from residual_data import load_sparse, REL
    c = d.cells
    files = {"geo": ("spatial_rook.npz", "W", False), "mob_lag": ("mob_lag.npz", "W", True), "mob_ext": ("mob_ext.npz", "W", True), "mob_lag_local": ("mob_lag_local.npz", "W", True), "trans": ("trans.npz", "W", False), "func_knn": ("func_similarity_knn10.npz", "W", False)}
    if name == "trans_hyper":
        Hm = load_sparse(REL / "trans_hyper.npz", "H")[c].tocsr(); w = np.load(REL / "trans_hyper.npz")["w"]
        de = np.asarray(Hm.sum(0)).ravel(); W = (Hm @ sp.diags(w / np.maximum(de, 1)) @ Hm.T).tocsr(); W.setdiag(0); W.eliminate_zeros()
    else:
        f, key, transpose = files[name]; W = load_sparse(REL / f, key)[c][:, c].tocsr()
        if transpose:
            W = W.T.tocsr()
    W.data = np.abs(W.data); dsum = np.asarray(W.sum(1)).ravel()
    return (sp.diags(1 / np.maximum(dsum, 1e-9)) @ W).tocsr().astype(np.float32)


def build_gate(n_static, H, k, hidden, backbone="linear", n_g=3, n_cal=11, n_f=0, film_K=0, d_f=8, attn=False, attn_cond="none", K_cl=6):
    import torch, torch.nn as nn
    from residual_models import LinearBackbone, MLPBackbone

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(2 + n_g + n_cal + n_static, hidden); self.enc = MLPBackbone(hidden, k) if backbone == "mlp" else LinearBackbone(hidden, k)
            self.tf_proj = nn.Linear(H * n_cal, hidden); self.n_f = n_f
            self.f_proj = nn.Linear(H * n_f, hidden) if n_f else None
            self.head = nn.Sequential(nn.Linear((3 if n_f else 2) * hidden, hidden), nn.GELU(), nn.Linear(hidden, H))
            # Task 14-G functional conditioning of the gate: FiLM of the temporal code by (cluster embedding, calendar now / targets)
            self.film = nn.Sequential(nn.Linear(d_f + 2 * n_cal, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden)) if film_K else None
            self.e_c = nn.Embedding(film_K, d_f) if film_K else None
            # Task 14-F dynamic attention over a fixed candidate set (mobility graph): alpha_ij(t) = softmax_j(q_i(t) . k_j(t)); q optionally conditioned on calendar / function
            self.attn = attn; self.attn_cond = attn_cond
            if attn:
                d_a = 32; n_q = hidden + {"none": 0, "cal": 2 * n_cal, "func": d_f}[attn_cond]
                self.a_q = nn.Linear(n_q, d_a); self.a_k = nn.Linear(hidden, d_a); self.a_v = nn.Linear(hidden, hidden)
                self.e_a = nn.Embedding(K_cl, d_f) if attn_cond == "func" else None
                self.head = nn.Sequential(nn.Linear((4 if n_f else 3) * hidden, hidden), nn.GELU(), nn.Linear(hidden, H))

        def encode(self, xh, xg, th, static):
            B, kk, N, _ = xh.shape
            inp = torch.cat([xh, xg, th[:, :, None, :].expand(B, kk, N, -1), static[None, None].expand(B, kk, N, -1)], -1)
            return torch.nn.functional.gelu(self.enc(self.inp(inp).permute(0, 2, 1, 3).reshape(B * N, kk, -1))).view(B, N, -1)

        def forward(self, xh, xg, th, tfu, static, xf=None, cl=None, h_nb=None, return_attn=False):
            B, kk, N, _ = xh.shape
            h = self.encode(xh, xg, th, static)
            if self.film is not None:
                q = torch.cat([self.e_c(cl)[None].expand(B, -1, -1), torch.cat([th[:, -1], tfu.mean(1)], -1)[:, None, :].expand(-1, N, -1)], -1)
                gb = self.film(q); h = (1 + gb[..., :h.shape[-1]]) * h + gb[..., h.shape[-1]:]
            t = self.tf_proj(tfu.reshape(B, -1))[:, None, :].expand(B, N, -1)
            parts = [h, t]; alpha = None
            if self.attn:
                cs = torch.cat([th[:, -1], tfu.mean(1)], -1)[:, None, :].expand(-1, N, -1)
                qin = h if self.attn_cond == "none" else (torch.cat([h, cs], -1) if self.attn_cond == "cal" else torch.cat([h, self.e_a(cl)[None].expand(B, -1, -1)], -1))
                q = self.a_q(qin); kk_ = self.a_k(h_nb)                                              # h_nb [B,N,M,hidden] neighbour codes
                alpha = torch.softmax((q[:, :, None, :] * kk_).sum(-1) / kk_.shape[-1] ** 0.5, -1)   # [B,N,M]
                parts.append((alpha[..., None] * self.a_v(h_nb)).sum(2))
            if self.f_proj is not None:
                parts.append(self.f_proj(xf.permute(0, 2, 1, 3).reshape(B, N, -1)))                 # xf [B,H,N,n_f] -> [B,N,H*n_f]
            out = self.head(torch.cat(parts, -1)).permute(0, 2, 1)                                 # logits [B,H,N]
            return (out, alpha) if return_attn else out
    return Gate()


def ece_score(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1); ece = 0.0; rel = []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean()); ece += m.mean() * abs(conf - acc); rel.append((b, conf, acc, int(m.sum())))
    return float(ece), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--k", type=int, default=24); ap.add_argument("--H", type=int, default=24); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--normal", default="linear", choices=["linear", "gru", "mlp"]); ap.add_argument("--normal_mask", default="masked", choices=["masked", "full"])
    ap.add_argument("--extreme", default="mlp", choices=["mlp", "gru", "linear"]); ap.add_argument("--extreme_rel", default="none", choices=["none", "spatial", "func_knn"])
    ap.add_argument("--gate_losses", default="wbce,focal1,focal2"); ap.add_argument("--event_def", default="zev", choices=["zev", "q95", "q99"])
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--node_sample", type=int, default=2000); ap.add_argument("--eval_stride", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None); ap.add_argument("--no_reuse", action="store_true")
    ap.add_argument("--gate_backbone", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--prior", default="", help="comma list of gate priors: cal, rain (causal), rainfc (target-hour rain, upper bound), venue")
    ap.add_argument("--gate_rel", default="", help="Task 14-G relational context channels for the gate: geo, mob_lag, mob_ext, trans, trans_hyper, func_knn (comma list)")
    ap.add_argument("--gate_film", action="store_true", help="Task 14-G functional conditioning of the gate (FiLM by cluster embedding x calendar)")
    ap.add_argument("--expert_cond", default="none", choices=["none", "film"], help="Task 14-H: FiLM-condition both experts on (cluster, calendar)")
    ap.add_argument("--gate_attn", default="", help="Task 14-F: dynamic attention over the top-10 candidate neighbours of this operator (geo, mob_lag, mob_lag_local, trans, ...)")
    ap.add_argument("--attn_cond", default="none", choices=["none", "cal", "func"], help="conditioning of the attention query")
    args = ap.parse_args()
    import torch
    from residual_models import ResidualForecaster
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RES.mkdir(parents=True, exist_ok=True); CACHE.mkdir(exist_ok=True); t0 = time.time()
    md = MoEData(args.k, args.H, args.event_def, args.prior, args.gate_rel); data = md.d; static = torch.tensor(data.static, device=dev)
    CL = torch.tensor(data.cluster.astype(np.int64), device=dev); K_cl = int(data.cluster.max()) + 1
    NB = None
    if args.gate_attn:                                   # candidate neighbours: top-10 columns of the row-stochastic operator (padded with self)
        P = prop_matrix(data, args.gate_attn).tolil(); NB = np.zeros((data.N, 10), np.int64)
        for i in range(data.N):
            cols, vals = np.array(P.rows[i]), np.array(P.data[i]); order = np.argsort(-vals)[:10]; nb = cols[order] if len(cols) else np.array([], int)
            NB[i] = np.concatenate([nb, np.full(10 - len(nb), i)])
        NB = torch.tensor(NB, device=dev); print("attention candidates from", args.gate_attn, flush=True)

    def gate_codes(gate, xh, xg, th, static_all):
        """neighbour codes for all nodes (no grad, chunked): [B,N,hidden]"""
        with torch.no_grad():
            return torch.cat([gate.encode(torch.tensor(xh[:, :, n0:n0 + 2000], device=dev), torch.tensor(xg[:, :, n0:n0 + 2000], device=dev), torch.tensor(th, device=dev), static_all[n0:n0 + 2000]) for n0 in range(0, xh.shape[2], 2000)], 1)

    def nb_codes(Hall, idx):
        return None if Hall is None else Hall[:, NB[idx]]                                          # [B,S,10,hidden]
    def cond_of(th, tfu, sel=None):                 # per-node conditioning vector for FiLM experts: [cluster one-hot; calendar now; mean target calendar]
        if args.expert_cond == "none":
            return None
        oh = np.eye(K_cl, dtype=np.float32)[data.cluster if sel is None else data.cluster[sel]]
        cs = np.concatenate([th[:, -1, :11], tfu[:, :, :11].mean(1)], -1)
        return torch.tensor(np.concatenate([np.repeat(oh[None], len(th), 0), np.repeat(cs[:, None, :], len(oh), 1)], -1), device=dev)
    print("data:", json.dumps(data.summary()), "event rate:", json.dumps(md.rate), flush=True)
    tr = data.windows("train", 1); ts_eval = {s: data.windows(s, args.eval_stride) for s in ("val", "test")}
    tag = f"k{args.k}_{args.event_def}" + ("" if args.seed == 0 else f"_s{args.seed}")
    ms = "" if args.max_steps is None else f"_ms{args.max_steps}"

    # ---------------- generic training / prediction ----------------
    def base_cal(th, tfu):
        return th[..., :11], tfu[..., :11]

    def train_forecaster(model, ops, mask_mode, key):
        """mask_mode: 'normal' (weight 1-e), 'extreme' (weight e), 'full' (all). Huber on z."""
        pt = CACHE / f"{key}.pt"
        if pt.exists() and not args.no_reuse:
            model.load_state_dict(torch.load(pt, map_location=dev)); print("reuse", key, flush=True); return model
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = (len(tr) // args.batch + 1) if not args.max_steps else args.max_steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps + 1); best = (np.inf, None)
        for ep in range(args.epochs):
            model.train(); perm = np.random.permutation(tr); losses = []; te = time.time()
            for step, i in enumerate(range(0, len(perm), args.batch)):
                if args.max_steps and step >= args.max_steps:
                    break
                xh, th, tfu, bf, y, ye, xg, dtf, xf = md.batch("train", perm[i:i + args.batch]); st = static; th, tfu = base_cal(th, tfu)
                wm = torch.tensor((dtf != 0).mean(1) > 0.5, device=dev)
                if not ops and args.node_sample:
                    sel = np.random.choice(data.N, args.node_sample, replace=False); xh, bf, y, ye, st = xh[:, :, sel], bf[:, :, sel], y[:, :, sel], ye[:, :, sel], static[sel]
                cq = cond_of(th, tfu, sel if (not ops and args.node_sample) else None)
                xh, th, tfu, bf, y, ye = (torch.tensor(v, device=dev) for v in (xh, th, tfu, bf, y, ye))
                pred = model(xh, th, tfu, bf, st, wm, cond=cq)
                w = {"normal": 1 - ye, "extreme": ye, "full": torch.ones_like(ye)}[mask_mode]
                loss = (w * torch.nn.functional.huber_loss(pred, y, reduction="none", delta=1.0)).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(loss.item())
            # validation masked loss (stride 48)
            model.eval(); vl, vw = 0.0, 0.0
            with torch.no_grad():
                tsv = data.windows("val", 48)
                for i in range(0, len(tsv), 8):
                    xh, th, tfu, bf, y, ye, xg, dtf, xf = md.batch("val", tsv[i:i + 8]); th, tfu = base_cal(th, tfu); pz = predict_forecaster(model, ops, xh, th, tfu, bf, dtf)
                    w = {"normal": 1 - ye, "extreme": ye, "full": np.ones_like(ye)}[mask_mode]; vl += float((w * np.abs(pz - y)).sum()); vw += float(w.sum())
            vl /= max(vw, 1); print(json.dumps({"expert": key, "epoch": ep, "train_loss": float(np.mean(losses)), "val_masked_mae_z": vl, "sec": time.time() - te}), flush=True)
            if vl < best[0]:
                best = (vl, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()})
        model.load_state_dict(best[1]); torch.save(model.state_dict(), pt); return model

    def predict_forecaster(model, ops, xh, th, tfu, bf, dtf):
        model.eval(); wm = torch.tensor((dtf != 0).mean(1) > 0.5, device=dev)
        with torch.no_grad():
            if ops:
                return model(*(torch.tensor(v, device=dev) for v in (xh, th, tfu, bf)), static, wm, cond=cond_of(th, tfu)).cpu().numpy()
            parts = [model(torch.tensor(xh[:, :, n0:n0 + 2000], device=dev), torch.tensor(th, device=dev), torch.tensor(tfu, device=dev), torch.tensor(bf[:, :, n0:n0 + 2000], device=dev), static[n0:n0 + 2000], wm, cond=cond_of(th, tfu, np.arange(n0, min(n0 + 2000, data.N)))) for n0 in range(0, data.N, 2000)]
            return torch.cat(parts, 2).cpu().numpy()

    def cached_preds(key, fn):
        """fn(split, ts) -> [n,H,N] float32; cached as float16 per split"""
        out = {}
        for s in ("val", "test"):
            f = CACHE / f"{key}_{s}_s{args.eval_stride}.npy"
            if f.exists() and not args.no_reuse:
                out[s] = np.load(f)
            else:
                out[s] = np.concatenate([fn(s, ts_eval[s][i:i + 8]) for i in range(0, len(ts_eval[s]), 8)]).astype(np.float16); np.save(f, out[s])
        return out

    # ---------------- experts ----------------
    ops_E = [data.operator(args.extreme_rel)] if args.extreme_rel != "none" else []
    ctag = "" if args.expert_cond == "none" else f"_{args.expert_cond}"; n_cond = 0 if args.expert_cond == "none" else K_cl + 22
    keyN = f"N_{args.normal}_{args.normal_mask}_{tag}{ctag}{ms}"; keyE = f"E_{args.extreme}_{args.extreme_rel}_{tag}{ctag}{ms}"
    fN = ResidualForecaster(data.static.shape[1], args.H, [], args.hidden, backbone=args.normal, k=args.k, n_cond=n_cond).to(dev)
    fN = train_forecaster(fN, [], "normal" if args.normal_mask == "masked" else "full", keyN)
    fE = ResidualForecaster(data.static.shape[1], args.H, ops_E, args.hidden, backbone=args.extreme, k=args.k, n_cond=n_cond).to(dev)
    fE = train_forecaster(fE, ops_E, "extreme", keyE)
    def pred_with(model, ops):
        def fn(s, ts):
            b = md.batch(s, ts); return predict_forecaster(model, ops, b[0], b[1][..., :11], b[2][..., :11], b[3], b[7])
        return fn
    zN = cached_preds(keyN, pred_with(fN, [])); zE = cached_preds(keyE, pred_with(fE, ops_E))
    YE = {s: np.concatenate([md.batch(s, ts_eval[s][i:i + 8])[5] for i in range(0, len(ts_eval[s]), 8)]) for s in ("val", "test")}
    DT = {s: np.concatenate([md.batch(s, ts_eval[s][i:i + 8])[7] for i in range(0, len(ts_eval[s]), 8)]) for s in ("val", "test")}
    del fN, fE; torch.cuda.empty_cache()

    # ---------------- gates ----------------
    pi = md.rate["train"]; gates = {}
    for gl in [g for g in args.gate_losses.split(",") if g]:
        _b = md.batch("train", tr[:1]); n_g, n_cal, n_f = _b[6].shape[-1], _b[2].shape[-1], _b[8].shape[-1]
        ptag = ("_p" + args.prior.replace(",", "+")) if args.prior else ""
        ptag += ("_r" + args.gate_rel.replace(",", "+")) if args.gate_rel else ""; ptag += "_film" if args.gate_film else ""
        ptag += (f"_a{args.gate_attn}" + ("" if args.attn_cond == "none" else f"_ac{args.attn_cond}")) if args.gate_attn else ""
        keyG = (f"G_{gl}_{args.gate_backbone}_{tag}{ptag}{ms}" if args.gate_backbone != "linear" else f"G_{gl}_{tag}{ptag}{ms}")
        gate = build_gate(data.static.shape[1], args.H, args.k, args.hidden, args.gate_backbone, n_g, n_cal, n_f, K_cl if args.gate_film else 0, attn=bool(args.gate_attn), attn_cond=args.attn_cond, K_cl=K_cl).to(dev); pt = CACHE / f"{keyG}.pt"
        if pt.exists() and not args.no_reuse:
            gate.load_state_dict(torch.load(pt, map_location=dev)); print("reuse", keyG, flush=True)
        else:
            opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
            steps = (len(tr) // args.batch + 1) if not args.max_steps else args.max_steps
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps + 1); best = (-np.inf, None)
            pw = torch.tensor((1 - pi) / pi, device=dev)
            for ep in range(args.epochs):
                gate.train(); perm = np.random.permutation(tr); losses = []
                for step, i in enumerate(range(0, len(perm), args.batch)):
                    if args.max_steps and step >= args.max_steps:
                        break
                    xh, th, tfu, bf, y, ye, xg, dtf, xf = md.batch("train", perm[i:i + args.batch])
                    Hall = gate_codes(gate, xh, xg, th, static) if NB is not None else None
                    sel = np.random.choice(data.N, args.node_sample, replace=False); xh, xg, ye, xf = xh[:, :, sel], xg[:, :, sel], ye[:, :, sel], xf[:, :, sel]
                    xh, xg, th, tfu, ye, xf = (torch.tensor(v, device=dev) for v in (xh, xg, th, tfu, ye, xf))
                    logit = gate(xh, xg, th, tfu, static[sel], xf, CL[sel], nb_codes(Hall, sel))
                    if gl == "wbce":
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, ye, pos_weight=pw)
                    else:
                        gamma = float(gl[-1]); alpha = 0.75; p = torch.sigmoid(logit); eps = 1e-6
                        loss = -(alpha * (1 - p) ** gamma * ye * torch.log(p + eps) + (1 - alpha) * p ** gamma * (1 - ye) * torch.log(1 - p + eps)).mean()
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step(); sched.step(); losses.append(loss.item())
                # val PR-AUC (stride 48, subsample)
                from sklearn.metrics import average_precision_score
                gate.eval(); pp, yy = [], []
                with torch.no_grad():
                    tsv = data.windows("val", 48)
                    for i in range(0, len(tsv), 8):
                        xh, th, tfu, bf, y, ye, xg, dtf, xf = md.batch("val", tsv[i:i + 8]); Hall = gate_codes(gate, xh, xg, th, static) if NB is not None else None
                        lg = torch.cat([gate(torch.tensor(xh[:, :, n0:n0 + 2000], device=dev), torch.tensor(xg[:, :, n0:n0 + 2000], device=dev), torch.tensor(th, device=dev), torch.tensor(tfu, device=dev), static[n0:n0 + 2000], torch.tensor(xf[:, :, n0:n0 + 2000], device=dev), CL[n0:n0 + 2000], nb_codes(Hall, np.arange(n0, min(n0 + 2000, data.N)))) for n0 in range(0, data.N, 2000)], 2)
                        pp.append(torch.sigmoid(lg).cpu().numpy().ravel()[::7]); yy.append(ye.ravel()[::7])
                ap_ = float(average_precision_score(np.concatenate(yy), np.concatenate(pp)))
                print(json.dumps({"gate": keyG, "epoch": ep, "train_loss": float(np.mean(losses)), "val_auprc": ap_}), flush=True)
                if ap_ > best[0]:
                    best = (ap_, {k_: v.detach().cpu().clone() for k_, v in gate.state_dict().items()})
            gate.load_state_dict(best[1]); torch.save(gate.state_dict(), pt)

        def gpred(s, ts, gate=gate):
            xh, th, tfu, bf, y, ye, xg, dtf, xf = md.batch(s, ts); gate.eval()
            with torch.no_grad():
                Hall = gate_codes(gate, xh, xg, th, static) if NB is not None else None
                lg = torch.cat([gate(torch.tensor(xh[:, :, n0:n0 + 2000], device=dev), torch.tensor(xg[:, :, n0:n0 + 2000], device=dev), torch.tensor(th, device=dev), torch.tensor(tfu, device=dev), static[n0:n0 + 2000], torch.tensor(xf[:, :, n0:n0 + 2000], device=dev), CL[n0:n0 + 2000], nb_codes(Hall, np.arange(n0, min(n0 + 2000, data.N)))) for n0 in range(0, data.N, 2000)], 2)
            return lg.cpu().numpy()
        gates[gl] = cached_preds(keyG + "_logit", gpred); del gate; torch.cuda.empty_cache()

    # ---------------- evaluation ----------------
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    result = {"name": args.name, "args": vars(args), "event_rate": md.rate, "data": data.summary(), "models": {}, "gates": {}}
    rng = np.random.default_rng(0)

    def regime(split, pz):
        Xh, Rh, X, R, B = data.people(split, ts_eval[split], pz); er = np.abs(Rh - R); e = YE[split] > 0
        return {"MAE_N": float(er[~e].mean()), "MAE_E": float(er[e].mean()), "share_E": float(e.mean())}

    def evaluate(label, preds):
        # preds[s] may be an array or a zero-argument callable (mixture built one split at a time to bound peak memory)
        result["models"][label] = {}
        for s in ("val", "test"):
            pz = preds[s]() if callable(preds[s]) else preds[s]; pz = pz.astype(np.float32); reg = regime(s, pz); gc.collect()
            m = metrics(data, s, ts_eval[s], pz, DT[s]); m.update(reg); result["models"][label][s] = m; del pz; gc.collect()
            print(s, label, json.dumps({k_: round(m[k_], 4) for k_ in ("X_wmape", "R_mae", "MAE_N", "MAE_E", "R_rel_err_top1pct", "extreme_recall_top1", "extreme_f1_top1", "auprc_top1", "peak_magnitude_rel_err", "peak_timing_mae_h", "top5pct_spatial_jaccard")}), flush=True)

    evaluate("normal_only", zN); evaluate("extreme_only", zE)
    evaluate("oracle", {s: (lambda s=s: (1 - YE[s]) * zN[s].astype(np.float32) + YE[s] * zE[s].astype(np.float32)) for s in zN})
    for gl, LG in gates.items():
        P = {s: (1 / (1 + np.exp(-LG[s].astype(np.float32)))).astype(np.float32) for s in LG}; g = {}
        for s in P:
            y = YE[s].ravel(); p = P[s].ravel(); sub = rng.choice(len(y), min(len(y), 4_000_000), replace=False)
            ece, rel = ece_score(p[sub], y[sub]); g[s] = {"auprc": float(average_precision_score(y[sub], p[sub])), "auroc": float(roc_auc_score(y[sub], p[sub])), "brier": float(brier_score_loss(y[sub], p[sub])), "ece": ece, "reliability": rel, "mean_p": float(p.mean()), "event_rate": float(y.mean())}
        # calibration fitted on validation, applied to test
        yv = YE["val"].ravel(); lv = LG["val"].astype(np.float32).ravel(); sub = rng.choice(len(yv), 2_000_000, replace=False)
        platt = LogisticRegression(C=1e6, max_iter=200).fit(lv[sub, None], yv[sub]); iso = IsotonicRegression(out_of_bounds="clip").fit(1 / (1 + np.exp(-lv[sub])), yv[sub])
        Pc = {"platt": {s: platt.predict_proba(LG[s].astype(np.float32).reshape(-1, 1))[:, 1].reshape(LG[s].shape).astype(np.float32) for s in LG},
              "isotonic": {s: iso.predict(P[s].ravel()).reshape(P[s].shape).astype(np.float32) for s in P}}
        for cal, PP in Pc.items():
            for s in PP:
                y = YE[s].ravel(); p = PP[s].ravel(); sub2 = rng.choice(len(y), min(len(y), 4_000_000), replace=False); ece, _ = ece_score(p[sub2], y[sub2])
                g[s][f"brier_{cal}"] = float(brier_score_loss(y[sub2], p[sub2])); g[s][f"ece_{cal}"] = ece
        result["gates"][gl] = g; print("gate", gl, json.dumps({s: {k_: round(v, 4) for k_, v in g[s].items() if k_ != "reliability"} for s in g}), flush=True)
        mix = lambda W, s: (lambda: (1 - W[s]) * zN[s].astype(np.float32) + W[s] * zE[s].astype(np.float32))
        evaluate(f"soft_{gl}", {s: mix(P, s) for s in P})
        evaluate(f"soft_{gl}_platt", {s: mix(Pc["platt"], s) for s in P})
        del Pc["isotonic"]; gc.collect()
        # hard gate: tau chosen on validation by R_mae
        cand = {}
        for tau in (0.3, 0.5, 0.7):
            pv = np.where(P["val"] > tau, zE["val"], zN["val"]).astype(np.float32); cand[tau] = metrics(data, "val", ts_eval["val"], pv, DT["val"])["R_mae"]
        tau = min(cand, key=cand.get); result["gates"][gl]["hard_tau_val_R_mae"] = {str(t_): float(v) for t_, v in cand.items()}; result["gates"][gl]["tau_selected"] = tau
        evaluate(f"hard_{gl}_tau{tau}", {s: (lambda s=s: np.where(P[s] > tau, zE[s], zN[s]).astype(np.float32)) for s in P})
        del P, Pc, pv; gc.collect()
        (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")   # partial save after each gate
    # reliability figure (val, first gate)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for gl, g in result["gates"].items():
            rel = g["val"]["reliability"]; ax.plot([r[1] for r in rel], [r[2] for r in rel], marker="o", label=f"{gl} (ECE {g['val']['ece']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("predicted p"); ax.set_ylabel("observed event rate"); ax.set_title(f"gate reliability (val), {args.name}"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(RES / f"fig_{args.name}_reliability.png", dpi=120); plt.close(fig)
    except Exception as ex:
        print("figure failed", ex)
    result["elapsed_sec"] = time.time() - t0
    (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8"); print("saved", RES / f"{args.name}.json")


if __name__ == "__main__":
    main()
