"""Task 14-A/B: function-conditioned residual forecasting (functional clusters as regime conditioning, not as edges).

    python train_func.py --name tf_k24_global      --model global        --k 24
    python train_func.py --name tf_k24_func        --model func          # low-rank cluster-conditioned linear W_0 + dW_c
    python train_func.py --name tf_k24_cal         --model cal           # W_0 + dW_cal
    python train_func.py --name tf_k24_func_cal    --model func_cal      # W_0 + dW_c + dW_cal
    python train_func.py --name tf_k24_func_x_cal  --model func_x_cal    # + dW_{c,cal} interaction
    python train_func.py --name tf_k24_film_none   --model film_none     # nonlinear temporal encoder, no conditioning
    python train_func.py --name tf_k24_film_func   --model film_func     # FiLM from cluster embedding
    python train_func.py --name tf_k24_film_cal    --model film_cal      # FiLM from calendar
    python train_func.py --name tf_k24_film_func_cal   --model film_func_cal      # FiLM from [e_c; cal]
    python train_func.py --name tf_k24_film_func_x_cal --model film_func_x_cal    # FiLM from e_c (x) cal outer product
    python train_func.py --name tf_k24_fmoe        --model fmoe --experts 4        # functional-cluster MoE, softmax gate on (e_c, cal, event state)

Inputs per cell: history [z, log-baseline] (k hours) + future log-baseline (H); calendar of the current hour and of the targets;
functional cluster c_i (K=6, Task 2 normalized weekly profiles). No facility/static features, no relational branch: function enters only
through the conditioning path. Loss: Huber on z (all targets). Metrics as train_moe (incl. MAE_N / MAE_E on q95 events).
Results -> results/base/results/tf_<name>.json (+ expert-weight analysis for fmoe).
"""
import argparse, gc, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_moe import MoEData, RES   # noqa: E402
from train_residual import metrics   # noqa: E402


def build_model(kind, k, H, K, r=16, hidden=64, d_f=8, n_cal=11, experts=4):
    import torch, torch.nn as nn
    n_in = 2 * k + H; n_cs = 2 * n_cal            # calendar summary: current hour + mean of target hours

    def emb(layer, c, B):
        return layer(c) if c.dim() == 2 else layer(c)[None].expand(B, -1, -1)

    class LowRankTerm(nn.Module):
        """dW(e) x = U (e * (V x)) : a rank-r weight perturbation modulated by a code e in R^r"""
        def __init__(self):
            super().__init__(); self.V = nn.Linear(n_in, r, bias=False); self.U = nn.Linear(r, H, bias=False)
        def forward(self, x, e):
            return self.U(e * self.V(x))

    class FuncLinear(nn.Module):
        def __init__(self, use_func, use_cal, use_int):
            super().__init__()
            self.W0 = nn.Linear(n_in, H); self.cal_fut = nn.Linear(H * n_cal, H); self.decay = nn.Parameter(torch.zeros(H))
            self.use_func, self.use_cal, self.use_int = use_func, use_cal, use_int
            if use_func:
                self.e_c = nn.Embedding(K, r); self.b_c = nn.Embedding(K, H); self.t_func = LowRankTerm(); nn.init.zeros_(self.b_c.weight)
            if use_cal:
                self.e_cal = nn.Linear(n_cs, r); self.t_cal = LowRankTerm()
            if use_int:
                self.t_int = LowRankTerm()
        def forward(self, x, last, cs, tfu_flat, c):
            out = self.W0(x) + self.cal_fut(tfu_flat)[:, None, :]
            if self.use_func:
                ec = emb(self.e_c, c, x.shape[0]); out = out + self.t_func(x, ec) + emb(self.b_c, c, x.shape[0])
            if self.use_cal:
                ecal = self.e_cal(cs)[:, None, :].expand(-1, x.shape[1], -1); out = out + self.t_cal(x, ecal)
            if self.use_int:
                out = out + self.t_int(x, ec * ecal)
            return out + torch.sigmoid(self.decay)[None, None, :] * last[..., None]

    class FiLMNet(nn.Module):
        def __init__(self, cond):
            super().__init__(); self.cond = cond
            self.enc = nn.Sequential(nn.Linear(n_in, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
            self.e_c = nn.Embedding(K, d_f); self.tf_proj = nn.Linear(H * n_cal, hidden)
            n_q = {"none": 0, "func": d_f, "cal": n_cs, "func_cal": d_f + n_cs, "func_x_cal": d_f * n_cs + d_f + n_cs}[cond]
            self.g = nn.Sequential(nn.Linear(n_q, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden)) if n_q else None
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, H)); self.decay = nn.Sequential(nn.Linear(hidden, H), nn.Sigmoid())
        def forward(self, x, last, cs, tfu_flat, c):
            B, N, _ = x.shape; h = self.enc(x)
            if self.g is not None:
                ec = emb(self.e_c, c, B); cc = cs[:, None, :].expand(-1, N, -1)
                q = {"func": lambda: ec, "cal": lambda: cc, "func_cal": lambda: torch.cat([ec, cc], -1),
                     "func_x_cal": lambda: torch.cat([ec, cc, (ec[..., :, None] * cc[..., None, :]).reshape(B, N, -1)], -1)}[self.cond]()
                gb = self.g(q); gamma, beta = gb[..., :hidden], gb[..., hidden:]; h = (1 + gamma) * h + beta
            t = self.tf_proj(tfu_flat)[:, None, :].expand(-1, N, -1)
            return self.head(torch.cat([h, t], -1)) + self.decay(h) * last[..., None]

    class FuncMoE(nn.Module):
        """M linear experts on the history; softmax gate on (cluster embedding, calendar summary, event state)"""
        def __init__(self):
            super().__init__(); self.experts = nn.ModuleList([FuncLinear(False, False, False) for _ in range(experts)])
            self.e_c = nn.Embedding(K, d_f); self.gate = nn.Sequential(nn.Linear(d_f + n_cs + 3, hidden), nn.GELU(), nn.Linear(hidden, experts))
        def forward(self, x, last, cs, tfu_flat, c, ev):
            B, N, _ = x.shape; ec = emb(self.e_c, c, B); cc = cs[:, None, :].expand(-1, N, -1)
            pi = torch.softmax(self.gate(torch.cat([ec, cc, ev], -1)), -1)                       # [B,N,M]
            outs = torch.stack([e(x, last, cs, tfu_flat, c) for e in self.experts], -1)          # [B,N,H,M]
            return (outs * pi[:, :, None, :]).sum(-1), pi

    if kind == "global":
        return FuncLinear(False, False, False)
    if kind == "func":
        return FuncLinear(True, False, False)
    if kind == "cal":
        return FuncLinear(False, True, False)
    if kind == "func_cal":
        return FuncLinear(True, True, False)
    if kind == "func_x_cal":
        return FuncLinear(True, True, True)
    if kind.startswith("film_"):
        return FiLMNet(kind[5:])
    if kind == "fmoe":
        return FuncMoE()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--model", required=True); ap.add_argument("--k", type=int, default=24); ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--d_f", type=int, default=8); ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--node_sample", type=int, default=2000); ap.add_argument("--eval_stride", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None); ap.add_argument("--event_def", default="q95"); ap.add_argument("--cluster_col", default="c_kmeans_norm")
    ap.add_argument("--dyn_state", action="store_true", help="Task 14-C: condition on the daily functional state of the previous complete day (eda/functional_state.py) instead of the static label")
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    md = MoEData(args.k, args.H, args.event_def); data = md.d
    if args.cluster_col != "c_kmeans_norm":                      # alternative functional labelling (e.g. c_weekday, c_temporal_demographic)
        import pandas as pd; from residual_data import EDA
        data.cluster = pd.read_parquet(EDA / "cluster_labels.parquet")[args.cluster_col].values.astype(int)
    cl = torch.tensor(data.cluster.astype(np.int64), device=dev); K = int(data.cluster.max()) + 1
    DYN = None
    if args.dyn_state:                                                   # state of the previous complete day (causal); -1 -> static label
        z = np.load(RES.parents[1] / "eda" / "func_state" / "state_daily.npz"); day_pos = {d: i for i, d in enumerate(z["days"])}; st_ = z["state"]
        DYN = {}
        for s_ in ("train", "val", "test"):
            ymd = data.ti[s_].ymd.astype(str).values; di = np.array([day_pos.get(d, -1) for d in ymd]) - 1
            S = np.where(di[:, None] >= 0, st_[np.maximum(di, 0)], -1); S = np.where(S < 0, data.cluster[None], S); DYN[s_] = S.astype(np.int64)   # [T,N]
        print("dynamic state: share differing from static", {s_: float((DYN[s_] != data.cluster[None]).mean()) for s_ in DYN}, flush=True)
    model = build_model(args.model, args.k, args.H, K, args.rank, args.hidden, args.d_f, 11, args.experts).to(dev)
    print("data:", json.dumps(data.summary()), "K", K, "params", sum(p.numel() for p in model.parameters()), flush=True)
    tr = data.windows("train", 1); ts_eval = {s: data.windows(s, args.eval_stride) for s in ("val", "test")}

    def prep(b, sel=None, split=None, ts=None):
        xh, th, tfu, bf, y, ye, xg, dtf, _ = b
        if sel is not None:
            xh, bf, y, xg = xh[:, :, sel], bf[:, :, sel], y[:, :, sel], xg[:, :, sel]
        B = xh.shape[0]
        x = np.concatenate([xh.transpose(0, 2, 1, 3).reshape(B, xh.shape[2], -1), bf.transpose(0, 2, 1)], -1)   # [B,N,2k+H]
        last = xh[:, -1, :, 0]; cs = np.concatenate([th[:, -1], tfu.mean(1)], -1); ev = xg[:, -1]                 # [B,N,3] event state at t
        T = lambda v: torch.tensor(v, device=dev)
        c = None
        if DYN is not None:
            c = DYN[split][np.asarray(ts)]; c = torch.tensor(c if sel is None else c[:, sel], device=dev)
        return T(x), T(last), T(cs), T(tfu.reshape(B, -1)), T(ev), y, c

    def fwd(t, sel=None):
        x, last, cs, tff, ev, y, c = t
        if c is None:
            c = cl if sel is None else cl[sel]
        if args.model == "fmoe":
            out, pi = model(x, last, cs, tff, c, ev); return out, y, pi
        return model(x, last, cs, tff, c), y, None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = (len(tr) // args.batch + 1) if not args.max_steps else args.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps + 1); best = (np.inf, None)
    huber = torch.nn.HuberLoss(delta=1.0)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(tr); losses = []
        for step, i in enumerate(range(0, len(perm), args.batch)):
            if args.max_steps and step >= args.max_steps:
                break
            sel = np.random.choice(data.N, args.node_sample, replace=False)
            out, y, _ = fwd(prep(md.batch("train", perm[i:i + args.batch]), sel, "train", perm[i:i + args.batch]), sel)
            loss = huber(out, torch.tensor(y.transpose(0, 2, 1), device=dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(loss.item())
        # validation loss (stride 48)
        model.eval(); vl = []
        with torch.no_grad():
            tsv = data.windows("val", 48)
            for i in range(0, len(tsv), 8):
                t = prep(md.batch("val", tsv[i:i + 8]), None, "val", tsv[i:i + 8]); out, y, _ = fwd(t); vl.append(huber(out, torch.tensor(y.transpose(0, 2, 1), device=dev)).item())
        print(json.dumps({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vl)), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        if np.mean(vl) < best[0]:
            best = (float(np.mean(vl)), {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval(); del opt, sched; gc.collect()

    result = {"name": args.name, "args": vars(args), "K": K, "models": {"model": {}}, "best_val_loss": best[0]}
    for s in ("val", "test"):
        ts = ts_eval[s]; preds, PI, DT, YE = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(ts), 8):
                b = md.batch(s, ts[i:i + 8]); out, y, pi = fwd(prep(b, None, s, ts[i:i + 8])); preds.append(out.permute(0, 2, 1).cpu().numpy().astype(np.float16)); DT.append(b[7]); YE.append(b[5])
                if pi is not None:
                    PI.append(pi.cpu().numpy().astype(np.float16))
        pz = np.concatenate(preds).astype(np.float32); del preds; DT = np.concatenate(DT); YE = np.concatenate(YE)
        # regime MAE first (people-scale arrays freed before metrics() to bound peak memory), then the full metric set
        Xh, Rh, X, R, B = data.people(s, ts, pz); er = np.abs(Rh - R); e = YE > 0; reg = {"MAE_N": float(er[~e].mean()), "MAE_E": float(er[e].mean())}
        del Xh, Rh, X, R, B, er, e; gc.collect()
        m = metrics(data, s, ts, pz, DT); m.update(reg); result["models"]["model"][s] = m; del pz; gc.collect()
        print(s, json.dumps({k_: round(m[k_], 4) for k_ in ("X_wmape", "R_mae", "MAE_N", "MAE_E", "R_rel_err_top1pct", "extreme_recall_top1", "extreme_f1_top1", "peak_magnitude_rel_err", "top5pct_spatial_jaccard")}), flush=True)
        if PI and s == "val":
            # expert-weight analysis: mean pi by hour, weekday/weekend, holiday, cluster, normal/extreme (target-hour regime uses the first target hour)
            P = np.concatenate(PI).astype(np.float32)                                              # [W,N,M]
            ti = data.ti[s]; t1 = np.asarray(ts) + 1; hour = ti.tt.values[t1]; dow = ti.datetime.dt.dayofweek.values[t1]; hol = data.tf[s][t1, 5] > 0
            ana = {"by_hour": {str(int(h)): P[hour == h].mean((0, 1)).tolist() for h in np.unique(hour)},
                   "weekday": P[dow < 5].mean((0, 1)).tolist(), "weekend": P[dow >= 5].mean((0, 1)).tolist(), "holiday": P[hol].mean((0, 1)).tolist(),
                   "by_cluster": {str(c): P[:, data.cluster == c].mean((0, 1)).tolist() for c in range(K)},
                   "normal": P[YE[:, 0] == 0].mean(0).tolist(), "extreme": P[YE[:, 0] > 0].mean(0).tolist(),
                   "entropy_mean": float((-(P * np.log(P + 1e-9)).sum(-1)).mean()), "argmax_share": np.bincount(P.reshape(-1, P.shape[-1]).argmax(1), minlength=P.shape[-1]).tolist()}
            ana["by_cluster_hour"] = {str(c): {str(int(h)): P[hour == h][:, data.cluster == c].mean((0, 1)).tolist() for h in (3, 9, 13, 18, 21)} for c in range(K)}
            result["expert_weights"] = ana; print("expert weights by cluster:", json.dumps({c: [round(v, 3) for v in w] for c, w in ana["by_cluster"].items()}), flush=True)
    result["minutes"] = round((time.time() - t0) / 60, 1)
    (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8"); print("saved", RES / f"{args.name}.json")


if __name__ == "__main__":
    main()
