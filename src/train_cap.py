"""Task 16 (capacity / optimisation study): Function x Calendar FiLM residual forecaster with configurable encoder width/depth,
Transformer+FiLM variants, AdamW with early stopping, AdamW continuation, L-BFGS refinement and a tiny-set interpolation check.

Reference model = Task 14 `train_func.py --model film_func_x_cal` (k 72, H 24): x = [z hist (k), log B hist (k), future log B (H)] (168)
-> MLP encoder (168 -> 128 -> 64, GELU) -> FiLM from q = [e_c (8); cal summary (22); e_c (x) cal (176)] -> head([h, proj(future cal)]) + decay(h) * z_t.
Same data (MoEData/ResidualData), same splits, same Huber(delta 1) on z, same metrics (train_residual.metrics + MAE_N/E).

    python train_cap.py --name t16c_A1 --arch mlp:256,64 --epochs 24 --patience 5                 # Stage A / B (AdamW + early stopping)
    python train_cap.py --name t16c_A1_lbfgs --arch mlp:256,64 --mode lbfgs --init t16c_A1        # Stage C
    python train_cap.py --name t16c_A1_long --arch mlp:256,64 --mode long --init t16c_A1 --epochs 4    # control C2
    python train_cap.py --name t16c_tiny --mode tiny --arch mlp:512,512,256,128                   # Stage D
    python train_cap.py --name t16c_T1 --arch tfm:128,2,4,256,0.1 --epochs 24 --patience 5        # Stage E (d_model, layers, heads, d_ff, dropout)

Per-epoch history: train Huber (running mean over sampled batches), val Huber and val wMAPE on val windows at stride 48 (all nodes).
Early stopping on val wMAPE (patience) and on relative train-loss improvement < 1e-4 for 3 consecutive epochs. Best checkpoint = lowest
val wMAPE -> results/base/ckpt/<name>.pt. Final metrics (val/test at stride 6) only for the best checkpoint. 'train_loss_eval' =
Huber on a fixed deterministic training subset (every 18th window x 2000 fixed nodes, eval mode) for every run, so AdamW / AdamW-long /
L-BFGS are compared on the same quantity. Results -> results/base/results/<name>.json.
"""
import argparse, gc, json, math, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_moe import MoEData, RES   # noqa: E402
from train_residual import metrics   # noqa: E402

import os as _os
CKPT = Path(_os.environ["LP_CKPT"]) if _os.environ.get("LP_CKPT") else RES.parent / "ckpt"


def build(arch, k, H, K, d_f=8, n_cal=11, persist=True, step_cal=False, future_b=True, cond="func_x_cal"):
    import torch, torch.nn as nn
    n_in = 2 * k + H; n_cs = 2 * n_cal; n_q = {"none": 0, "func": d_f, "cal": n_cs, "func_cal": d_f + n_cs, "func_x_cal": d_f * n_cs + d_f + n_cs}[cond]
    kind, spec = arch.split(":")

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.kind = kind
            if kind == "lin":                                                  # fully linear reference: one affine map, no FiLM, head or anchor
                self.lin = nn.Linear(n_in + H * n_cal, H); self.hidden = H; self.g = None; self.decay = None; self.rel = nn.ModuleList(); self.ops = []
                return
            if kind == "mlp":
                widths = [int(w) for w in spec.split(",")]; layers, d = [], n_in
                for i, w in enumerate(widths):
                    layers.append(nn.Linear(d, w)); d = w
                    if i < len(widths) - 1:
                        layers.append(nn.GELU())
                self.enc = nn.Sequential(*layers); hidden = widths[-1]
            elif kind == "ptst":
                dm, nl, nh, dff, dr, pl, st = spec.split(","); dm, nl, nh, dff, dr, pl, st = int(dm), int(nl), int(nh), int(dff), float(dr), int(pl), int(st); hidden = dm
                self.pl, self.st, self.step_cal = pl, st, step_cal; self.n_tok = (k - pl) // st + 1; ch = 2 + (n_cal if step_cal else 0)
                self.proj = nn.Linear(pl * ch, dm); self.pos = nn.Parameter(torch.zeros(1, self.n_tok, dm)); nn.init.normal_(self.pos, std=0.02)
                self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dff, dropout=dr, batch_first=True, norm_first=True, activation="gelu"), nl)
                self.norm = nn.LayerNorm(dm); self.flat = nn.Linear(self.n_tok * dm, dm); self.bf_proj = None
            else:
                dm, nl, nh, dff, dr = spec.split(","); dm, nl, nh, dff, dr = int(dm), int(nl), int(nh), int(dff), float(dr); hidden = dm
                self.proj = nn.Linear(2 + (n_cal if step_cal else 0), dm); self.step_cal = step_cal; self.pos = nn.Parameter(torch.zeros(1, k, dm)); nn.init.normal_(self.pos, std=0.02)
                self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, dff, dropout=dr, batch_first=True, norm_first=True, activation="gelu"), nl)
                self.norm = nn.LayerNorm(dm); self.bf_proj = nn.Linear(H, dm) if future_b else None
            self.hidden = hidden; self.rel = nn.ModuleList(); self.ops = []
            self.cond = cond; self.e_c = nn.Embedding(K, d_f); self.g = nn.Sequential(nn.Linear(n_q, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden)) if n_q else None
            self.tf_proj = nn.Linear(H * n_cal, hidden)
            self.head = nn.Sequential(nn.Linear((3 if (kind == "tfm" and future_b) else 2) * hidden, hidden), nn.GELU(), nn.Linear(hidden, H)); self.decay = nn.Sequential(nn.Linear(hidden, H), nn.Sigmoid()) if persist else None

        def forward(self, xs, bf, cs, tff, c, th=None):
            """xs [B,N,k,2] history (z, log B); bf [B,N,H] future log B; cs [B,22] calendar summary; tff [B,H*11] future calendar; c [N] cluster; th [B,k,11] per-step calendar (step_cal)"""
            B, N, kk, _ = xs.shape
            if self.kind == "lin":
                return self.lin(torch.cat([xs.reshape(B, N, -1), bf, tff[:, None, :].expand(-1, N, -1)], -1))
            if self.kind == "mlp":
                h = self.enc(torch.cat([xs.reshape(B, N, -1), bf], -1))
            elif self.kind == "ptst":
                tok = xs if not self.step_cal else torch.cat([xs, th[:, None, :, :].expand(B, N, kk, -1)], -1)
                pt = tok.reshape(B * N, kk, -1).unfold(1, self.pl, self.st)                      # [BN, n_tok, ch, pl]
                z = self.norm(self.enc(self.proj(pt.reshape(B * N, self.n_tok, -1)) + self.pos)); h = self.flat(z.reshape(B * N, -1)).view(B, N, -1)
            else:
                tok = xs if not getattr(self, "step_cal", False) else torch.cat([xs, th[:, None, :, :].expand(B, N, kk, -1)], -1)
                h = self.norm(self.enc(self.proj(tok).reshape(B * N, kk, -1) + self.pos)[:, -1]).view(B, N, -1)
            if self.g is not None:
                ec = self.e_c(c)[None].expand(B, -1, -1); cc = cs[:, None, :].expand(-1, N, -1)
                q = {"func": lambda: ec, "cal": lambda: cc, "func_cal": lambda: torch.cat([ec, cc], -1),
                     "func_x_cal": lambda: torch.cat([ec, cc, (ec[..., :, None] * cc[..., None, :]).reshape(B, N, -1)], -1)}[self.cond]()
                gb = self.g(q); h = (1 + gb[..., :self.hidden]) * h + gb[..., self.hidden:]
            for P, br in zip(self.ops, self.rel):                                   # relational branches (full node set required)
                m = torch.stack([torch.sparse.mm(P, h[b]) for b in range(B)])
                mt = br["msg"](m); h = h + torch.sigmoid(br["gate"](torch.cat([h, mt], -1))) * mt
            t = self.tf_proj(tff)[:, None, :].expand(-1, N, -1)
            parts = [h, t] + ([self.bf_proj(bf)] if (self.kind == "tfm" and self.bf_proj is not None) else [])
            out = self.head(torch.cat(parts, -1))
            return out if self.decay is None else out + self.decay(h) * xs[:, :, -1, 0][..., None]
    net = Net()

    def add_relations(ops):
        import torch.nn as nn
        net.ops = ops
        for _ in ops:
            net.rel.append(nn.ModuleDict({"msg": nn.Sequential(nn.Linear(net.hidden, net.hidden), nn.GELU(), nn.Linear(net.hidden, net.hidden)), "gate": nn.Linear(2 * net.hidden, net.hidden)}))
    net.add_relations = add_relations
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--event_def", default="q95", help="q95 | zev | q95c (clustered or persistent) | q95s (clustered and persistent)"); ap.add_argument("--arch", default="mlp:128,64"); ap.add_argument("--mode", default="adamw", choices=["adamw", "long", "lbfgs", "tiny", "eval"])
    ap.add_argument("--k", type=int, default=72); ap.add_argument("--H", type=int, default=24); ap.add_argument("--d_f", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--patience", type=int, default=5); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--node_sample", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval_stride", type=int, default=6)
    ap.add_argument("--init", default=None, help="checkpoint name (results/base/ckpt/<init>.pt) for --mode long / lbfgs")
    ap.add_argument("--sub_stride", type=int, default=18); ap.add_argument("--sub_nodes", type=int, default=2000, help="fixed training subset for train_loss_eval and L-BFGS")
    ap.add_argument("--outer", type=int, default=5); ap.add_argument("--max_iter", type=int, default=20); ap.add_argument("--max_eval", type=int, default=25); ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--tiny_windows", type=int, default=256); ap.add_argument("--tiny_nodes", type=int, default=64); ap.add_argument("--tiny_steps", type=int, default=3000); ap.add_argument("--tiny_outer", type=int, default=60)
    ap.add_argument("--no_final_eval", action="store_true"); ap.add_argument("--dump_pred", action="store_true", help="save test forecasts (people) to <name>_pred_test.npz"); ap.add_argument("--loss_mask", default="none", choices=["none", "normal", "extreme"], help="MoE expert training: weight targets by 1-e (normal) or e (extreme, q95 events)"); ap.add_argument("--max_steps", type=int, default=None); ap.add_argument("--no_future_b", action="store_true", help="Transformer: drop the future log B branch from the head (earlier backbone-study convention)"); ap.add_argument("--step_cal", action="store_true", help="Transformer: append the per-step calendar (11) to every history token, as in the earlier backbone study"); ap.add_argument("--rel", default="", help="relational branches on the final encoder (comma list of prop_matrix names: geo, func_knn, mob_lag_local, trans); forces full-node batches"); ap.add_argument("--cond", default="func_x_cal", choices=["none", "func", "cal", "func_cal", "func_x_cal"], help="FiLM conditioning source"); ap.add_argument("--no_persist", action="store_true", help="drop the persistence anchor decay(h) * z_t from the head (Transformer ablation)"); ap.add_argument("--node_chunk", type=int, default=500, help="Transformer: nodes per forward/backward chunk"); ap.add_argument("--no_flat_stop", action="store_true", help="disable the train-loss-plateau stop (long runs)"); ap.add_argument("--anneal", type=int, default=3, help="epochs of lr annealing from the best checkpoint after an early stop under OneCycle")
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time(); CKPT.mkdir(exist_ok=True)
    md = MoEData(args.k, args.H, args.event_def); data = md.d; H, N, k = args.H, data.N, args.k
    cl = torch.tensor(data.cluster.astype(np.int64), device=dev); K = int(data.cluster.max()) + 1
    model = build(args.arch, k, H, K, args.d_f, persist=not args.no_persist, step_cal=args.step_cal, future_b=not args.no_future_b, cond=args.cond)
    if args.rel:
        from train_moe import prop_matrix
        from models import sparse_to_torch
        model.add_relations([sparse_to_torch(prop_matrix(data, n_)).to(dev) for n_ in args.rel.split(",")]); args.node_sample = N
        print("relational branches:", args.rel, "(full-node batches)", flush=True)
    model = model.to(dev); n_par = sum(p.numel() for p in model.parameters())
    print("data:", json.dumps(data.summary()), "K", K, "arch", args.arch, "params", n_par, flush=True)
    tr = data.windows("train", 1); tsv48 = data.windows("val", 48); ts_eval = {s: data.windows(s, args.eval_stride) for s in ("val", "test")}
    huber = torch.nn.HuberLoss(delta=1.0); T = lambda v: torch.tensor(v, device=dev)

    def prep(split, ts, sel=None):
        xh, th, tfu, bf, y, ye, xg, dtf, _ = md.batch(split, ts); B = xh.shape[0]
        if sel is not None:
            xh, bf, y, ye = xh[:, :, sel], bf[:, :, sel], y[:, :, sel], ye[:, :, sel]
        xs = xh.transpose(0, 2, 1, 3).astype(np.float32); bfn = bf.transpose(0, 2, 1).astype(np.float32); cs = np.concatenate([th[:, -1], tfu.mean(1)], -1).astype(np.float32)
        return xs, bfn, cs, tfu.reshape(B, -1).astype(np.float32), y.transpose(0, 2, 1).astype(np.float32), dtf, ye, ye.transpose(0, 2, 1).astype(np.float32), th[:, :, :11].astype(np.float32)

    NCH = args.node_chunk if args.arch.startswith(("tfm", "ptst")) else 10 ** 9      # Transformer: chunk the node axis (memory); MLP: one shot

    def fwd(t, sel=None):
        """forward over node chunks (eval / no-grad use); returns (pred [B,N,H], y)"""
        xs, bfn, cs, tff, y = t[:5]; c = cl if sel is None else cl[sel]; outs = []; th_ = T(t[8]) if args.step_cal else None
        for n0 in range(0, xs.shape[1], NCH):
            outs.append(model(T(xs[:, n0:n0 + NCH]), T(bfn[:, n0:n0 + NCH]), T(cs), T(tff), c[n0:n0 + NCH], th_))
        return torch.cat(outs, 1), T(y)

    def wmask(t):
        """target weights for masked expert training: 1-e (normal), e (extreme), None (plain)"""
        return None if args.loss_mask == "none" else (t[7] if args.loss_mask == "extreme" else 1.0 - t[7])

    def wloss(out, yy, w):
        if w is None:
            return huber(out, yy)
        return (w * torch.nn.functional.huber_loss(out, yy, reduction="none", delta=1.0)).sum() / w.sum().clamp(min=1)

    def train_step_loss(t, sel):
        """loss.backward() over node chunks with gradient accumulation; returns the (chunk-weighted) loss value"""
        xs, bfn, cs, tff, y = t[:5]; c = cl[sel]; tot = 0.0; N_ = xs.shape[1]; wm = wmask(t); th_ = T(t[8]) if args.step_cal else None
        for n0 in range(0, N_, NCH):
            n1 = min(N_, n0 + NCH); w = (n1 - n0) / N_
            loss = wloss(model(T(xs[:, n0:n1]), T(bfn[:, n0:n1]), T(cs), T(tff), c[n0:n1], th_), T(y[:, n0:n1]), None if wm is None else T(wm[:, n0:n1])) * w; loss.backward(); tot += loss.item()
        return tot

    # fixed deterministic training subset (train_loss_eval for every mode; L-BFGS objective)
    rng = np.random.RandomState(0); sub_nodes = (np.arange(N) if args.rel else np.sort(rng.choice(N, args.sub_nodes, replace=False))); sub_ts = tr[::args.sub_stride]
    SUB = None
    def load_sub():
        nonlocal SUB
        if SUB is None:
            parts = [prep("train", sub_ts[i:i + 8], sub_nodes) for i in range(0, len(sub_ts), 8)]
            SUB = [np.concatenate([p[j] for p in parts]) for j in range(5)] + [np.concatenate([p[7] for p in parts]), np.concatenate([p[8] for p in parts])]; print(f"fixed subset: {len(sub_ts)} windows x {len(sub_nodes)} nodes", flush=True)
        return SUB

    def sub_loss(grad=False):
        """Huber over the fixed subset, chunked over 8 windows; accumulates gradients when grad=True (full-subset gradient for L-BFGS)"""
        xs, bfn, cs, tff, y, yem, thh = load_sub(); W = xs.shape[0]; tot = 0.0
        for i in range(0, W, 8):
            t = (xs[i:i + 8], bfn[i:i + 8], cs[i:i + 8], tff[i:i + 8], y[i:i + 8], None, None, yem[i:i + 8], thh[i:i + 8]); w = t[0].shape[0] / W; wm = wmask(t)
            if grad:
                out, yy = fwd(t, sub_nodes); loss = wloss(out, yy, None if wm is None else T(wm)) * w; loss.backward(); tot += loss.item()
            else:
                with torch.no_grad():
                    out, yy = fwd(t, sub_nodes); tot += wloss(out, yy, None if wm is None else T(wm)).item() * w
        return tot

    def val_quick():
        """val Huber + val wMAPE at stride 48, all nodes (per-epoch monitor)"""
        model.eval(); vl, preds, mn, md_ = [], [], 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(tsv48), 8):
                t = prep("val", tsv48[i:i + 8]); out, yy = fwd(t); vl.append(huber(out, yy).item()); preds.append(out.permute(0, 2, 1).cpu().numpy())
                wm = wmask(t)
                if wm is not None:
                    mn += float((T(wm) * (out - yy).abs()).sum().item()); md_ += float(wm.sum())
        pz = np.concatenate(preds); Xh, _, X, _, _ = data.people("val", tsv48, pz); w = float(np.abs(Xh - X).sum() / np.abs(X).sum()); del Xh, X, pz; gc.collect()
        return float(np.mean(vl)), (w if args.loss_mask == "none" else mn / max(md_, 1)), w

    def snapshot():
        return {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    hist = []; best = {"val_wmape": np.inf, "epoch": -1, "state": snapshot()}
    result = {"name": args.name, "args": vars(args), "arch": args.arch, "params": n_par, "mode": args.mode}
    if args.init:
        sd = torch.load(CKPT / f"{args.init}.pt", map_location=dev); model.load_state_dict(sd["state"]); result["init"] = args.init; result["init_meta"] = {k_: v for k_, v in sd.items() if k_ != "state"}
        vl, vw, vx = val_quick(); best = {"val_wmape": vw, "epoch": 0, "state": snapshot()}; hist.append({"epoch": 0, "stage": "init", "train_loss_eval": sub_loss(), "val_loss": vl, "val_wmape": vx, "monitor": vw, "min": 0.0}); print(json.dumps(hist[-1]), flush=True)

    def run_epoch(opt, sched):
        model.train(); perm = np.random.permutation(tr); losses = []
        for step, i in enumerate(range(0, len(perm), args.batch)):
            if args.max_steps and step >= args.max_steps:
                break
            sel = np.random.choice(N, args.node_sample, replace=False); opt.zero_grad(); lv = train_step_loss(prep("train", perm[i:i + args.batch], sel), sel)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(lv)
        return float(np.mean(losses)), sched.get_last_lr()[0]

    def adamw_epochs(n_ep, lr, sched_kind, patience):
        """AdamW loop with early stopping; on early stop under OneCycle the best checkpoint is annealed (lr -> 0 over args.anneal epochs)
        so that early-stopped runs are not penalised for missing the schedule's annealing phase. Returns the number of epochs run."""
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd); steps = len(tr) // args.batch + 1; total = n_ep * steps + 1
        sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total) if sched_kind == "onecycle" else torch.optim.lr_scheduler.CosineAnnealingLR(opt, total))
        bad, flat, prev, stopped = 0, 0, None, None; e0 = len([h for h in hist if h.get("stage") != "init"]); n_run = 0
        for ep in range(n_ep):
            trl, lr_now = run_epoch(opt, sched); vl, vw, vx = val_quick(); n_run += 1
            hist.append({"epoch": e0 + ep + 1, "stage": args.mode, "train_loss": trl, "val_loss": vl, "val_wmape": vx, "monitor": vw, "lr": lr_now, "min": round((time.time() - t0) / 60, 1)}); print(json.dumps(hist[-1]), flush=True)
            if vw < best["val_wmape"] - 1e-6:
                best.update(val_wmape=vw, epoch=e0 + ep + 1, state=snapshot()); bad = 0
            else:
                bad += 1
            flat = flat + 1 if (prev is not None and (prev - trl) / max(prev, 1e-12) < 1e-4) else 0; prev = trl
            if bad >= patience:
                stopped = f"val wMAPE patience {patience}"; break
            if flat >= 3 and not args.no_flat_stop:
                stopped = "train loss flat (<1e-4 rel. improvement, 3 epochs)"; break
        result["stop_reason"] = stopped or "max epochs"
        if stopped and sched_kind == "onecycle" and args.anneal > 0 and lr_now > 0.05 * lr:      # anneal from the best checkpoint
            model.load_state_dict(best["state"]); opt = torch.optim.AdamW(model.parameters(), lr=lr_now, weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=args.anneal * steps)
            for ep in range(args.anneal):
                trl, lr_a = run_epoch(opt, sched); vl, vw, vx = val_quick(); n_run += 1
                hist.append({"epoch": e0 + n_run, "stage": "anneal", "train_loss": trl, "val_loss": vl, "val_wmape": vx, "monitor": vw, "lr": lr_a, "min": round((time.time() - t0) / 60, 1)}); print(json.dumps(hist[-1]), flush=True)
                if vw < best["val_wmape"] - 1e-6:
                    best.update(val_wmape=vw, epoch=e0 + n_run, state=snapshot())
            result["stop_reason"] += f" + anneal {args.anneal} ep from best (lr {lr_now:.1e} -> 0)"
        return n_run

    if args.mode == "eval":
        result["epochs_run"] = 0; result["stop_reason"] = "eval only"
    elif args.mode == "adamw":
        result["epochs_run"] = adamw_epochs(args.epochs, args.lr, "onecycle", args.patience)
    elif args.mode == "long":                       # control C2: continue from checkpoint, cosine from lr to 0 over the given epochs
        result["epochs_run"] = adamw_epochs(args.epochs, args.lr, "cosine", args.patience)
    elif args.mode == "lbfgs":
        load_sub(); model.train(); bad, prev = 0, hist[-1]["train_loss_eval"]; lb = []
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.max_iter, max_eval=args.max_eval, history_size=args.history, line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-7)
        def closure():
            opt.zero_grad(); return torch.tensor(sub_loss(grad=True))
        for it in range(args.outer):
            tb = time.time(); loss_before = sub_loss(); opt.step(closure); loss_after = sub_loss(); vl, vw, vx = val_quick()
            lb.append({"outer": it + 1, "loss_before": loss_before, "loss_after": loss_after, "val_loss": vl, "val_wmape": vx, "monitor": vw, "min": round((time.time() - tb) / 60, 2)})
            hist.append({"epoch": it + 1, "stage": "lbfgs", "train_loss_eval": loss_after, "val_loss": vl, "val_wmape": vx, "monitor": vw, "min": round((time.time() - t0) / 60, 1)}); print(json.dumps(hist[-1]), flush=True)
            if vw < best["val_wmape"] - 1e-6:
                best.update(val_wmape=vw, epoch=it + 1, state=snapshot()); bad = 0
            else:
                bad += 1
            if (prev - loss_after) / max(prev, 1e-12) < 1e-4:
                result["stop_reason"] = "subset loss rel. improvement < 1e-4"; break
            if bad >= 2:
                result["stop_reason"] = "val wMAPE no improvement for 2 outer iterations"; break
            prev = loss_after
        else:
            result["stop_reason"] = f"{args.outer} outer iterations"
        result["lbfgs"] = lb; result["epochs_run"] = len(lb)
    elif args.mode == "tiny":                        # Stage D: interpolate a fixed tiny subset (no node resampling, fixed ordering, wd 0)
        rng = np.random.RandomState(0); tw = tr[:: max(1, len(tr) // args.tiny_windows)][:args.tiny_windows]; tn = np.sort(rng.choice(N, args.tiny_nodes, replace=False))
        parts = [prep("train", tw[i:i + 8], tn) for i in range(0, len(tw), 8)]; xs, bfn, cs, tff, y = [np.concatenate([p[j] for p in parts]) for j in range(5)]
        Xs, Bf, Cs, Tf, Y = T(xs), T(bfn), T(cs), T(tff), T(y); ct = cl[tn]; print(f"tiny set: {len(tw)} windows x {len(tn)} nodes = {len(tw) * len(tn)} samples x {H} targets; params {n_par}", flush=True)
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.tiny_steps); curve = []
        for step in range(args.tiny_steps):
            loss = huber(model(Xs, Bf, Cs, Tf, ct), Y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 100 == 0 or step == args.tiny_steps - 1:
                curve.append({"stage": "adamw", "step": step, "loss": loss.item(), "min": round((time.time() - t0) / 60, 1)}); print(json.dumps(curve[-1]), flush=True)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.max_iter, max_eval=args.max_eval, history_size=args.history, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); loss = huber(model(Xs, Bf, Cs, Tf, ct), Y); loss.backward(); return loss
        for it in range(args.tiny_outer):
            opt.step(closure); l = huber(model(Xs, Bf, Cs, Tf, ct), Y).item(); curve.append({"stage": "lbfgs", "step": it + 1, "loss": l, "min": round((time.time() - t0) / 60, 1)})
            if it % 5 == 0 or l <= 1e-5:
                print(json.dumps(curve[-1]), flush=True)
            if l <= 1e-5:
                break
        result.update(tiny={"windows": int(len(tw)), "nodes": int(len(tn)), "samples": int(len(tw) * len(tn)), "targets": int(len(tw) * len(tn) * H)}, curve=curve, final_loss=curve[-1]["loss"], reached_target=bool(curve[-1]["loss"] <= 1e-5), minutes=round((time.time() - t0) / 60, 1))
        (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8"); print("saved", RES / f"{args.name}.json", "final loss", curve[-1]["loss"]); return

    # ---- best checkpoint: save, fixed-subset train loss, full val/test metrics ----
    model.load_state_dict(best["state"]); model.eval(); result.update(history=hist, best_epoch=best["epoch"], best_val_wmape_quick=best["val_wmape"], train_minutes=round((time.time() - t0) / 60, 1))
    torch.save({"state": best["state"], "arch": args.arch, "persist": not args.no_persist, "step_cal": args.step_cal, "future_b": not args.no_future_b, "cond": args.cond, "name": args.name, "best_epoch": best["epoch"], "lr": args.lr, "wd": args.wd, "seed": args.seed}, CKPT / f"{args.name}.pt")
    result["train_loss_eval"] = sub_loss(); result["models"] = {"model": {}}
    if not args.no_final_eval:
        for s in ("val", "test"):
            ts = ts_eval[s]; preds, DT, YE, hl, hm = [], [], [], [], []
            with torch.no_grad():
                for i in range(0, len(ts), 8):
                    t = prep(s, ts[i:i + 8]); out, yy = fwd(t); hl.append(huber(out, yy).item() * out.shape[0]); wm = wmask(t)
                    if wm is not None:
                        hm.append(wloss(out, yy, T(wm)).item() * out.shape[0])
                    preds.append(out.permute(0, 2, 1).cpu().numpy().astype(np.float16)); DT.append(t[5]); YE.append(t[6])
            pz = np.concatenate(preds).astype(np.float32); del preds; DT = np.concatenate(DT); YE = np.concatenate(YE)
            Xh, Rh, X, R, Bp = data.people(s, ts, pz)
            if args.dump_pred and s == "test":                                     # T5: forecasts in people for downstream (pressure) analysis
                np.savez(RES / f"{args.name}_pred_test.npz", Xhat=Xh.astype(np.float32), X=X.astype(np.float32), B=Bp.astype(np.float32), E=(YE > 0), cells=data.cells, time=np.stack([data.ti[s].datetime.values[t_ + 1:t_ + 1 + H] for t_ in ts]).astype("datetime64[s]").astype(str))
            er = np.abs(Rh - R); e = YE > 0; reg = {"MAE_N": float(er[~e].mean()), "MAE_E": float(er[e].mean()), "huber_loss": float(np.sum(hl) / len(ts)), "huber_masked": (float(np.sum(hm) / len(ts)) if hm else None)}
            del Xh, Rh, X, R, Bp, er, e; gc.collect()
            m = metrics(data, s, ts, pz, DT); m.update(reg); result["models"]["model"][s] = m; del pz; gc.collect()
            print(s, json.dumps({k_: round(m[k_], 4) for k_ in ("X_wmape", "huber_loss", "R_mae", "R_rel_err_top1pct", "extreme_recall_top1", "extreme_f1_top1", "peak_magnitude_rel_err", "top5pct_spatial_jaccard")}), flush=True)
    result["minutes"] = round((time.time() - t0) / 60, 1)
    (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8"); print("saved", RES / f"{args.name}.json")


if __name__ == "__main__":
    main()
