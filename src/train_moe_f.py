"""Task 16 Stage F: binary Normal/Extreme MoE assembled from frozen experts (reference Task 13/14 experts or Task 16 checkpoints) and a
frozen / refitted binary gate.  No graph, mobility or facility context.

    python train_moe_f.py --name moef_F0                                                       # reference experts + reference gate
    python train_moe_f.py --name moef_F1a --normal ckpt:t16c_B_A5_lr0.001_wd0.0001            # Task 16 single model as normal expert
    python train_moe_f.py --name moef_F2  --normal ckpt:t16c_N_A5 --extreme ckpt:t16c_E1      # optimised masked experts
    python train_moe_f.py --name moef_F4  --normal ckpt:... --extreme ckpt:... --gate refit_bce   # gate retrained (wBCE, same inputs) with frozen experts
    python train_moe_f.py --name moef_F4m --normal ckpt:... --extreme ckpt:... --gate refit_mix   # gate fine-tuned on the mixture Huber with frozen experts

Experts: 'ref' = cached Task 13/14 ResidualForecaster experts (N_mlp_masked_k72_q95_film / E_mlp_none_k72_q95_film, FiLM-MLP normal, MLP extreme);
'ckpt:<name>' = train_cap.py checkpoint (results/base/ckpt/<name>.pt).  Gate: 'ref' = cached G_wbce_k72_q95 (linear backbone, wBCE);
'refit_bce' = same architecture / inputs / loss retrained from scratch (<= --gate_epochs, best val PR-AUC); 'refit_mix' = initialised from the
reference gate and trained on Huber((1-p) z_N + p z_E, y) with the experts frozen (best val mixture MAE).
Mixtures evaluated as in train_moe: normal_only, extreme_only, oracle, soft (raw p), soft_platt (Platt on validation), hard tau (val-selected).
Results -> results/base/results/<name>.json (same layout as train_moe results).
"""
import argparse, gc, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_moe import MoEData, RES, CACHE, build_gate, ece_score   # noqa: E402
from train_residual import metrics                                  # noqa: E402
from train_cap import build as build_cap, CKPT                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--event_def", default="q95", help="q95 | zev | q95c (clustered or persistent) | q95s (clustered and persistent)"); ap.add_argument("--normal", default="ref"); ap.add_argument("--extreme", default="ref"); ap.add_argument("--gate", default="ref")
    ap.add_argument("--blend_set", default="all", choices=["all", "const"], help="const: blend candidates restricted to constant weights w")
    ap.add_argument("--k", type=int, default=72); ap.add_argument("--H", type=int, default=24); ap.add_argument("--eval_stride", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_epochs", type=int, default=12); ap.add_argument("--gate_patience", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--node_sample", type=int, default=2000)
    args = ap.parse_args()
    import torch
    from residual_models import ResidualForecaster
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    md = MoEData(args.k, args.H, args.event_def); data = md.d; N, H, k = data.N, args.H, args.k; static = torch.tensor(data.static, device=dev)
    cl = torch.tensor(data.cluster.astype(np.int64), device=dev); K_cl = int(data.cluster.max()) + 1
    tr = data.windows("train", 1); ts_eval = {s: data.windows(s, args.eval_stride) for s in ("val", "test")}; tsv48 = data.windows("val", 48)
    T = lambda v: torch.tensor(v, device=dev)
    print("data:", json.dumps(data.summary()), "event rate", json.dumps(md.rate), flush=True)

    # ---------------- experts ----------------
    def cond_ref(th, tfu, sel=None):
        oh = np.eye(K_cl, dtype=np.float32)[data.cluster if sel is None else data.cluster[sel]]; cs = np.concatenate([th[:, -1, :11], tfu[:, :, :11].mean(1)], -1)
        return T(np.concatenate([np.repeat(oh[None], len(th), 0), np.repeat(cs[:, None, :], len(oh), 1)], -1))

    def load_expert(spec, role):
        """returns predict(batch, sel=None) -> z_hat [B,H,S] (numpy) ; batch = md.batch output"""
        if spec.startswith("npz:"):                                                  # pre-computed expert (e.g. zero-shot foundation model), evaluation windows only
            assert args.gate == "ref" or args.gate.startswith("ckpt:"), "a pre-computed expert has no training-window forecasts: use a frozen or saved gate"
            return None, "precomputed " + Path(spec.split(":", 1)[1]).name, 0
        if spec == "ref":
            key = "N_mlp_masked_k72_q95_film" if role == "normal" else "E_mlp_none_k72_q95_film"
            m = ResidualForecaster(data.static.shape[1], H, [], 64, backbone="mlp", k=k, n_cond=K_cl + 22).to(dev); m.load_state_dict(torch.load(CACHE / f"{key}.pt", map_location=dev)); m.eval()
            def pred(b, sel=None):
                xh, th, tfu, bf, y, ye, xg, dtf, xf = b; th, tfu = th[..., :11], tfu[..., :11]; wm = T((dtf != 0).mean(1) > 0.5); outs = []
                idx = np.arange(N) if sel is None else np.asarray(sel)
                with torch.no_grad():
                    for n0 in range(0, len(idx), 2000):
                        ii = idx[n0:n0 + 2000]; outs.append(m(T(xh[:, :, ii]), T(th), T(tfu), T(bf[:, :, ii]), static[ii], wm, cond=cond_ref(th, tfu, ii)))
                return torch.cat(outs, 2)
            return pred, key, sum(p.numel() for p in m.parameters())
        name = spec.split(":", 1)[1]; sd = torch.load(CKPT / f"{name}.pt", map_location=dev)
        m = build_cap(sd["arch"], k, H, K_cl, persist=sd.get("persist", True), step_cal=sd.get("step_cal", False), future_b=sd.get("future_b", True), cond=sd.get("cond", "func_x_cal")).to(dev); m.load_state_dict(sd["state"]); m.eval(); step_cal = sd.get("step_cal", False)
        def pred(b, sel=None):
            xh, th, tfu, bf, y, ye, xg, dtf, xf = b; idx = np.arange(N) if sel is None else np.asarray(sel); B = xh.shape[0]
            xs = xh[:, :, idx].transpose(0, 2, 1, 3).astype(np.float32); bfn = bf[:, :, idx].transpose(0, 2, 1).astype(np.float32)
            cs = np.concatenate([th[:, -1, :11], tfu[:, :, :11].mean(1)], -1).astype(np.float32); tff = tfu[:, :, :11].reshape(B, -1).astype(np.float32); outs = []
            with torch.no_grad():
                for n0 in range(0, len(idx), 2000):
                    outs.append(m(T(xs[:, n0:n0 + 2000]), T(bfn[:, n0:n0 + 2000]), T(cs), T(tff), cl[idx[n0:n0 + 2000]], T(th[:, :, :11].astype(np.float32)) if step_cal else None).permute(0, 2, 1))
            return torch.cat(outs, 2)
        return pred, f"{name} ({sd['arch']})", sum(p.numel() for p in m.parameters())

    predN, nameN, parN = load_expert(args.normal, "normal"); predE, nameE, parE = load_expert(args.extreme, "extreme")
    print("normal expert:", nameN, parN, "| extreme expert:", nameE, parE, flush=True)

    NPZ_E = np.load(args.extreme.split(":", 1)[1]) if args.extreme.startswith("npz:") else None

    def expert_preds(s):
        zn, ze = [], []
        if NPZ_E is not None:
            for i in range(0, len(ts_eval[s]), 8):
                zn.append(predN(md.batch(s, ts_eval[s][i:i + 8])).cpu().numpy().astype(np.float16))
            ze_ = np.clip(NPZ_E[s].astype(np.float32) / data.r_std, -60000, 60000).astype(np.float16); assert len(ze_) == len(ts_eval[s]), (len(ze_), len(ts_eval[s]))
            return np.concatenate(zn), ze_
        for i in range(0, len(ts_eval[s]), 8):
            b = md.batch(s, ts_eval[s][i:i + 8]); zn.append(predN(b).cpu().numpy().astype(np.float16)); ze.append(predE(b).cpu().numpy().astype(np.float16))
        return np.concatenate(zn), np.concatenate(ze)
    Z = {s: expert_preds(s) for s in ("val", "test")}; zN = {s: Z[s][0] for s in Z}; zE = {s: Z[s][1] for s in Z}; del Z
    YE = {s: np.concatenate([md.batch(s, ts_eval[s][i:i + 8])[5] for i in range(0, len(ts_eval[s]), 8)]) for s in ("val", "test")}
    DT = {s: np.concatenate([md.batch(s, ts_eval[s][i:i + 8])[7] for i in range(0, len(ts_eval[s]), 8)]) for s in ("val", "test")}
    print("expert predictions ready", round((time.time() - t0) / 60, 1), "min", flush=True)

    # ---------------- gate ----------------
    _b = md.batch("train", tr[:1]); n_g, n_cal = _b[6].shape[-1], _b[2].shape[-1]
    gate = build_gate(data.static.shape[1], H, k, 64, "linear", n_g, n_cal, 0).to(dev)
    gate.load_state_dict(torch.load(CACHE / "G_wbce_k72_q95.pt", map_location=dev)); gate_desc = "reference G_wbce_k72_q95 (frozen)"

    def gate_logits(b, sel=None):
        xh, th, tfu, bf, y, ye, xg, dtf, xf = b; idx = np.arange(N) if sel is None else np.asarray(sel); outs = []
        for n0 in range(0, len(idx), 2000):
            ii = idx[n0:n0 + 2000]; outs.append(gate(T(xh[:, :, ii]), T(xg[:, :, ii]), T(th), T(tfu), static[ii], T(xf[:, :, ii]), cl[ii]))
        return torch.cat(outs, 2)                                                                     # [B,H,S]

    def val_gate(kind):
        """stride-48 validation: PR-AUC (bce) or mixture MAE z (mix)"""
        from sklearn.metrics import average_precision_score
        gate.eval(); pp, yy, mae, cnt = [], [], 0.0, 0
        with torch.no_grad():
            for i in range(0, len(tsv48), 8):
                b = md.batch("val", tsv48[i:i + 8]); lg = gate_logits(b)
                if kind == "bce":
                    pp.append(torch.sigmoid(lg).cpu().numpy().ravel()[::7]); yy.append(b[5].ravel()[::7])
                else:
                    p = torch.sigmoid(lg); zn, ze = predN(b), predE(b); mae += float(((1 - p) * zn + p * ze - T(b[4])).abs().sum().item()); cnt += p.numel()
        return float(average_precision_score(np.concatenate(yy), np.concatenate(pp))) if kind == "bce" else -mae / max(cnt, 1)   # higher is better

    gate_hist = []
    if args.gate.startswith("ckpt:"):                                                           # a saved (e.g. mixture-tuned) gate from a previous run
        gate.load_state_dict(torch.load(CKPT / f"{args.gate.split(':', 1)[1]}.pt", map_location=dev)); gate_desc = f"{args.gate} (frozen)"
    if args.gate.startswith("refit"):
        kind = "bce" if args.gate == "refit_bce" else "mix"
        if kind == "bce":
            gate = build_gate(data.static.shape[1], H, k, 64, "linear", n_g, n_cal, 0).to(dev)      # from scratch, same architecture / inputs / loss
        pi = md.rate["train"]; pw = T((1 - pi) / pi); opt = torch.optim.AdamW(gate.parameters(), lr=args.lr if kind == "bce" else args.lr * 0.3, weight_decay=1e-4)
        steps = len(tr) // args.batch + 1; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=args.gate_epochs * steps + 1)
        best = (val_gate(kind), {k_: v.detach().cpu().clone() for k_, v in gate.state_dict().items()}, 0); bad = 0
        gate_hist.append({"epoch": 0, "val_score": best[0]}); print(json.dumps(gate_hist[-1]), flush=True)
        for ep in range(args.gate_epochs):
            gate.train(); perm = np.random.permutation(tr); losses = []
            for i in range(0, len(perm), args.batch):
                b = md.batch("train", perm[i:i + args.batch]); sel = np.random.choice(N, args.node_sample, replace=False); lg = gate_logits(b, sel); ye = T(b[5][:, :, sel])
                if kind == "bce":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, ye, pos_weight=pw)
                else:
                    with torch.no_grad():
                        zn, ze = predN(b, sel), predE(b, sel)
                    p = torch.sigmoid(lg); loss = torch.nn.functional.huber_loss((1 - p) * zn + p * ze, T(b[4][:, :, sel]), delta=1.0)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step(); sched.step(); losses.append(loss.item())
            sc = val_gate(kind); gate_hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_score": sc, "min": round((time.time() - t0) / 60, 1)}); print(json.dumps(gate_hist[-1]), flush=True)
            if sc > best[0] + 1e-6:
                best = (sc, {k_: v.detach().cpu().clone() for k_, v in gate.state_dict().items()}, ep + 1); bad = 0
            else:
                bad += 1
                if bad >= args.gate_patience:
                    break
        gate.load_state_dict(best[1]); torch.save(gate.state_dict(), CKPT / f"{args.name}_gate.pt"); gate_desc = f"{args.gate} (best epoch {best[2]}, val score {best[0]:.4f})"
    gate.eval(); LG = {}
    for s in ("val", "test"):
        with torch.no_grad():
            LG[s] = np.concatenate([gate_logits(md.batch(s, ts_eval[s][i:i + 8])).cpu().numpy().astype(np.float16) for i in range(0, len(ts_eval[s]), 8)])
    print("gate:", gate_desc, round((time.time() - t0) / 60, 1), "min", flush=True)

    # ---------------- evaluation (as train_moe) ----------------
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    result = {"name": args.name, "args": vars(args), "normal": nameN, "extreme": nameE, "params": {"normal": parN, "extreme": parE}, "gate": gate_desc, "gate_history": gate_hist, "event_rate": md.rate, "models": {}, "gates": {}}
    rng = np.random.default_rng(0)

    def regime(split, pz):
        Xh, Rh, X, R, B = data.people(split, ts_eval[split], pz); er = np.abs(Rh - R); e = YE[split] > 0
        return {"MAE_N": float(er[~e].mean()), "MAE_E": float(er[e].mean()), "share_E": float(e.mean())}

    def evaluate(label, preds):
        result["models"][label] = {}
        for s in ("val", "test"):
            pz = preds[s]() if callable(preds[s]) else preds[s]; pz = pz.astype(np.float32); reg = regime(s, pz); gc.collect()
            m = metrics(data, s, ts_eval[s], pz, DT[s]); m.update(reg); result["models"][label][s] = m; del pz; gc.collect()
            print(s, label, json.dumps({k_: round(m[k_], 4) for k_ in ("X_wmape", "R_mae", "MAE_N", "MAE_E", "R_rel_err_top1pct", "extreme_recall_top1", "extreme_f1_top1", "peak_magnitude_rel_err", "top5pct_spatial_jaccard")}), flush=True)

    evaluate("normal_only", zN); evaluate("extreme_only", zE)
    evaluate("oracle", {s: (lambda s=s: (1 - YE[s]) * zN[s].astype(np.float32) + YE[s] * zE[s].astype(np.float32)) for s in zN})
    P = {s: (1 / (1 + np.exp(-LG[s].astype(np.float32)))).astype(np.float32) for s in LG}; g = {}
    for s in P:
        y = YE[s].ravel(); p = P[s].ravel(); sub = rng.choice(len(y), min(len(y), 4_000_000), replace=False); ece, rel = ece_score(p[sub], y[sub])
        g[s] = {"auprc": float(average_precision_score(y[sub], p[sub])), "auroc": float(roc_auc_score(y[sub], p[sub])), "brier": float(brier_score_loss(y[sub], p[sub])), "ece": ece, "mean_p": float(p.mean()), "event_rate": float(y.mean())}
    yv = YE["val"].ravel(); lv = LG["val"].astype(np.float32).ravel(); sub = rng.choice(len(yv), 2_000_000, replace=False)
    platt = LogisticRegression(C=1e6, max_iter=200).fit(lv[sub, None], yv[sub]); Pp = {s: platt.predict_proba(LG[s].astype(np.float32).reshape(-1, 1))[:, 1].reshape(LG[s].shape).astype(np.float32) for s in LG}
    for s in Pp:
        y = YE[s].ravel(); p = Pp[s].ravel(); sub2 = rng.choice(len(y), min(len(y), 4_000_000), replace=False); ece, _ = ece_score(p[sub2], y[sub2]); g[s]["brier_platt"] = float(brier_score_loss(y[sub2], p[sub2])); g[s]["ece_platt"] = ece
    result["gates"]["gate"] = g; print("gate metrics", json.dumps({s: {k_: round(v, 4) for k_, v in g[s].items()} for s in g}), flush=True)
    mix = lambda W, s: (lambda: (1 - W[s]) * zN[s].astype(np.float32) + W[s] * zE[s].astype(np.float32))
    evaluate("soft", {s: mix(P, s) for s in P}); evaluate("soft_platt", {s: mix(Pp, s) for s in Pp})
    if NPZ_E is not None:
        def quick(W):
            pz = ((1 - W) * zN["val"].astype(np.float32) + W * zE["val"].astype(np.float32)); Xh, Rh, X, R, B = data.people("val", ts_eval["val"], pz); v = float(np.abs(Xh - X).sum() / X.sum()); del pz, Xh, Rh, X, R, B; gc.collect(); return v
        cands = {f"const w={w}": (lambda s, w=w: np.float32(w)) for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)}
        cands.update({f"Platt p x {c}": (lambda s, c=c: np.clip(c * Pp[s], 0, 1)) for c in (1, 2, 4, 8)}); cands.update({f"raw p x {c}": (lambda s, c=c: np.clip(c * P[s], 0, 1)) for c in (0.5, 1, 2)})
        if args.blend_set == "const":
            cands = {k_: f_ for k_, f_ in cands.items() if k_.startswith("const")}
        sc = {k_: quick(f_("val")) for k_, f_ in cands.items()}; best = min(sc, key=sc.get); result["blend_candidates_val_wmape"] = sc; result["blend_desc"] = best; print("blend (val wMAPE):", json.dumps({k_: round(v, 5) for k_, v in sc.items()}), "->", best, flush=True)
        evaluate("blend_val", {s: (lambda s=s: ((1 - cands[best](s)) * zN[s].astype(np.float32) + cands[best](s) * zE[s].astype(np.float32))) for s in P})
    cand = {}
    for tau in (0.3, 0.5, 0.7):
        pv = np.where(P["val"] > tau, zE["val"], zN["val"]).astype(np.float32); cand[tau] = metrics(data, "val", ts_eval["val"], pv, DT["val"])["R_mae"]; del pv
    tau = min(cand, key=cand.get); result["gates"]["gate"]["hard_tau_val_R_mae"] = {str(t_): float(v) for t_, v in cand.items()}; result["gates"]["gate"]["tau_selected"] = tau
    evaluate(f"hard_tau{tau}", {s: (lambda s=s: np.where(P[s] > tau, zE[s], zN[s]).astype(np.float32)) for s in P})
    result["minutes"] = round((time.time() - t0) / 60, 1)
    (RES / f"{args.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8"); print("saved", RES / f"{args.name}.json")


if __name__ == "__main__":
    main()
