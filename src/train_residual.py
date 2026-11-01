"""Task 4: residual forecasting with relational ablations.

    python train_residual.py --name r_k24_none --k 24 --relations none
    python train_residual.py --name r_k24_spatial --k 24 --relations spatial
    python train_residual.py --name r_k24_all --k 24 --relations spatial,func_soft,facility
    python train_residual.py --name r_zero --baseline_only           # X_hat = B (zero residual)
    python train_residual.py --name r_persist --persistence          # R_hat = last observed residual

relations: none | spatial | func_knn | func_hard | func_soft | func_dynamic | facility, comma-separated.
Results -> results/base/results/<name>.json ; metrics on val (2024) and test (2025).
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_data import ResidualData, ML_DIR   # noqa: E402

RES = ML_DIR / "results"
HORIZONS = (1, 3, 6, 12, 24)


def metrics(data, split, ts, pred_z, dtf):
    Xhat, Rhat, X, R, B = data.people(split, ts, pred_z)
    e = Xhat - X; er = Rhat - R
    out = {"n_windows": int(len(ts)),
           "X_mae": float(np.abs(e).mean()), "X_rmse": float(np.sqrt((e ** 2).mean())), "X_wmape": float(np.abs(e).sum() / X.sum()),
           "R_mae": float(np.abs(er).mean()), "R_rmse": float(np.sqrt((er ** 2).mean())),
           "R_wmape": float(np.abs(er).sum() / np.abs(R).sum()),
           "X_wmape_h": {str(h): float(np.abs(e[:, h - 1]).sum() / X[:, h - 1].sum()) for h in HORIZONS},
           "R_mae_h": {str(h): float(np.abs(er[:, h - 1]).mean()) for h in HORIZONS}}
    # top-q% residual cells-hours (by actual |R|)
    absR = np.abs(R)
    for q in (10, 5, 1):
        thr = np.quantile(absR, 1 - q / 100); m = absR >= thr
        out[f"R_mae_top{q}pct"] = float(np.abs(er[m]).mean())
        out[f"R_rel_err_top{q}pct"] = float(np.abs(er[m]).sum() / absR[m].sum())
    # extreme recall / precision on positive surges (top 1% of R)
    thr = np.quantile(R, 0.99); ext = R >= thr; pext = Rhat >= thr
    out["extreme_recall_top1"] = float((ext & pext).sum() / max(ext.sum(), 1)); out["extreme_precision_top1"] = float((ext & pext).sum() / max(pext.sum(), 1))
    thr5 = np.quantile(R, 0.95); ext5 = R >= thr5; pext5 = Rhat >= thr5
    out["extreme_recall_top5"] = float((ext5 & pext5).sum() / max(ext5.sum(), 1)); out["extreme_precision_top5"] = float((ext5 & pext5).sum() / max(pext5.sum(), 1))
    for q in ("top1", "top5"):
        p, r_ = out[f"extreme_precision_{q}"], out[f"extreme_recall_{q}"]
        out[f"extreme_f1_{q}"] = float(2 * p * r_ / (p + r_)) if (p + r_) > 0 else 0.0
    # AUPRC (score = R_hat, label = surge) on a window subsample to bound memory/time
    from sklearn.metrics import average_precision_score
    sub = slice(0, None, max(1, len(ts) // 40)); sc = Rhat[sub].ravel()
    for q, ex in (("top1", ext), ("top5", ext5)):
        lab = ex[sub].ravel(); out[f"auprc_{q}"] = float(average_precision_score(lab, sc)) if lab.any() else None
    # normal vs extreme periods: window-hours whose city-wide |R| mass is in the top 10%
    cityR = absR.sum(2); ext_p = cityR >= np.quantile(cityR, 0.9)
    out["R_mae_normal_period"] = float(np.abs(er[~ext_p]).mean()); out["R_mae_extreme_period"] = float(np.abs(er[ext_p]).mean())
    out["X_wmape_normal_period"] = float(np.abs(e[~ext_p]).sum() / X[~ext_p].sum()); out["X_wmape_extreme_period"] = float(np.abs(e[ext_p]).sum() / X[ext_p].sum())
    out["R_rel_err_extreme_period"] = float(np.abs(er[ext_p]).sum() / absR[ext_p].sum())
    # by month of the first target hour
    mon = data.ti[split].datetime.dt.month.values[np.asarray(ts) + 1]
    out["X_wmape_by_month"] = {str(int(m)): float(np.abs(e[mon == m]).sum() / X[mon == m].sum()) for m in np.unique(mon)}
    out["R_mae_by_month"] = {str(int(m)): float(np.abs(er[mon == m]).mean()) for m in np.unique(mon)}
    # peak magnitude / timing per window (max over cells and hours of R)
    pk_a = R.reshape(len(ts), -1).max(1); pk_p = Rhat.reshape(len(ts), -1).max(1)
    out["peak_magnitude_rel_err"] = float(np.abs(pk_p - pk_a).sum() / np.abs(pk_a).sum())
    ha = R.max(2).argmax(1); hp = Rhat.max(2).argmax(1)
    out["peak_timing_mae_h"] = float(np.abs(ha - hp).mean())
    # spatial overlap of top-5% residual cells per window-hour (Jaccard)
    kk = max(1, int(0.05 * R.shape[2])); jac = []
    for b in range(0, len(ts), max(1, len(ts) // 60)):
        for h in (0, 5, 11, 23):
            ta = set(np.argpartition(-R[b, h], kk)[:kk]); tp = set(np.argpartition(-Rhat[b, h], kk)[:kk])
            jac.append(len(ta & tp) / len(ta | tp))
    out["top5pct_spatial_jaccard"] = float(np.mean(jac))
    # by cluster / by daytype
    out["X_wmape_by_cluster"] = {str(c): float(np.abs(e[:, :, data.cluster == c]).sum() / X[:, :, data.cluster == c].sum()) for c in range(int(data.cluster.max()) + 1)}
    out["R_mae_by_cluster"] = {str(c): float(np.abs(er[:, :, data.cluster == c]).mean()) for c in range(int(data.cluster.max()) + 1)}
    wkend = (dtf != 0).mean(1) > 0.5
    out["R_mae_weekday"] = float(np.abs(er[~wkend]).mean()) if (~wkend).any() else None
    out["R_mae_weekend"] = float(np.abs(er[wkend]).mean()) if wkend.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=24); ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--relations", default="none")
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_stride", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--baseline_only", action="store_true"); ap.add_argument("--persistence", action="store_true")
    ap.add_argument("--loss", default="mse", choices=["mse", "mae", "huber", "tail"]); ap.add_argument("--tail_lambda", type=float, default=4.0)
    ap.add_argument("--backbone", default="gru", choices=["gru", "transformer", "linear", "mlp"])
    ap.add_argument("--node_sample", type=int, default=None, help="train on a random subset of nodes per step (relations must be none)")
    ap.add_argument("--seasonal_persistence", action="store_true", help="R_hat(t+h) = R(t+h-168): needs k >= 168+H-1")
    ap.add_argument("--target", default="residual", choices=["residual", "raw"], help="raw: no baseline decomposition, model predicts X")
    ap.add_argument("--source", default="X", choices=["X", "L"], help="L: forecast the causal regular component (F2) instead of the observed population")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); t0 = time.time()
    data = ResidualData(k=args.k, horizon=args.H, source=args.source)
    print("data:", json.dumps(data.summary()), flush=True)
    result = {"name": args.name, "args": vars(args), "data": data.summary()}

    if args.baseline_only or args.persistence or args.seasonal_persistence:
        for split in ("val", "test"):
            ts = data.windows(split, args.eval_stride)
            preds, dts = [], []
            for i in range(0, len(ts), 64):
                xh, th, tfu, bf, y, dtf = data.batch(split, ts[i:i + 64])
                if args.baseline_only:
                    pr = np.zeros_like(y)
                elif args.seasonal_persistence and data.k >= 168 + data.H - 1:
                    s0 = data.k - 168                                  # index of t+1-168 inside the history window
                    pr = xh[:, s0:s0 + data.H, :, 0]
                else:
                    pr = np.repeat(xh[:, -1:, :, 0], data.H, axis=1)
                preds.append(pr); dts.append(dtf)
            result[split] = metrics(data, split, ts, np.concatenate(preds), np.concatenate(dts))
            print(split, json.dumps({k: v for k, v in result[split].items() if not isinstance(v, dict)}), flush=True)
    else:
        import torch
        from residual_models import ResidualForecaster
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rel = [] if args.relations == "none" else args.relations.split(",")
        ops = [data.operator(r) for r in rel]
        if args.node_sample:
            assert not ops, "node sampling only with relations=none"
        if args.target == "raw":
            data.use_raw_target()
        model = ResidualForecaster(data.static.shape[1], data.H, ops, args.hidden, backbone=args.backbone, k=data.k).to(dev)
        static = torch.tensor(data.static, device=dev)
        n_params = sum(p.numel() for p in model.parameters()); print(f"{args.name}: relations {rel} params {n_params:,}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        tr = data.windows("train", 1)
        steps_per_epoch = (len(tr) // args.batch + 1) if not args.max_steps else args.max_steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch + 1)
        q95 = float(np.nanquantile(data.RP["train"], 0.95))

        def loss_fn(pred, y):
            if args.loss == "mse":
                return torch.nn.functional.mse_loss(pred, y)
            if args.loss == "mae":
                return (pred - y).abs().mean()
            if args.loss == "huber":
                return torch.nn.functional.huber_loss(pred, y, delta=1.0)
            w = 1 + args.tail_lambda * (y * torch.tensor(data.r_std, device=dev) > q95).float()
            return (w * (pred - y).abs()).mean()

        def predict(xh, th, tfu, bf, dtf):
            model.eval(); outs = []
            bs = 8 if data.k <= 24 else (4 if data.k <= 72 else 2)
            with torch.no_grad():
                for i in range(0, len(xh), bs):
                    wm = torch.tensor((dtf[i:i + bs] != 0).mean(1) > 0.5, device=dev)
                    if ops:
                        o = model(*(torch.tensor(a[i:i + bs], device=dev) for a in (xh, th, tfu, bf)), static, wm)
                    else:                                              # no relations: nodes are independent, chunk them
                        parts = []
                        for n0 in range(0, data.N, 2000):
                            parts.append(model(torch.tensor(xh[i:i + bs, :, n0:n0 + 2000], device=dev), torch.tensor(th[i:i + bs], device=dev),
                                               torch.tensor(tfu[i:i + bs], device=dev), torch.tensor(bf[i:i + bs, :, n0:n0 + 2000], device=dev),
                                               static[n0:n0 + 2000], wm))
                        o = torch.cat(parts, dim=2)
                    outs.append(o.cpu().numpy())
            return np.concatenate(outs)

        def evaluate(split, stride):
            ts = data.windows(split, stride); preds, dts = [], []
            for i in range(0, len(ts), 32):
                xh, th, tfu, bf, y, dtf = data.batch(split, ts[i:i + 32]); preds.append(predict(xh, th, tfu, bf, dtf)); dts.append(dtf)
            return metrics(data, split, ts, np.concatenate(preds), np.concatenate(dts))

        history, best = [], (np.inf, None)
        for ep in range(args.epochs):
            model.train(); perm = np.random.permutation(tr); losses = []; te = time.time()
            for step, i in enumerate(range(0, len(perm), args.batch)):
                if args.max_steps and step >= args.max_steps:
                    break
                xh, th, tfu, bf, y, dtf = data.batch("train", perm[i:i + args.batch])
                wm = torch.tensor((dtf != 0).mean(1) > 0.5, device=dev)
                st = static
                if args.node_sample:
                    sel = np.random.choice(data.N, args.node_sample, replace=False)
                    xh, bf, y, st = xh[:, :, sel], bf[:, :, sel], y[:, :, sel], static[sel]
                xh, th, tfu, bf, y = (torch.tensor(a, device=dev) for a in (xh, th, tfu, bf, y))
                pred = model(xh, th, tfu, bf, st, wm)
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                losses.append(loss.item())
            val = evaluate("val", 48)
            rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "val_R_mae": val["R_mae"], "val_X_wmape": val["X_wmape"], "val_top1_relerr": val["R_rel_err_top1pct"], "sec": time.time() - te}
            history.append(rec); print(json.dumps(rec), flush=True)
            if val["R_mae"] < best[0]:
                best = (val["R_mae"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[1]); torch.save(model.state_dict(), RES / f"{args.name}.pt")
        result["history"] = history; result["n_params"] = n_params
        for split in ("val", "test"):
            result[split] = evaluate(split, args.eval_stride)
            print(split, json.dumps({k: v for k, v in result[split].items() if not isinstance(v, dict)}), flush=True)
    result["elapsed_sec"] = time.time() - t0
    (RES / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", RES / f"{args.name}.json")


if __name__ == "__main__":
    main()
