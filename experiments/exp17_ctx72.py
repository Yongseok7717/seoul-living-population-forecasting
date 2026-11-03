"""Experiment 17 - every zero-shot comparison with the SAME 72 h history as M2 (Chronos-Bolt context = 72 h), raw (R5) and residual (R6).

  (a) standard protocol (train 2023 / val 2024 / test 2025, origins every 6 h): forecasts for val and test, then train_moe_f with
      normal = fixed M2, extreme = the zero-shot forecast and no gate -> zero-shot alone (R5, R6), oracle, and the blend B1
      (1-w) M2 + w zero-shot with constant w selected on validation;
  (b) rolling protocol (four quarters, folds of exp01): forecasts for each fold's validation months and test quarter, assembly with the
      rolling M2 in the same way -> alone, oracle, blend by quarter;
  (c) 2023 training year (windows with t >= 72): zero-shot metrics next to the in-sample M2 of exp16.
Forecasts are stored as residual-equivalents (raw: max(X_hat,0) - B).  Outputs results/exp17_ctx72/{summary_std.csv,
summary_rolling.csv, summary_rolling_mean.csv, summary_train.csv}.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp17_ctx72"; LOGS = OUT / "logs"
for d_ in (OUT, LOGS):
    d_.mkdir(parents=True, exist_ok=True)
ROLL_CKPT = ROOT / "results" / "exp01_rolling" / "ckpt"; ORIG_CKPT = ROOT / "results" / "base" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}
CTX, H, MODEL = 72, 24, "amazon/chronos-bolt-small"
LAB = {"normal_only": "M2 alone", "extreme_only": "zero-shot alone", "oracle": "oracle gate", "blend_val": "blend (val-selected)"}


def worker(tag, mode, splits):
    """subprocess; RD_SPLITS set for rolling folds. mode = raw | residual. splits = comma list of splits to forecast"""
    sys.path.insert(0, str(SRC))
    import torch
    from chronos import BaseChronosPipeline
    from train_moe import MoEData
    from train_residual import metrics
    md = MoEData(72, H, "q95"); d = md.d; pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
    Xall = np.concatenate([d.X[s] for s in ("train", "val", "test")]).astype(np.float32); Ball = np.concatenate([d.B[s] for s in ("train", "val", "test")]).astype(np.float32)
    S = Xall if mode == "raw" else Xall - Ball; off = {"train": 0, "val": len(d.X["train"]), "test": len(d.X["train"]) + len(d.X["val"])}; out = {}; rows = []
    for split in splits.split(","):
        ts = d.windows(split, 6); ts = ts[ts >= CTX] if split == "train" else ts; P = np.zeros((len(ts), H, d.N), np.float32); t0 = time.time()
        for wi, t in enumerate(ts):
            a = off[split] + t + 1; ctx = torch.tensor(S[a - CTX:a].T)
            for n0 in range(0, d.N, 4096):
                q, _ = pipe.predict_quantiles(ctx[n0:n0 + 4096], prediction_length=H, quantile_levels=[0.5]); P[wi, :, n0:n0 + 4096] = q[..., 0].float().cpu().numpy().T
            if wi % 300 == 0:
                print(tag, split, wi, len(ts), round((time.time() - t0) / 60, 1), "min", flush=True)
        Bt = np.stack([d.B[split][t + 1:t + 1 + H] for t in ts]); Pr = (np.maximum(P, 0) - Bt) if mode == "raw" else P
        out[split] = np.clip(Pr, -65000, 65000).astype(np.float16)
        DT = np.stack([d.dt[split][t + 1:t + 1 + H] for t in ts]); E = np.stack([md.E[split][t + 1:t + 1 + H] for t in ts]) > 0
        pz = (Pr / d.r_std).astype(np.float32); m = metrics(d, split, ts, pz, DT); Xh, Rh, X, R, B = d.people(split, ts, pz); er = np.abs(Rh - R)
        rows.append({"tag": tag, "mode": mode, "split": split, "wMAPE": 100 * m["X_wmape"], "MAE_N": float(er[~E].mean()), "MAE_E": float(er[E].mean()), "top1": m["R_rel_err_top1pct"], "recall": m["extreme_recall_top1"], "F1": m["extreme_f1_top1"], "peak_mag": m["peak_magnitude_rel_err"]})
        print(rows[-1], flush=True); del P, Pr, pz, Xh, Rh, X, R, B, er
    np.savez(OUT / f"chronos_{tag}.npz", **out); pd.DataFrame(rows).to_csv(OUT / f"metrics_{tag}.csv", index=False, encoding="utf-8-sig")


def call(name, splits, cmd, log, env_extra=None):
    env = dict(os.environ, LP_RESULTS=str(OUT), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    if splits:
        env["RD_SPLITS"] = ",".join(splits)
    env.update(env_extra or {}); print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{log}.log", "w", encoding="utf-8") as lg:
        return subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)


def rows_of(r, extra):
    rows = []
    for mix, nm in LAB.items():
        if mix in r["models"]:
            t = r["models"][mix]["test"]; rows.append({**extra, "model": nm, "desc": r.get("blend_desc", "") if mix == "blend_val" else "", "test_wMAPE": 100 * t["X_wmape"], "MAE_N": t["MAE_N"], "MAE_E": t["MAE_E"],
                                                       "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "peak_mag": t["peak_magnitude_rel_err"]})
    return rows


def main():
    me = str(Path(__file__).resolve()); rows = []
    # (a) + (c): standard protocol, train/val/test in one worker per mode
    for mode in ("raw", "residual"):
        tag = f"std_{mode}"; npz = OUT / f"chronos_{tag}.npz"
        if not npz.exists():
            rc = call(f"chronos_{tag}", None, [sys.executable, me, "--worker", tag, mode, "train,val,test"], f"chronos_{tag}")
            if rc != 0 or not npz.exists():
                print("FAILED", tag, rc, flush=True); continue
        name = f"exp17_{tag}"; f = OUT / f"{name}.json"
        if not f.exists():
            rc = call(name, None, [sys.executable, str(SRC / "train_moe_f.py"), "--name", name, "--normal", "ckpt:t16c_B_A5_lr0.003_wd0", "--extreme", f"npz:{npz}", "--gate", "none"], name, {"LP_CKPT": str(ORIG_CKPT)})
            if rc != 0 or not f.exists():
                print("FAILED", name, rc, flush=True); continue
        rows += rows_of(json.loads(f.read_text(encoding="utf-8")), {"mode": mode})
        pd.DataFrame(rows).to_csv(OUT / "summary_std.csv", index=False, encoding="utf-8-sig")
    tr = [pd.read_csv(OUT / f"metrics_std_{m}.csv") for m in ("raw", "residual") if (OUT / f"metrics_std_{m}.csv").exists()]
    if tr:
        pd.concat(tr).to_csv(OUT / "summary_train.csv", index=False, encoding="utf-8-sig")
    # (b) rolling folds
    rows = []
    for mode in ("raw", "residual"):
        for q, fold in FOLDS.items():
            tag = f"{q}_roll_{mode}"; npz = OUT / f"chronos_{tag}.npz"; name = f"exp17_{tag}"
            if not npz.exists():
                rc = call(f"chronos_{tag}", fold, [sys.executable, me, "--worker", tag, mode, "val,test"], f"chronos_{tag}")
                if rc != 0 or not npz.exists():
                    print("FAILED", tag, rc, flush=True); continue
            f = OUT / f"{name}.json"
            if not f.exists():
                rc = call(name, fold, [sys.executable, str(SRC / "train_moe_f.py"), "--name", name, "--normal", f"ckpt:roll_{q}_A5", "--extreme", f"npz:{npz}", "--gate", "none"], name, {"LP_CKPT": str(ROLL_CKPT)})
                if rc != 0 or not f.exists():
                    print("FAILED", name, rc, flush=True); continue
            rows += rows_of(json.loads(f.read_text(encoding="utf-8")), {"mode": mode, "fold": q})
            pd.DataFrame(rows).to_csv(OUT / "summary_rolling.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    if len(df):
        mean = df.groupby(["mode", "model"])[["test_wMAPE", "MAE_N", "MAE_E", "top1", "recall", "F1", "peak_mag"]].mean().round(3); mean.to_csv(OUT / "summary_rolling_mean.csv", encoding="utf-8-sig"); print(mean.to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
