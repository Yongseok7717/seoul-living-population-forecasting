"""Experiment 7 - patch-token Transformer (PatchTST-style) under the same frame as M2 / M3.

The hourly-token encoder Transformer (M3) trails the MLP.  A common objection is that hourly tokens are a poor tokenisation; PatchTST
uses patches of the history as tokens with a flatten head.  Here: tokens = patches of the 72 h history (z, log B, per-step calendar),
channel-independent per cell, 3 encoder layers d=128, flatten head -> 128, then the same FiLM conditioning, calendar head and persistence
anchor as M2/M3 (no future-log-B branch, as for the best M3).  AdamW 3e-4, wd 0, OneCycle <= 24 epochs, patience 5 + annealing.
  P12  patch 12 h, stride 6  (11 tokens)      P24  patch 24 h, stride 12 (5 tokens)
The better one on validation is repeated with seeds 1, 2.  Outputs results/exp07_patchtst/.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp07_patchtst"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ARCH = {"P12": "ptst:128,3,4,256,0.1,12,6", "P24": "ptst:128,3,4,256,0.1,24,12"}


def run(name, arch, seed):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(CK), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name, "--arch", arch, "--step_cal", "--no_future_b", "--epochs", "24", "--patience", "5", "--lr", "0.0003", "--wd", "0.0", "--node_chunk", "2000", "--seed", str(seed)]
    print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def row(tag, seed, r):
    m = r["models"]["model"]; t = m["test"]
    return {"model": tag, "seed": seed, "params": r["params"], "val_wMAPE": m["val"]["X_wmape"] * 100, "test_wMAPE": t["X_wmape"] * 100, "MAE_E": t["MAE_E"], "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes")}


def main():
    rows = []; res = {}
    for tag, arch in ARCH.items():
        res[tag] = run(f"exp07_{tag}", arch, 0)
        if res[tag]:
            rows.append(row(tag, 0, res[tag])); pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    ok = {k_: v for k_, v in res.items() if v}
    if ok:
        best = min(ok, key=lambda k_: ok[k_]["models"]["model"]["val"]["X_wmape"]); print("best on validation:", best, flush=True)
        for sd in (1, 2):
            r = run(f"exp07_{best}_s{sd}", ARCH[best], sd)
            if r:
                rows.append(row(best, sd, r)); pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows); print(df.round(3).to_string(), flush=True)
    if len(df):
        print(df.groupby("model")[["val_wMAPE", "test_wMAPE", "MAE_E", "top1", "recall", "F1"]].agg(["mean", "std"]).round(3).to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
