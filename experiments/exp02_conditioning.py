"""Experiment 2 - conditioning ablation on the final encoder (FiLM-MLP-3L, 168-512-256-128).

FiLM source: none / func (cluster embedding) / cal (calendar summary) / func_cal (concatenation) / func_x_cal (with outer product,
the final model).  Seeds 0, 1, 2; AdamW 3e-3, wd 0, OneCycle <= 48 epochs, early stopping (val wMAPE, patience 7) + annealing.
The func_x_cal runs already exist from the Task 16 study (t16c_B_A5_lr0.003_wd0, t16c_S_A5_s1, t16c_S_A5_s2) and are copied in.
Outputs results/exp02_conditioning/{*.json, summary.csv, summary_mean.csv}.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp02_conditioning"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ORIG = ROOT / "results" / "base" / "results"
REUSE = {("func_x_cal", 0): "t16c_B_A5_lr0.003_wd0", ("func_x_cal", 1): "t16c_S_A5_s1", ("func_x_cal", 2): "t16c_S_A5_s2"}
CONDS = ["none", "func", "cal", "func_cal", "func_x_cal"]


def run(name, *argv):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(CK), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name] + [str(a) for a in argv]; print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def main():
    rows = []
    for cond in CONDS:
        for sd in (0, 1, 2):
            name = f"exp02_{cond}_s{sd}"
            if (cond, sd) in REUSE and not (OUT / f"{name}.json").exists() and (ORIG / f"{REUSE[(cond, sd)]}.json").exists():
                shutil.copy(ORIG / f"{REUSE[(cond, sd)]}.json", OUT / f"{name}.json")
            r = run(name, "--arch", "mlp:512,256,128", "--cond", cond, "--epochs", 48, "--patience", 7, "--lr", 0.003, "--wd", 0.0, "--seed", sd)
            if r:
                m = r["models"]["model"]; t = m["test"]
                rows.append({"cond": cond, "seed": sd, "params": r["params"], "val_wMAPE": m["val"]["X_wmape"] * 100, "test_wMAPE": t["X_wmape"] * 100, "MAE_E": t["MAE_E"], "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes")})
            pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    if len(df):
        agg = df.groupby("cond")[["val_wMAPE", "test_wMAPE", "MAE_E", "top1", "recall", "F1"]].agg(["mean", "std"]).round(3); agg.to_csv(OUT / "summary_mean.csv", encoding="utf-8-sig"); print(agg.to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
