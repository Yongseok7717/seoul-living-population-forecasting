"""Experiment 11 - training-data recency versus training-set size (control for the rolling result of exp01).

For each test quarter Q of 2025 (validation = the two months before Q) three M2 models are compared, all with the same
architecture, procedure and validation months:
  old       10-month window 2023-03..2023-12                       (same length as the recent window, older data)
  recent    10-month window ending at the validation months        (same length, most recent data)
  expanding all data from 2023-03 to the validation months          (= the rolling model of exp01, reused)
Reading:  recent << old  -> recency (distribution shift) matters;  expanding ~ recent -> older data add little;
          expanding < recent -> both recency and sample size matter.
Outputs results/exp11_recency/{*.json, summary.csv}
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp11_recency"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ROLL = ROOT / "results" / "exp01_rolling"
FOLDS = {"Q1": ("202411-202412", "202501-202503", "202401-202410"), "Q2": ("202502-202503", "202504-202506", "202404-202501"),
         "Q3": ("202505-202506", "202507-202509", "202407-202504"), "Q4": ("202508-202509", "202510-202512", "202410-202507")}   # val, test, recent 10-month train
OLD = "202303-202312"; ARCH = "mlp:512,256,128"


def run(name, splits):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, RD_SPLITS=",".join(splits), LP_RESULTS=str(OUT), LP_CKPT=str(CK), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name, "--arch", ARCH, "--epochs", "36", "--patience", "7", "--lr", "0.003", "--wd", "0.0"]
    print(time.strftime("%m-%d %H:%M"), "RUN", name, splits, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def row(r, fold, kind, train):
    if r is None:
        return None
    m = r["models"]["model"]; t = m["test"]
    return {"fold": fold, "training": kind, "train_window": train, "n_train_windows": r["data"]["windows"]["train"] if "data" in r else None, "test_wMAPE": 100 * t["X_wmape"], "MAE_E": t.get("MAE_E"), "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "val_wMAPE": 100 * m["val"]["X_wmape"], "best_epoch": r.get("best_epoch")}


def main():
    rows = []
    for q, (va, te, rec) in FOLDS.items():
        rows.append(row(run(f"old_{q}_A5", (OLD, va, te)), q, "old (10 months, 2023)", OLD))
        rows.append(row(run(f"recent_{q}_A5", (rec, va, te)), q, "recent (10 months)", rec))
        f = ROLL / f"roll_{q}_A5.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8")); rows.append(row(r, q, "expanding (2023-03 to val)", f"202303-{va[:6]}"))
        pd.DataFrame([x for x in rows if x]).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame([x for x in rows if x]); pd.set_option("display.width", 220); print(df.round(3).to_string())
    print(df.groupby("training")[["test_wMAPE", "MAE_E", "top1", "recall"]].mean().round(3).to_string()); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
