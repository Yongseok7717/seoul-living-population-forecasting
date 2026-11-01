"""Experiment 1 - rolling re-training (T1).

Question: how much of the forecast error of the 2023-trained model is year-to-year drift?  For each test quarter of 2025 the
main model is re-trained on all data up to two months before the quarter (validation = those two months) and compared with
the fixed model trained on 2023-03..12 only, evaluated on the same quarter with its own normalisation.

Folds (train, val, test):
  Q1  202303-202410 | 202411-202412 | 202501-202503
  Q2  202303-202501 | 202502-202503 | 202504-202506
  Q3  202303-202504 | 202505-202506 | 202507-202509
  Q4  202303-202507 | 202508-202509 | 202510-202512
Models per fold: M2 FiLM-MLP-3L (seed 0), E2 masked extreme expert, X2 MoE (M2 + E2 + mixture-tuned gate, raw p);
fixed counterparts: t16c_B_A5_lr0.003_wd0 / t16c_E_E2_long / moef_F4mix gate evaluated with RD_SPLITS train=202303-202312.
Resumable (existing result json = skip).  Outputs results/exp01_rolling/{*.json, ckpt/, logs/, summary.csv}.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp01_rolling"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ORIG_CKPT = ROOT / "results" / "base" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}
ARCH_N, ARCH_E = "mlp:512,256,128", "mlp:256,256,128,64"


def run(script, name, splits, ckpt_dir, *argv):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, RD_SPLITS=",".join(splits), LP_RESULTS=str(OUT), LP_CKPT=str(ckpt_dir), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(SRC / script), "--name", name] + [str(a) for a in argv]; print(time.strftime("%m-%d %H:%M"), "RUN", name, splits, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def row(r, fold, model, kind, key=("models", "model")):
    if r is None:
        return None
    m = r
    for k in key:
        m = m[k]
    t = m["test"]
    return {"fold": fold, "model": model, "kind": kind, "name": r["name"], "test_wMAPE": t["X_wmape"] * 100, "resid_explained": 1 - t["R_wmape"], "MAE_N": t.get("MAE_N"), "MAE_E": t.get("MAE_E"),
            "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "peak_mag": t["peak_magnitude_rel_err"], "jaccard": t["top5pct_spatial_jaccard"],
            "val_wMAPE": m["val"]["X_wmape"] * 100, "epochs": r.get("epochs_run"), "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes", r.get("minutes"))}


def main():
    rows = []
    for fold, (tr, va, te) in FOLDS.items():
        fixed = ("202303-202312", "202411-202412", te)
        # fixed 2023 model on this quarter (own normalisation)
        rows.append(row(run("train_cap.py", f"roll_{fold}_A5fixed", fixed, ORIG_CKPT, "--arch", ARCH_N, "--mode", "eval", "--init", "t16c_B_A5_lr0.003_wd0"), fold, "M2 FiLM-MLP-3L", "fixed 2023"))
        rows.append(row(run("train_moe_f.py", f"roll_{fold}_X2fixed", fixed, ORIG_CKPT, "--normal", "ckpt:t16c_B_A5_lr0.003_wd0", "--extreme", "ckpt:t16c_E_E2_long", "--gate", "ckpt:moef_F4mix_gate"), fold, "X2 MoE", "fixed 2023", ("models", "soft")))
        # rolling re-trained
        rN = run("train_cap.py", f"roll_{fold}_A5", (tr, va, te), CK, "--arch", ARCH_N, "--epochs", 36, "--patience", 7, "--lr", 0.003, "--wd", 0.0)
        rows.append(row(rN, fold, "M2 FiLM-MLP-3L", "rolling"))
        rE = run("train_cap.py", f"roll_{fold}_E2", (tr, va, te), CK, "--arch", ARCH_E, "--loss_mask", "extreme", "--epochs", 36, "--patience", 7, "--lr", 0.003, "--wd", 0.0)
        if rN and rE:
            rows.append(row(run("train_moe_f.py", f"roll_{fold}_X2", (tr, va, te), CK, "--normal", f"ckpt:roll_{fold}_A5", "--extreme", f"ckpt:roll_{fold}_E2", "--gate", "refit_mix", "--gate_epochs", 6, "--gate_patience", 3), fold, "X2 MoE", "rolling", ("models", "soft")))
        df = pd.DataFrame([x for x in rows if x]); df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame([x for x in rows if x])
    if len(df):
        agg = df.groupby(["model", "kind"])[["test_wMAPE", "MAE_E", "top1", "recall", "F1"]].mean().round(3); agg.to_csv(OUT / "summary_mean.csv", encoding="utf-8-sig")
        print(df[["fold", "model", "kind", "test_wMAPE", "MAE_E", "top1", "recall", "F1", "epochs", "minutes"]].round(3).to_string(), flush=True); print(agg.to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
