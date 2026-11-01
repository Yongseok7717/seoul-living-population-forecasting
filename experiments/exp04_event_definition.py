"""Experiment 4 (T2) - refined event definition for the extreme-aware MoE.

Event labels: q95 (|R| > Q95 of the training residual, per cell-hour; the definition used so far), q95c (q95 AND clustered
(>= 3 q95 cells in the rook neighbourhood at the same hour) OR persistent (q95 in the previous or next hour)), q95k (clustered only), q95s (clustered AND
persistent).  For each definition: extreme expert (masked extreme Huber, MLP 256-256-128-64), gate retrained from scratch on the
new labels (wBCE, Platt) and the mixture-tuned gate; normal expert = M2 (t16c_B_A5_lr0.003_wd0).  Reports event rate, gate PR-AUC,
oracle upper bound and the mixture metrics (which are defined on the top-1 % residual and therefore comparable across definitions).
Outputs results/exp04_event_definition/.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp04_event_definition"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ORIG_CKPT = ROOT / "results" / "base" / "ckpt"; ORIG_RES = ROOT / "results" / "base" / "results"
DEFS = ["q95", "q95c", "q95s"]


def run(script, name, ckpt_dir, *argv):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(ckpt_dir), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    cmd = [sys.executable, str(SRC / script), "--name", name] + [str(a) for a in argv]; print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def mrow(r, d, label, which):
    if r is None:
        return None
    m = r["models"][which]["test"]; g = r["gates"]["gate"]
    return {"event_def": d, "model": label, "mixture": which, "event_rate_test": r["event_rate"]["test"], "gate_val_auprc": g["val"]["auprc"], "gate_test_auprc": g["test"]["auprc"], "test_wMAPE": m["X_wmape"] * 100, "MAE_N": m["MAE_N"], "MAE_E": m["MAE_E"],
            "top1": m["R_rel_err_top1pct"], "recall": m["extreme_recall_top1"], "F1": m["extreme_f1_top1"], "peak_mag": m["peak_magnitude_rel_err"]}


def main():
    rows = []
    for d in DEFS:
        # extreme expert on this definition (q95 reuses the Stage F expert checkpoint copied into the experiment ckpt dir)
        if d == "q95":
            src = ORIG_CKPT / "t16c_E_E2_long.pt"; dst = CK / "exp04_E_q95.pt"
            if src.exists() and not dst.exists():
                import shutil; shutil.copy(src, dst)
            eN = "exp04_E_q95"
        else:
            rE = run("train_cap.py", f"exp04_E_{d}", CK, "--arch", "mlp:256,256,128,64", "--loss_mask", "extreme", "--event_def", d, "--epochs", 48, "--patience", 7, "--lr", 0.003, "--wd", 0.0); eN = f"exp04_E_{d}"
            if rE is None:
                continue
        # normal expert checkpoint alongside (copy once so LP_CKPT has both)
        srcN = ORIG_CKPT / "t16c_B_A5_lr0.003_wd0.pt"; dstN = CK / "exp04_N_A5.pt"
        if srcN.exists() and not dstN.exists():
            import shutil; shutil.copy(srcN, dstN)
        r1 = run("train_moe_f.py", f"exp04_{d}_bce", CK, "--event_def", d, "--normal", "ckpt:exp04_N_A5", "--extreme", f"ckpt:{eN}", "--gate", "refit_bce", "--gate_epochs", 12, "--gate_patience", 4)
        r2 = run("train_moe_f.py", f"exp04_{d}_mix", CK, "--event_def", d, "--normal", "ckpt:exp04_N_A5", "--extreme", f"ckpt:{eN}", "--gate", "refit_mix", "--gate_epochs", 6, "--gate_patience", 3)
        rows += [mrow(r1, d, "wBCE gate + Platt", "soft_platt"), mrow(r1, d, "oracle", "oracle"), mrow(r2, d, "mixture-tuned gate, raw p", "soft")]
        pd.DataFrame([x for x in rows if x]).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame([x for x in rows if x]); print(df.round(3).to_string(), flush=True); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
