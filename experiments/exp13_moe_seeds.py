"""Experiment 13 - three seeds for the extreme-aware mixtures X1 (class-weighted gate + Platt) and X2 (mixture-tuned gate).

Seed 0 = the runs of exp04 (event_def q95: exp04_q95_bce / exp04_q95_mix, experts t16c_B_A5_lr0.003_wd0 and t16c_E_E2_long).
Seeds 1, 2: normal expert = M2 seed s (t16c_S_A5_s{s}, the seeds of Table 2), extreme expert retrained with seed s
(mlp:256,256,128,64, masked extreme Huber), gate refitted with seed s (wBCE + Platt -> X1; mixture-tuned raw p -> X2).
Outputs results/exp13_moe_seeds/{*.json, summary.csv}
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp13_moe_seeds"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
ORIG_CKPT = ROOT / "results" / "base" / "ckpt"; EXP04 = ROOT / "results" / "exp04_event_definition"


def run(script, name, *argv):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(CK), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    cmd = [sys.executable, str(SRC / script), "--name", name] + [str(a) for a in argv]; print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return json.loads(f.read_text(encoding="utf-8"))


def mrow(r, seed, label, which):
    if r is None:
        return None
    m = r["models"][which]["test"]; g = r["gates"]["gate"]
    return {"seed": seed, "model": label, "mixture": which, "gate_test_auprc": g["test"]["auprc"], "test_wMAPE": m["X_wmape"] * 100, "MAE_N": m["MAE_N"], "MAE_E": m["MAE_E"],
            "top1": m["R_rel_err_top1pct"], "recall": m["extreme_recall_top1"], "F1": m["extreme_f1_top1"], "peak_mag": m["peak_magnitude_rel_err"]}


def main():
    rows = []
    r1 = json.loads((EXP04 / "exp04_q95_bce.json").read_text(encoding="utf-8")); r2 = json.loads((EXP04 / "exp04_q95_mix.json").read_text(encoding="utf-8"))
    rows += [mrow(r1, 0, "X1 wBCE gate + Platt", "soft_platt"), mrow(r1, 0, "X3 oracle", "oracle"), mrow(r2, 0, "X2 mixture-tuned gate", "soft")]
    for s in (1, 2):
        srcN = ORIG_CKPT / f"t16c_S_A5_s{s}.pt"; dstN = CK / f"exp13_N_s{s}.pt"
        if not srcN.exists():
            print("FAILED missing normal expert", srcN, flush=True); continue
        if not dstN.exists():
            shutil.copy(srcN, dstN)
        rE = run("train_cap.py", f"exp13_E_s{s}", "--arch", "mlp:256,256,128,64", "--loss_mask", "extreme", "--event_def", "q95", "--epochs", 48, "--patience", 7, "--lr", 0.003, "--wd", 0.0, "--seed", s)
        if rE is None:
            continue
        g1 = run("train_moe_f.py", f"exp13_s{s}_bce", "--event_def", "q95", "--normal", f"ckpt:exp13_N_s{s}", "--extreme", f"ckpt:exp13_E_s{s}", "--gate", "refit_bce", "--gate_epochs", 12, "--gate_patience", 4, "--seed", s)
        g2 = run("train_moe_f.py", f"exp13_s{s}_mix", "--event_def", "q95", "--normal", f"ckpt:exp13_N_s{s}", "--extreme", f"ckpt:exp13_E_s{s}", "--gate", "refit_mix", "--gate_epochs", 6, "--gate_patience", 3, "--seed", s)
        rows += [mrow(g1, s, "X1 wBCE gate + Platt", "soft_platt"), mrow(g1, s, "X3 oracle", "oracle"), mrow(g2, s, "X2 mixture-tuned gate", "soft")]
        pd.DataFrame([x for x in rows if x]).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame([x for x in rows if x]); pd.set_option("display.width", 220); print(df.round(3).to_string(), flush=True)
    print(df.groupby("model")[["test_wMAPE", "MAE_E", "top1", "recall", "F1"]].agg(["mean", "std"]).round(3).to_string(), flush=True); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
