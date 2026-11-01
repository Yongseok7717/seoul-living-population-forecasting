"""Experiment 5 (T3) - disaster-message windows as gate prior channels.

Reference MoE (Task 14: FiLM-MLP normal + MLP extreme, k = 72, q95, wBCE gate + Platt) with the gate additionally fed the three
disaster-message channels (any / weather / crowd-traffic, cells of the named 자치구, [first message, last + 12 h]) at the target hours.
Experts are reused from the cache; only the gate is trained.  Compared with the same run without the prior (moe_k72_q95_efilm_mlpN).
Outputs results/exp05_dmsg_gate/{exp05_dmsg.json, summary.csv}.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp05_dmsg_gate"; LOGS = OUT / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
ORIG = ROOT / "results" / "base" / "results"


def main():
    name = "exp05_dmsg"; f = OUT / f"{name}.json"
    if not f.exists():
        env = dict(os.environ, LP_RESULTS=str(OUT), LP_CACHE=str(ORIG / "moe_cache"), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
        cmd = [sys.executable, str(SRC / "train_moe.py"), "--name", name, "--k", "72", "--event_def", "q95", "--normal", "mlp", "--extreme", "mlp", "--extreme_rel", "none", "--gate_losses", "wbce", "--expert_cond", "film", "--prior", "dmsg"]
        print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
        with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
        print("rc", rc, flush=True)
    rows = []
    for lab, p in (("reference gate (no prior)", ORIG / "moe_k72_q95_efilm_mlpN.json"), ("gate + disaster-message channels", f)):
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8")); g = r["gates"]["wbce"]; m = r["models"]["soft_wbce_platt"]
            rows.append({"gate": lab, "val_auprc": g["val"]["auprc"], "test_auprc": g["test"]["auprc"], "val_wMAPE": m["val"]["X_wmape"] * 100, "test_wMAPE": m["test"]["X_wmape"] * 100, "test_MAE_E": m["test"]["MAE_E"], "test_top1": m["test"]["R_rel_err_top1pct"], "test_recall": m["test"]["extreme_recall_top1"], "test_F1": m["test"]["extreme_f1_top1"]})
    df = pd.DataFrame(rows); df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig"); print(df.round(4).to_string(), flush=True); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
