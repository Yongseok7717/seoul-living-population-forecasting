"""Experiment 3 - relational branches on the final encoder (negative control on the paper's main model).

FiLM-MLP-3L (M2) + one gated propagation branch h <- h + sigma(Wg[h; m~]) * m~, m~ = MLP(P h), with P row-stochastic:
  geo (rook adjacency) | func_knn (functional kNN10, Task 3) | mob_lag_local (lag-correlation, <= 3 km) | trans (subway connectivity)
plus the same model without a branch trained with the same full-node batches (fair baseline; the sampled-node M2 differs only in batching).
Seed 0, AdamW 3e-3 / wd 0, OneCycle <= 30 epochs, patience 5 (+ annealing).  Outputs results/exp03_relations/.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp03_relations"; CK = OUT / "ckpt"; LOGS = OUT / "logs"
for d in (OUT, CK, LOGS):
    d.mkdir(parents=True, exist_ok=True)
RELS = ["none", "geo", "func_knn", "mob_lag_local", "trans"]


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
    for seed in (0, 1, 2):                                                         # seed 0 = original run names (exp03_<rel>), seeds 1-2 added 2026-09-19
        for rel in RELS:
            extra = ["--rel", rel] if rel != "none" else ["--node_sample", 7807]
            r = run(f"exp03_{rel}" + ("" if seed == 0 else f"_s{seed}"), "--arch", "mlp:512,256,128", *extra, "--epochs", 30, "--patience", 5, "--lr", 0.003, "--wd", 0.0, "--seed", seed)
            if r:
                m = r["models"]["model"]; t = m["test"]
                rows.append({"relation": rel, "seed": seed, "params": r["params"], "val_wMAPE": m["val"]["X_wmape"] * 100, "test_wMAPE": t["X_wmape"] * 100, "MAE_E": t["MAE_E"], "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "jaccard": t["top5pct_spatial_jaccard"], "best_epoch": r.get("best_epoch"), "minutes": r.get("train_minutes")})
            pd.DataFrame(rows).to_csv(OUT / "summary_seeds.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows); df[df.seed == 0].drop(columns="seed").to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    ms = df.groupby("relation")[["val_wMAPE", "test_wMAPE", "MAE_E", "top1", "recall", "F1"]].agg(["mean", "std"]).reindex(RELS); ms.to_csv(OUT / "summary_mean.csv", encoding="utf-8-sig")
    print(ms.round(3).to_string(), flush=True); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
