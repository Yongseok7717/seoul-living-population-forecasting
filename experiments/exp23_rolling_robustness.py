"""Experiment 23 - robustness of the rolling gain: (a) fixed vs rolling M2 wMAPE excluding the cells whose 2025 residual exceeds 20 training
scales at some hour (growth cells); (b) paired weekly block bootstrap of the yearly fixed-minus-rolling wMAPE difference.
Forecasts: six-hourly test origins of each 2025 quarter (train_cap eval dumps). Outputs results/exp23_rolling_robustness/summary.csv
"""
import os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp23_rolling_robustness"; LOGS = OUT / "logs"
for d_ in (OUT, LOGS):
    d_.mkdir(parents=True, exist_ok=True)
ORIG_CKPT = ROOT / "results" / "base" / "ckpt"; ROLL_CKPT = ROOT / "results" / "exp01_rolling" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}


def dump(name, splits, ckpt_dir, init):
    f = OUT / f"{name}_pred_test.npz"
    if f.exists():
        return f
    env = dict(os.environ, RD_SPLITS=",".join(splits), LP_RESULTS=str(OUT), LP_CKPT=str(ckpt_dir), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name, "--arch", "mlp:512,256,128", "--mode", "eval", "--init", init, "--eval_stride", "6", "--dump_pred"]
    print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    return f if rc == 0 and f.exists() else None


def growth_cells():
    sys.path.insert(0, str(SRC))
    from train_moe import MoEData
    d = MoEData(72, 24, "q95").d; z = np.abs(d.RP["test"]) / d.r_std[None, :]; g = np.nanmax(z, 0) > 20
    return d.cells, g


def main():
    cells, g = growth_cells(); print("growth cells", int(g.sum()), "of", len(cells), flush=True)
    errs = {"fixed": [], "rolling": []}; rows = []
    for q, fold in FOLDS.items():
        fixed = ("202303-202312", "202411-202412", fold[2])
        for lab, f in (("fixed", dump(f"r_{q}_fixed", fixed, ORIG_CKPT, "t16c_B_A5_lr0.003_wd0")), ("rolling", dump(f"r_{q}_roll", fold, ROLL_CKPT, f"roll_{q}_A5"))):
            z = np.load(f); X, Xh = z["X"], np.maximum(z["Xhat"], 0); assert (z["cells"] == cells).all()
            tm = pd.to_datetime(z["time"][:, 0].astype(str)); ae = np.abs(Xh - X)
            keep = ~g
            rows.append({"quarter": q, "model": lab, "wMAPE_all": 100 * ae.sum() / X.sum(), "wMAPE_excl_growth": 100 * ae[:, :, keep].sum() / X[:, :, keep].sum(), "wMAPE_growth_cells": 100 * ae[:, :, g].sum() / X[:, :, g].sum(), "share_pop_growth": X[:, :, g].sum() / X.sum()})
            errs[lab].append(pd.DataFrame({"week": tm.isocalendar().week.values, "ae": ae.sum((1, 2)), "ae_ng": ae[:, :, keep].sum((1, 2)), "x": X.sum((1, 2)), "x_ng": X[:, :, keep].sum((1, 2))}))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "by_quarter.csv", index=False, encoding="utf-8-sig")
    F = pd.concat(errs["fixed"]).reset_index(drop=True); R = pd.concat(errs["rolling"]).reset_index(drop=True); assert len(F) == len(R)
    year = {"fixed_all": 100 * F.ae.sum() / F.x.sum(), "rolling_all": 100 * R.ae.sum() / R.x.sum(), "fixed_excl": 100 * F.ae_ng.sum() / F.x_ng.sum(), "rolling_excl": 100 * R.ae_ng.sum() / R.x_ng.sum()}
    # paired weekly block bootstrap of the difference fixed - rolling
    rng = np.random.default_rng(0); wk = F.week.values; weeks = np.unique(wk); gF = F.groupby(wk)[["ae", "x", "ae_ng", "x_ng"]].sum(); gR = R.groupby(wk)[["ae", "x", "ae_ng", "x_ng"]].sum()
    diffs, diffs_ng = [], []
    for _ in range(2000):
        s = rng.choice(weeks, len(weeks), replace=True); a, b = gF.loc[s].sum(), gR.loc[s].sum()
        diffs.append(100 * (a.ae / a.x - b.ae / b.x)); diffs_ng.append(100 * (a.ae_ng / a.x_ng - b.ae_ng / b.x_ng))
    lo, hi = np.percentile(diffs, [2.5, 97.5]); lo2, hi2 = np.percentile(diffs_ng, [2.5, 97.5])
    summ = {**year, "gain_all": year["fixed_all"] - year["rolling_all"], "gain_all_ci95": (lo, hi), "gain_excl": year["fixed_excl"] - year["rolling_excl"], "gain_excl_ci95": (lo2, hi2), "n_growth_cells": int(g.sum()), "weeks": len(weeks)}
    pd.DataFrame([summ]).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig"); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
