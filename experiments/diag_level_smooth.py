"""Smoothness of the population LEVEL versus the RESIDUAL on the four relations.

For each relation P (row-normalised) and field F in {X (level), log(1+X), B (causal baseline), R = X - B (people), z = clip(R/s, +-20)}:
  smooth share = 1 - ||F - PF||^2 / ||F||^2     (F centred over time per cell for X, log X and B, so that the constant
  cell mean does not inflate the share; the uncentred value is reported as well)
  contemporaneous correlation corr(F_i(t), (PF)_i(t)).
Argument: the graph-exploitable structure lives in the level, which the causal baseline already removes.
Outputs results/diag_propagation/level_smooth.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "diag_propagation"; OUT.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(SRC))
from train_moe import MoEData, prop_matrix   # noqa: E402

RELS = ["geo", "func_knn", "mob_lag_local", "trans"]


def stats(F, P, has):
    F = F.astype(np.float64); w = (~np.isnan(F)).astype(np.float64); F0 = np.nan_to_num(F)
    num = (P @ F0.T).T; den = (P @ w.T).T; M = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan); M[:, ~has] = np.nan
    ok = ~(np.isnan(F) | np.isnan(M)); f, m = F[ok], M[ok]
    return float(np.corrcoef(f, m)[0, 1]), 1 - float(((f - m) ** 2).sum() / (f ** 2).sum())


def main():
    md = MoEData(72, 24, "q95"); d = md.d; rows = []
    for s in ("train", "test"):
        X = d.X[s].astype(np.float64); B = d.B[s].astype(np.float64); R = X - B; z = np.clip(R / d.r_std, -20, 20)
        fields = {"X level": X, "log(1+X)": np.log1p(np.maximum(X, 0)), "B baseline": B, "R residual (people)": R, "z normalized residual": z}
        for rel in RELS:
            P = prop_matrix(d, rel).tocsr(); deg = np.asarray((P != 0).sum(1)).ravel(); has = deg > 0
            for name, F in fields.items():
                c, sm = stats(F, P, has)
                Fc = F - np.nanmean(F, 0, keepdims=True); cc, smc = stats(Fc, P, has)
                rows.append({"split": s, "relation": rel, "field": name, "corr": c, "smooth_share": sm, "corr_centered": cc, "smooth_share_centered": smc})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "level_smooth.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows); pd.set_option("display.width", 200); print(df.round(3).to_string()); print("ALLDONE")


if __name__ == "__main__":
    main()
