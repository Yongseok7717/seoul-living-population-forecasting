"""Propagation diagnostic: do neighbours' residuals help to predict a cell's future residual once its own history is known?

For each relation P (row-normalised; rook adjacency, functional kNN, lag-correlation <= 3 km, subway connectivity) and horizon h:
  contemporaneous similarity   corr(z_i(t), (Pz)_i(t))
  own model      z_i(t+h) ~ z_i(t), z_i(t-1), z_i(t-2), z_i(t+h-24)
  full model     own + (Pz)_i at the same four lags
  dR2 = R2(full) - R2(own)   (pooled OLS over all analysis cells and hours; fitted on 2023, evaluated on 2025 as well)
Also the share of residual energy that is smooth on the graph:  1 - ||z - Pz||^2 / ||z||^2.
Outputs results/diag_propagation/summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "diag_propagation"; OUT.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(SRC))
from train_moe import MoEData, prop_matrix   # noqa: E402

RELS = ["geo", "func_knn", "mob_lag_local", "trans"]; HS = [1, 6, 24]


def design(z, m, h):
    """rows = (t, i) with all lags valid; returns own features, neighbour features, target (flattened over cells)"""
    T = z.shape[0]; t0 = max(2, 24 - h); ts = np.arange(t0, T - h)
    lag = lambda a: [a[ts], a[ts - 1], a[ts - 2], a[ts + h - 24]]
    own = np.stack(lag(z), -1); nb = np.stack(lag(m), -1); y = z[ts + h]
    ok = ~(np.isnan(own).any(-1) | np.isnan(nb).any(-1) | np.isnan(y))
    return own[ok], nb[ok], y[ok]


def r2(X, y, beta=None):
    X1 = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
    if beta is None:
        beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta; return 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum()), beta


def main():
    md = MoEData(72, 24, "q95"); d = md.d; rows = []
    Z = {s: np.clip(d.RP[s] / d.r_std, -20, 20).astype(np.float64) for s in ("train", "test")}   # RP = residual in people (d.R is the relative residual)
    for rel in RELS:
        P = prop_matrix(d, rel).tocsr(); deg = np.asarray((P != 0).sum(1)).ravel(); has = deg > 0
        M = {}
        for s in Z:
            z0 = np.nan_to_num(Z[s]); w = (~np.isnan(Z[s])).astype(np.float64); num = (P @ z0.T).T; den = (P @ w.T).T
            m = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan); m[:, ~has] = np.nan; M[s] = m
        for s in Z:
            z, m = Z[s], M[s]; ok = ~(np.isnan(z) | np.isnan(m)); c0 = float(np.corrcoef(z[ok], m[ok])[0, 1]); smooth = 1 - float(((z[ok] - m[ok]) ** 2).sum() / (z[ok] ** 2).sum())
            rows.append({"relation": rel, "split": s, "h": 0, "cells_with_neighbours": int(has.sum()), "mean_degree": float(deg[has].mean()), "corr_contemporaneous": c0, "smooth_energy_share": smooth})
        for h in HS:
            o, n, y = design(Z["train"], M["train"], h); r_own, b_own = r2(o, y); r_full, b_full = r2(np.concatenate([o, n], 1), y)
            o2, n2, y2 = design(Z["test"], M["test"], h); t_own, _ = r2(o2, y2, b_own); t_full, _ = r2(np.concatenate([o2, n2], 1), y2, b_full)
            rows.append({"relation": rel, "split": "fit 2023 / test 2025", "h": h, "n_train": len(y), "R2_own_train": r_own, "R2_full_train": r_full, "dR2_train": r_full - r_own, "R2_own_test": t_own, "R2_full_test": t_full, "dR2_test": t_full - t_own})
            print(rel, h, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string()); print("ALLDONE")


if __name__ == "__main__":
    main()
