"""Experiment 25 - propagation diagnostic with all 24 lags of the neighborhood average (and of the cell itself), h = 1 and 24, pooled OLS
fitted on 2023 and evaluated on 2025 with the 2023 coefficients (normal equations accumulated in time chunks).
Outputs results/exp25_lag24/summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp25_lag24"; OUT.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(SRC))
from train_moe import MoEData, prop_matrix  # noqa: E402

RELS = ["geo", "func_knn", "mob_lag_local", "trans"]; HS = [1, 24]; L = 24


def accumulate(z, m, h, chunk=240):
    """normal equations for y = z[t+h] on [1, z lags 0..L-1, m lags 0..L-1]; returns XtX, Xty, yty, n, and the same for the own-only block"""
    T, N = z.shape; p = 1 + 2 * L; XtX = np.zeros((p, p)); Xty = np.zeros(p); yy = 0.0; ysum = 0.0; n = 0
    for t0 in range(L - 1, T - h, chunk):
        ts = np.arange(t0, min(t0 + chunk, T - h))
        own = np.stack([z[ts - l] for l in range(L)], -1); nb = np.stack([m[ts - l] for l in range(L)], -1); y = z[ts + h]
        Xf = np.concatenate([np.ones(own.shape[:2] + (1,), np.float32), own, nb], -1).reshape(-1, p); yf = y.reshape(-1)
        ok = ~(np.isnan(Xf).any(1) | np.isnan(yf)); Xf = Xf[ok].astype(np.float64); yf = yf[ok].astype(np.float64)
        XtX += Xf.T @ Xf; Xty += Xf.T @ yf; yy += float(yf @ yf); ysum += float(yf.sum()); n += len(yf)
    return XtX, Xty, yy, ysum, n


def solve_r2(XtX, Xty, yy, ysum, n, idx, beta=None):
    A = XtX[np.ix_(idx, idx)]; b = Xty[idx]
    if beta is None:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    rss = yy - 2 * beta @ b + beta @ A @ beta; tss = yy - ysum ** 2 / n
    return 1 - rss / tss, beta


def main():
    md = MoEData(72, 24, "q95"); d = md.d; rows = []
    Z = {s: np.clip(d.RP[s] / d.r_std, -20, 20).astype(np.float32) for s in ("train", "test")}
    for rel in RELS:
        P = prop_matrix(d, rel).tocsr(); deg = np.asarray((P != 0).sum(1)).ravel(); has = deg > 0; M = {}
        for s in Z:
            z0 = np.nan_to_num(Z[s]); w = (~np.isnan(Z[s])).astype(np.float32); num = (P @ z0.T).T; den = (P @ w.T).T
            m = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32); m[:, ~has] = np.nan; M[s] = m
        own_idx = list(range(0, 1 + L)); full_idx = list(range(1 + 2 * L))
        for h in HS:
            tr = accumulate(Z["train"], M["train"], h); te = accumulate(Z["test"], M["test"], h)
            r_own, b_own = solve_r2(*tr, own_idx); r_full, b_full = solve_r2(*tr, full_idx)
            t_own, _ = solve_r2(*te, own_idx, b_own); t_full, _ = solve_r2(*te, full_idx, b_full)
            rows.append({"relation": rel, "h": h, "lags": L, "n_train": tr[4], "R2_own_train": r_own, "R2_full_train": r_full, "dR2_train": r_full - r_own, "R2_own_test": t_own, "R2_full_test": t_full, "dR2_test": t_full - t_own})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
