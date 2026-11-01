"""F3: local scale sigma_{i,h,d} for the standardized anomaly score z = (X - mu) / (sigma + eps).

mu_{i,h,d} is the causal rolling-median baseline (ml/residual_data.build_baseline: previous 8 weeks, same hour x daytype).
sigma_{i,h,d} is the matching causal rolling MAD (1.4826 * median |X - median| over the same window), saved per month as
processed/residual/scale_YYYYMM.npy (pop10, all cells). eps = sqrt(mu + 1) (Poisson floor) is applied at scoring time.

    python build_anomaly_scale.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys
from pathlib import Path
import numpy as np
import torch
from loader import load_pop10, anomalous_days                          # noqa: E402
from residual_data import RES_DIR, ALL_MONTHS, WINDOW_WEEKS, calendar  # noqa: E402


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, ti = load_pop10(ALL_MONTHS); X = X.astype(np.float32).copy()
    bad = (~ti.available.values) | ti.ymd.isin(anomalous_days()).values
    X[bad] = np.nan
    T, N = X.shape; _, dt = calendar(ti); tt = ti.tt.values
    SC = np.full((T, N), np.nan, np.float32); occ = {0: 5, 1: 1, 2: 1}
    for d in range(3):
        w = WINDOW_WEEKS * occ[d]
        for h in range(24):
            idx = np.where((dt == d) & (tt == h))[0]
            if len(idx) <= w:
                continue
            V = torch.tensor(X[idx], device=dev); out = torch.full_like(V, float("nan"))
            for j in range(w, len(idx)):
                win = V[j - w:j]; med = torch.nanmedian(win, dim=0).values
                out[j] = 1.4826 * torch.nanmedian((win - med).abs(), dim=0).values
            SC[idx] = out.cpu().numpy()
        print(f"daytype {d} done", flush=True)
    for ym in ALL_MONTHS:
        m = (ti.ymd.str[:6] == ym).values
        np.save(RES_DIR / f"scale_{ym}.npy", SC[m])
    ok = ~np.isnan(SC[:, 0])
    print("median sigma (all cells, valid hours):", float(np.nanmedian(SC[ok])), "share sigma==0:", float((SC[ok] == 0).mean()))


if __name__ == "__main__":
    main()
