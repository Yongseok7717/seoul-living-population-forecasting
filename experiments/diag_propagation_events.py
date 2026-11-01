"""Propagation diagnostic restricted to stress conditions: do neighbours help predict a cell's residual DURING events?

Same regressions as diag_propagation.py (own lags vs own + neighbourhood lags, fitted on 2023, evaluated on 2025), but the target
rows are restricted to (a) hours inside emergency-alert windows (weather channel of prior_dmsg: heavy rain, heat, cold, snow),
(b) q95 event cell-hours, (c) cell-hours of cells that contain a shelter or a health facility during alert windows.
Also the contemporaneous correlation / smooth share on the same subsets.  Outputs results/diag_propagation/events.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "diag_propagation"; sys.path.insert(0, str(SRC))
from train_moe import MoEData, prop_matrix   # noqa: E402
import config as C                           # noqa: E402
from loader import months_between            # noqa: E402
sys.path.insert(0, str(ROOT / "experiments")); from diag_propagation import design, r2, RELS   # noqa: E402


def main():
    md = MoEData(72, 24, "q95"); d = md.d; rows = []
    Z = {s: np.clip(d.RP[s] / d.r_std, -20, 20).astype(np.float32) for s in ("train", "test")}   # RP = residual in people (d.R is the relative residual)
    E = {s: md.E[s].astype(bool) for s in ("train", "test")}
    # weather-alert windows from the disaster-message prior (channel 1 = weather); available from 2023-09
    months = {"train": months_between("202303", "202312"), "test": months_between("202501", "202512")}
    W = {}
    for s, ms in months.items():
        parts = []
        for m in ms:
            f = C.OUT / "priors" / f"prior_dmsg_{m}.npy"
            parts.append(np.load(f)[:, d.cells, 1] > 0 if f.exists() else np.zeros((len(pd.read_parquet(C.OUT_LOCAL / f"time_index_{m}.parquet")), d.N), bool))
        W[s] = np.concatenate(parts); assert W[s].shape == Z[s].shape, (W[s].shape, Z[s].shape)
    # facility cells (any shelter or health facility)
    fc = pd.read_parquet(C.OUT / "facilities" / "facility_counts_by_cell.parquet").set_index("cell_idx").reindex(d.cells).fillna(0)
    fcols = [c for c in fc.columns if any(k in c for k in ("shelter", "hospital", "clinic", "emergency"))]; fac = (fc[fcols].sum(1) > 0).values
    print("alert cell-hour share train/test:", W["train"].mean().round(4), W["test"].mean().round(4), "| facility cells:", int(fac.sum()), flush=True)
    for rel in RELS:
        P = prop_matrix(d, rel).tocsr(); deg = np.asarray((P != 0).sum(1)).ravel(); has = deg > 0; M = {}
        for s in Z:
            z0 = np.nan_to_num(Z[s]); w = (~np.isnan(Z[s])).astype(np.float32); num = (P @ z0.T).T; den = (P @ w.T).T
            m = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32); m[:, ~has] = np.nan; M[s] = m; del z0, w, num, den
        for h in (1, 24):
            o, n, y = design(Z["train"], M["train"], h); r_own, b_own = r2(o, y); r_full, b_full = r2(np.concatenate([o, n], 1), y)
            # masks aligned with design(): rows are (t, i) with t = t0..T-h-1 and target at t+h
            T = Z["test"].shape[0]; t0 = max(2, 24 - h); ts = np.arange(t0, T - h)
            lagZ = lambda k_: Z["test"][ts + k_]; lagM = lambda k_: M["test"][ts + k_]; yy = Z["test"][ts + h]
            ok = ~(np.isnan(yy) | np.isnan(lagZ(0)) | np.isnan(lagZ(-2)) | np.isnan(lagZ(h - 24)) | np.isnan(lagM(0)) | np.isnan(lagM(-2)) | np.isnan(lagM(h - 24)) | np.isnan(lagZ(-1)) | np.isnan(lagM(-1)))
            own = nb = None
            subsets = {"all": np.ones_like(ok), "alert (weather) hours": W["test"][ts + h], "q95 event cell-hours": E["test"][ts + h], "facility cells in alert hours": W["test"][ts + h] & fac[None, :]}
            for name, msk in subsets.items():
                mm = (ok & msk); o3 = np.stack([lagZ(0)[mm], lagZ(-1)[mm], lagZ(-2)[mm], lagZ(h - 24)[mm]], -1); n3 = np.stack([lagM(0)[mm], lagM(-1)[mm], lagM(-2)[mm], lagM(h - 24)[mm]], -1); y3 = yy[mm]
                if len(y3) < 1000:
                    continue
                t_own, _ = r2(o3, y3, b_own); t_full, _ = r2(np.concatenate([o3, n3], 1), y3, b_full)
                zc, mc = Z["test"][ts + h][mm], M["test"][ts + h][mm]; c0 = float(np.corrcoef(zc, mc)[0, 1]); smooth = 1 - float(((zc - mc) ** 2).sum() / (zc ** 2).sum())
                rows.append({"relation": rel, "subset": name, "h": h, "n": int(len(y3)), "corr_contemporaneous": c0, "smooth_energy_share": smooth, "R2_own_test": t_own, "dR2_test": t_full - t_own})
                print(rel, name, h, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string()); print("ALLDONE")


if __name__ == "__main__":
    main()
