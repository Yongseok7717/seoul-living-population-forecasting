"""Task 4.1: baseline-residual decomposition X_t = B_t + R_t with a causal rolling baseline, and the dataset for
residual forecasting with relational-structure ablations.

Baseline B_{i,t}: median of X_i over the previous WINDOW_WEEKS weeks of observations with the same hour and daytype
(weekday / saturday / sunday-holiday). It is causal (uses only past data), so it follows the 2025 level drift.
Residuals: R = X - B (people) and r = (X - B) / (B + EPS) (relative). The model works on the relative residual
clipped to [-1, R_CLIP]; metrics are computed on both.

    python residual_data.py            # builds processed/residual/{baseline,resid}_YYYYMM.npy (pop10, all cells)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                                     # noqa: E402
from loader import load_pop10, load_grid, months_between, anomalous_days   # noqa: E402
from build_baseline_atlas import daytype_of, HOLIDAY_SET               # noqa: E402
from data import ML_DIR                                                # noqa: E402

RES_DIR = C.OUT / "residual"
REL = C.OUT / "relations"
EDA = C.OUT / "eda"
WINDOW_WEEKS = 8
EPS = 20.0
R_CLIP = 10.0
ALL_MONTHS = months_between("202301", "202601")
SPLITS = {"train": ("202303", "202312"), "val": ("202401", "202412"), "test": ("202501", "202512")}
import os as _os
if _os.environ.get("RD_SPLITS"):                                   # e.g. RD_SPLITS=202303-202410,202411-202412,202501-202512  (train,val,test) for year-generalisation runs
    _p = [x.split("-") for x in _os.environ["RD_SPLITS"].split(",")]; SPLITS = {"train": tuple(_p[0]), "val": tuple(_p[1]), "test": tuple(_p[2])}
    print("RD_SPLITS override:", SPLITS, flush=True)


def calendar(ti):
    dt = daytype_of(ti)
    h = ti.tt.values.astype(float); dow = ti.datetime.dt.dayofweek.values.astype(float); month = ti.datetime.dt.month.values.astype(float)
    f = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                  (dow >= 5).astype(float), ti.ymd.isin(HOLIDAY_SET).values.astype(float),
                  np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                  (dt == 0).astype(float), (dt == 1).astype(float), (dt == 2).astype(float)], axis=1).astype(np.float32)
    return f, dt


def build_baseline():
    """Causal rolling-median baseline for pop10 on all grid cells, saved per month."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RES_DIR.mkdir(parents=True, exist_ok=True)
    X, ti = load_pop10(ALL_MONTHS)
    X = X.astype(np.float32).copy()
    bad = (~ti.available.values) | ti.ymd.isin(anomalous_days()).values
    X[bad] = np.nan
    T, N = X.shape
    _, dt = calendar(ti)
    tt = ti.tt.values
    B = np.full((T, N), np.nan, np.float32)
    occ_per_week = {0: 5, 1: 1, 2: 1}
    for d in range(3):
        w = WINDOW_WEEKS * occ_per_week[d]
        for h in range(24):
            idx = np.where((dt == d) & (tt == h))[0]
            if len(idx) <= w:
                continue
            V = torch.tensor(X[idx], device=dev)                       # [n_occ, N] with NaN for bad hours
            # rolling median over the previous w occurrences (exclusive of the current one), NaN-aware
            out = torch.full_like(V, float("nan"))
            for j in range(w, len(idx)):
                win = V[j - w:j]
                out[j] = torch.nanmedian(win, dim=0).values
            B[idx] = out.cpu().numpy()
        print(f"daytype {d} done", flush=True)
    # per month files
    for ym in ALL_MONTHS:
        m = (ti.ymd.str[:6] == ym).values
        np.save(RES_DIR / f"baseline_{ym}.npy", B[m])
        np.save(RES_DIR / f"resid_{ym}.npy", (X[m] - B[m]).astype(np.float32))
    meta = {"window_weeks": WINDOW_WEEKS, "conditioning": "hour x daytype(weekday/saturday/sunday_holiday)", "eps": EPS,
            "first_valid": str(ti.datetime[~np.isnan(B[:, 0])].min()), "note": "causal: median of the previous window occurrences"}
    (RES_DIR / "baseline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ok = ~np.isnan(B[:, 0]) & ~np.isnan(X[:, 0])
    tot_err = np.abs(X[ok] - B[ok]).sum() / X[ok].sum()
    print("baseline wMAPE (all valid hours):", round(float(tot_err), 4), "first valid", meta["first_valid"])


def load_sparse(path, key):
    z = np.load(path, allow_pickle=True)
    return sp.csr_matrix((z[f"{key}__data"], z[f"{key}__indices"], z[f"{key}__indptr"]), shape=tuple(z[f"{key}__shape"]))


class ResidualData:
    """Windows of relative residual history -> future residual, on the populated-cell subset used in Task 2/3."""
    def __init__(self, k=24, horizon=24, source="X"):
        """source: 'X' = observed pop10 with the rolling-median baseline; 'L' = causal regular component (build_rolling_L.py) with its own baseline (F2)"""
        self.k, self.H = k, horizon; self.source = source
        grid = load_grid(with_geometry=False)
        lab = pd.read_parquet(EDA / "cluster_labels.parquet")
        self.cells = lab.cell_idx.values; self.N = len(self.cells)
        self.cluster = lab.c_kmeans_norm.values
        self.grid = grid
        anom = anomalous_days()
        self.X, self.B, self.R, self.ti, self.tf, self.dt = {}, {}, {}, {}, {}, {}
        for split, (a, b) in SPLITS.items():
            months = months_between(a, b)
            X, ti = load_pop10(months)
            if source == "L":
                X = np.concatenate([np.load(RES_DIR / f"rollL_{ym}.npy") for ym in months]); Bm = np.concatenate([np.load(RES_DIR / f"baseL_{ym}.npy") for ym in months])
            else:
                Bm = np.concatenate([np.load(RES_DIR / f"baseline_{ym}.npy") for ym in months])
            X = X.astype(np.float32)[:, self.cells].copy(); Bm = Bm[:, self.cells]
            bad = (~ti.available.values) | ti.ymd.isin(anom).values | np.isnan(Bm[:, 0]) | np.isnan(X[:, 0])
            X[bad] = np.nan; Bm[bad] = np.nan
            r = np.clip((X - Bm) / (Bm + EPS), -1, R_CLIP).astype(np.float32)
            self.X[split], self.B[split], self.R[split], self.ti[split] = X, Bm, r, ti
            self.tf[split], self.dt[split] = calendar(ti)
        # target scale: residual in PEOPLE divided by a per-cell scale s_i = std of the training residual (floor 5 people).
        # (a relative residual divided by its std explodes for tiny cells and lets them dominate the loss)
        tr_people = self.X["train"] - self.B["train"]; ok = ~np.isnan(tr_people[:, 0])
        self.r_std = np.maximum(np.nanstd(tr_people[ok], axis=0), 5.0).astype(np.float32)
        self.RP = {s: (self.X[s] - self.B[s]).astype(np.float32) for s in self.X}
        # log-baseline as an input channel (scale information), standardised per cell
        lb = np.log1p(self.B["train"][ok]); self.lb_mu = lb.mean(0).astype(np.float32); self.lb_sd = np.maximum(lb.std(0), 0.05).astype(np.float32)
        # static features: facility counts + cluster one-hot + Vs30
        fc = pd.read_parquet(C.OUT / "facilities" / "facility_counts_by_cell.parquet").set_index("cell_idx")
        feats = [np.log1p(fc[c].reindex(self.cells).fillna(0).values) for c in fc.columns if c.startswith("n_")]
        feats += [(self.cluster == c).astype(float) for c in range(int(self.cluster.max()) + 1)]
        vs = grid.Vs30.reindex(self.cells).values.astype(float); vs = np.where(np.isnan(vs), np.nanmedian(vs), vs); feats.append(vs)
        S = np.stack(feats, 1).astype(np.float32); self.static = (S - S.mean(0)) / (S.std(0) + 1e-6)

    # ---- relational operators on the subset (propagation matrices, row-stochastic-ish) ----
    def operator(self, name):
        """Return a dict describing the operator: {'kind': 'dense_sparse', 'P': csr} or hypergraph {'kind': 'hyper', 'H': csr, 'w': array}
        or dynamic {'kind': 'dynamic', 'P_by_daytype': [P_wd, P_we]}."""
        c = self.cells
        def norm_adj(W):
            W = W[c][:, c]; W = W + sp.eye(len(c), format="csr") * 1e-6
            d = np.asarray(W.sum(1)).ravel(); dis = 1 / np.sqrt(np.maximum(d, 1e-9))
            return (sp.diags(dis) @ W @ sp.diags(dis)).tocsr().astype(np.float32)
        if name == "spatial":
            return {"kind": "sparse", "P": norm_adj(load_sparse(REL / "spatial_rook.npz", "W"))}
        if name == "func_knn":
            return {"kind": "sparse", "P": norm_adj(load_sparse(REL / "func_similarity_knn10.npz", "W"))}
        if name in ("func_knn_L", "func_knn_X"):                       # R7: kNN graphs from the RPCA regular component L (or X, same recipe)
            return {"kind": "sparse", "P": norm_adj(load_sparse(REL / f"func_similarity_knn10_{name[-1]}.npz", "W"))}
        if name in ("func_hard", "func_soft"):
            f = "func_hard_k6.npz" if name == "func_hard" else "func_soft_k6.npz"
            H = load_sparse(REL / f, "H")[c]; return {"kind": "hyper", "H": H.tocsr().astype(np.float32), "w": np.load(REL / f)["w"]}
        if name == "func_dynamic":
            Hs = [load_sparse(REL / f, "H")[c].tocsr().astype(np.float32) for f in ("func_weekday_k6.npz", "func_weekend_k6.npz")]
            return {"kind": "hyper_dynamic", "H": Hs, "w": [np.ones(h.shape[1], np.float32) for h in Hs]}
        if name == "facility":
            z = np.load(REL / "fac_combined.npz", allow_pickle=True)
            return {"kind": "hyper", "H": load_sparse(REL / "fac_combined.npz", "H_hard")[c].tocsr().astype(np.float32), "w": z["w"]}
        raise ValueError(name)

    def use_raw_target(self):
        """Task 6.4 raw target: no baseline decomposition. The 'baseline' becomes the constant training mean per cell,
        so the same code path yields X_hat = mean_i + s_i * z and the history channel is X - mean_i."""
        m = np.nanmean(self.X["train"], axis=0).astype(np.float32)
        for s in self.B:
            self.B[s] = np.where(np.isnan(self.B[s]), np.nan, m[None, :]).astype(np.float32)
            self.RP[s] = (self.X[s] - self.B[s]).astype(np.float32)
        ok = ~np.isnan(self.RP["train"][:, 0])
        self.r_std = np.maximum(np.nanstd(self.RP["train"][ok], axis=0), 5.0).astype(np.float32)
        self.raw_mode = True

    # ---- windows ----
    def windows(self, split, stride=1):
        ok = ~np.isnan(self.R[split][:, 0]); T = len(ok)
        csum = np.concatenate([[0], np.cumsum(ok)])
        starts = np.arange(self.k - 1, T - self.H, stride)
        good = csum[starts + self.H + 1] - csum[starts - self.k + 1] == self.k + self.H
        return starts[good]

    def batch(self, split, ts):
        """x_hist [B,k,N,2] (r/std, log-baseline z), tf_hist [B,k,11], tf_fut [B,H,11], b_fut [B,H,N] (log-baseline z), y [B,H,N] (r/std)"""
        R, Bm, tf = self.RP[split], self.B[split], self.tf[split]
        rz = lambda t0, t1: np.clip(R[t0:t1] / self.r_std, -20, 20)
        bz = lambda t0, t1: (np.log1p(Bm[t0:t1]) - self.lb_mu) / self.lb_sd
        xh = np.stack([np.stack([rz(t - self.k + 1, t + 1), bz(t - self.k + 1, t + 1)], -1) for t in ts]).astype(np.float32)
        y = np.stack([rz(t + 1, t + 1 + self.H) for t in ts]).astype(np.float32)
        th = np.stack([tf[t - self.k + 1:t + 1] for t in ts]); tfu = np.stack([tf[t + 1:t + 1 + self.H] for t in ts])
        bf = np.stack([bz(t + 1, t + 1 + self.H) for t in ts]).astype(np.float32)
        dtf = np.stack([self.dt[split][t + 1:t + 1 + self.H] for t in ts])
        return xh, th, tfu, bf, y, dtf

    def people(self, split, ts, r_pred_z):
        """relative residual (normalised) -> (X_hat, R_hat, X, R, B) in people for the window targets"""
        Bm, X = self.B[split], self.X[split]
        Bt = np.stack([Bm[t + 1:t + 1 + self.H] for t in ts]); Xt = np.stack([X[t + 1:t + 1 + self.H] for t in ts])
        Rhat = r_pred_z * self.r_std; Xhat = Bt + Rhat
        return Xhat, Rhat, Xt, Xt - Bt, Bt

    def summary(self):
        return {"N": int(self.N), "k": self.k, "H": self.H, "windows": {s: int(len(self.windows(s))) for s in self.R},
                "first_valid_by_split": {s: str(self.ti[s].datetime[~np.isnan(self.R[s][:, 0])].min()) for s in self.R}}


if __name__ == "__main__":
    build_baseline()
