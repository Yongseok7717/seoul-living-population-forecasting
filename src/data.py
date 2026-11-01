"""Data access for population-field forecasting.

Everything is expressed on the populated-cell subset (cells with any population in 2023).
Hourly series keep NaN for missing / anomalous hours; windows that touch a NaN are skipped.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                   # noqa: E402
from loader import load_pop10, load_grid, load_adjacency, months_between, anomalous_days  # noqa: E402
from build_baseline_atlas import daytype_of          # noqa: E402

ML_DIR = C.BASE                                       # base runs (results/base)
YEARS = {"train": ("202301", "202312"), "val": ("202401", "202412"), "test": ("202501", "202512")}
SERVICE_RADIUS_M = {"shelter_eq_outdoor": 500, "shelter_eq_indoor": 500, "shelter_civil_defense": 500,
                    "hospital": 2000, "school_elem": 1000, "school_mid": 1000, "school_high": 1000,
                    "senior_center": 500}
FACILITY_TYPES = ["shelter_eq_outdoor", "shelter_eq_indoor", "shelter_civil_defense", "hospital", "clinic",
                  "school_elem", "school_mid", "school_high", "senior_center", "senior_welfare_center", "subway_station", "bus_stop", "emergency"]


def time_features(ti: pd.DataFrame) -> np.ndarray:
    """[T, 8]: sin/cos hour, sin/cos day-of-week, holiday flag, daytype one-hot(3)."""
    h = ti.tt.values.astype(float)
    dow = ti.datetime.dt.dayofweek.values.astype(float)
    dt = daytype_of(ti)
    f = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                  np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                  (dt == 2).astype(float), (dt == 0).astype(float), (dt == 1).astype(float), (dt == 2).astype(float)], axis=1)
    return f.astype(np.float32)


class PopulationData:
    def __init__(self, k: int = 24, horizon: int = 24, cache: bool = True):
        self.k, self.H = k, horizon
        ML_DIR.mkdir(parents=True, exist_ok=True)
        grid = load_grid(with_geometry=False)
        self.grid = grid
        anom = anomalous_days()

        # ---- hourly series per split, NaN where unavailable ----
        self.X, self.ti, self.tf = {}, {}, {}
        for split, (a, b) in YEARS.items():
            X, ti = load_pop10(months_between(a, b))
            bad = (~ti.available.values) | ti.ymd.isin(anom).values
            X = X.astype(np.float32).copy()
            X[bad] = np.nan
            self.X[split], self.ti[split], self.tf[split] = X, ti, time_features(ti)

        # ---- populated cells (from train year) ----
        tr = self.X["train"]
        valid_h = ~np.isnan(tr[:, 0])
        self.cells = np.where(np.nanmax(tr[valid_h], axis=0) > 0)[0]
        self.N = len(self.cells)
        for s in self.X:
            self.X[s] = self.X[s][:, self.cells]

        # ---- normalisation from the train year ----
        L = np.log1p(self.X["train"])
        self.mu = np.nanmean(L, axis=0).astype(np.float32)
        self.sigma = np.maximum(np.nanstd(L, axis=0), 0.1).astype(np.float32)
        self.Z = {s: ((np.log1p(self.X[s]) - self.mu) / self.sigma).astype(np.float32) for s in self.X}

        # ---- seasonal profile (train year only): median per (cell, daytype, hour) in log space ----
        dt = daytype_of(self.ti["train"]); tt = self.ti["train"].tt.values
        self.profile = np.zeros((3, 24, self.N), np.float32)
        self.profile_scale = np.zeros((3, 24, self.N), np.float32)
        for d in range(3):
            for h in range(24):
                blk = L[(dt == d) & (tt == h)]
                med = np.nanmedian(blk, axis=0)
                self.profile[d, h] = med
                self.profile_scale[d, h] = np.maximum(np.nanmedian(np.abs(blk - med), axis=0) * 1.4826, 0.05)

        # ---- static node features: log1p facility counts, Vs30 ----
        fc_path = C.OUT / "facilities" / "facility_counts_by_cell.parquet"
        fc = pd.read_parquet(fc_path).set_index("cell_idx") if fc_path.exists() else None
        feats = []
        for t in FACILITY_TYPES:
            v = fc[f"n_{t}"].reindex(self.cells).fillna(0).values if fc is not None and f"n_{t}" in fc else np.zeros(self.N)
            feats.append(np.log1p(v))
        vs = grid.Vs30.reindex(self.cells).values.astype(float)
        vs = np.where(np.isnan(vs), np.nanmedian(vs), vs)
        feats.append((vs - vs.mean()) / vs.std())
        feats.append(self.mu); feats.append(self.sigma)
        S = np.stack(feats, axis=1).astype(np.float32)
        self.static = (S - S.mean(0)) / (S.std(0) + 1e-6)

        # ---- graph operators on the subset ----
        self.adj = {}
        for kind in ("rook", "queen"):
            A = load_adjacency(kind)[self.cells][:, self.cells]
            A = A + sp.eye(self.N, format="csr")
            d = np.asarray(A.sum(1)).ravel()
            Dm = sp.diags(1 / np.sqrt(d))
            self.adj[kind] = (Dm @ A @ Dm).tocsr().astype(np.float32)

        # ---- facility hypergraph incidence H [N, E] ----
        self.Hinc = self.build_hypergraph()

    def build_hypergraph(self):
        f_path = C.OUT / "facilities" / "facilities.parquet"
        if not f_path.exists():
            return sp.csr_matrix((self.N, 0), dtype=np.float32)
        fac = pd.read_parquet(f_path)
        fac = fac[fac.has_coords & fac.ftype.isin(SERVICE_RADIUS_M)]
        xy = self.grid.loc[self.cells, ["CELL_X", "CELL_Y"]].values
        tree = cKDTree(xy)
        rows, cols = [], []
        e = 0
        for t, r in SERVICE_RADIUS_M.items():
            pts = fac.loc[fac.ftype == t, ["x", "y"]].values
            for hits in tree.query_ball_point(pts, r):
                if hits:
                    rows.extend(hits); cols.extend([e] * len(hits)); e += 1
        H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(self.N, e))
        self.hyper_types = list(SERVICE_RADIUS_M)
        return H

    # ---- windows ----
    def windows(self, split: str, stride: int = 1):
        """Start indices t such that hours [t-k+1, t+H] are all valid."""
        ok = ~np.isnan(self.Z[split][:, 0])
        T = len(ok)
        csum = np.concatenate([[0], np.cumsum(ok)])
        L = self.k + self.H
        starts = np.arange(self.k - 1, T - self.H, stride)
        good = csum[starts + self.H + 1] - csum[starts - self.k + 1] == L
        return starts[good]

    def batch(self, split: str, ts: np.ndarray):
        """Return (x_hist [B,k,N], tf_hist [B,k,8], tf_fut [B,H,8], y [B,H,N]) in normalised space."""
        Z, tf = self.Z[split], self.tf[split]
        xh = np.stack([Z[t - self.k + 1:t + 1] for t in ts])
        y = np.stack([Z[t + 1:t + 1 + self.H] for t in ts])
        th = np.stack([tf[t - self.k + 1:t + 1] for t in ts])
        tfu = np.stack([tf[t + 1:t + 1 + self.H] for t in ts])
        return xh, th, tfu, y

    def profile_pred(self, split: str, ts: np.ndarray):
        """Seasonal-profile baseline in normalised space for the H future hours of each window."""
        ti = self.ti[split]
        dt_all = daytype_of(ti); tt_all = ti.tt.values
        out = np.empty((len(ts), self.H, self.N), np.float32)
        for i, t in enumerate(ts):
            idx = np.arange(t + 1, t + 1 + self.H)
            out[i] = (self.profile[dt_all[idx], tt_all[idx]] - self.mu) / self.sigma
        return out

    def inverse(self, z):
        return np.expm1(z * self.sigma + self.mu)

    def extreme_threshold(self, split: str, ts: np.ndarray, z_thresh: float, min_ratio: float):
        """Log-space threshold [B,H,N] = profile + max(z_thresh*scale, log(min_ratio)).
        The ratio floor keeps 'extreme' meaning a real surge (e.g. +50%) even in cells with tiny MAD."""
        ti = self.ti[split]; dt_all = daytype_of(ti); tt_all = ti.tt.values
        out = np.empty((len(ts), self.H, self.N), np.float32)
        for i, t in enumerate(ts):
            idx = np.arange(t + 1, t + 1 + self.H)
            out[i] = self.profile[dt_all[idx], tt_all[idx]] + np.maximum(z_thresh * self.profile_scale[dt_all[idx], tt_all[idx]], np.log(min_ratio))
        return out

    def extreme_mask(self, split: str, ts: np.ndarray, z_thresh: float = 3.0, min_ratio: float = 1.5, min_pop: float = 100.0):
        """Boolean [B,H,N]: actual exceeds the extreme threshold and the cell holds at least min_pop people."""
        thr = self.extreme_threshold(split, ts, z_thresh, min_ratio)
        L = np.log1p(np.stack([self.X[split][t + 1:t + 1 + self.H] for t in ts]))
        return (L > thr) & (L > np.log1p(min_pop))

    def summary(self):
        return {"N_cells": int(self.N), "k": self.k, "H": self.H,
                "hours": {s: int(len(self.Z[s])) for s in self.Z},
                "valid_windows": {s: int(len(self.windows(s))) for s in self.Z},
                "hyperedges": int(self.Hinc.shape[1]), "static_features": int(self.static.shape[1]),
                "rook_nnz": int(self.adj["rook"].nnz)}


if __name__ == "__main__":
    d = PopulationData()
    print(json.dumps(d.summary(), indent=2))
