"""Convenience loaders for the processed products.

    from loader import load_grid, load_adjacency, load_local, load_pop10, load_foreign, months_between

    grid = load_grid()                                 # GeoDataFrame, EPSG:5179, index = cell_idx
    A = load_adjacency("rook")                         # scipy CSR (N, N)
    X, ti = load_pop10(months_between("202301", "202312"))   # X: float32 [T, N], ti: time index DataFrame
    L, ti = load_local(["202306"], cols=["SPOP", "F70", "M70"])  # [T, N, len(cols)]
    F, ti = load_foreign("long", ["202306"])           # [T, N] total long-term foreigners
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import config as C


def months_between(start: str, end: str):
    return [m for m in C.LOCAL_MONTHS if start <= m <= end]


def load_grid(with_geometry: bool = True):
    if with_geometry:
        import geopandas as gpd
        g = gpd.read_file(C.OUT_GRID / "grid_standard.gpkg", layer="grid")
    else:
        g = pd.read_parquet(C.OUT_GRID / "grid_index.parquet")
    return g.set_index("cell_idx", drop=False)


def load_grid_meta():
    return json.loads((C.OUT_GRID / "grid_meta.json").read_text(encoding="utf-8"))


def load_adjacency(kind: str = "rook"):
    return sp.load_npz(C.OUT_GRID / f"adjacency_{kind}.npz")


def load_cell_dong_map():
    return pd.read_parquet(C.OUT_GRID / "cell_dong_map.parquet")


def local_columns():
    return json.loads((C.OUT_LOCAL / "columns.json").read_text(encoding="utf-8"))


def _time_index(prefix_dir, name_fmt, months):
    parts = []
    offset = 0
    for ym in months:
        ti = pd.read_parquet(prefix_dir / name_fmt.format(ym=ym))
        ti["ym"] = ym
        ti["t_global"] = ti.t_idx + offset
        offset += len(ti)
        parts.append(ti)
    return pd.concat(parts, ignore_index=True)


def anomalous_days():
    """Days flagged by qc_report.py (doubled values, ~1,750 cells dropped, truncated). Empty set if not yet built."""
    f = C.OUT_QC / "anomalous_days.csv"
    if not f.exists():
        return set()
    return set(pd.read_csv(f, dtype={"ymd": str}).ymd)


def _filter_days(X, ti, drop_missing_days, drop_anomalous):
    keep = np.ones(len(ti), dtype=bool)
    if drop_missing_days:
        keep &= ti.available.values
    if drop_anomalous:
        keep &= ~ti.ymd.isin(anomalous_days()).values
    if keep.all():
        return X, ti
    return X[keep], ti[keep].reset_index(drop=True)


def load_local(months, cols=None, drop_missing_days: bool = False, drop_anomalous: bool = False):
    """Return (array [T, N, len(cols)], time_index). cols=None loads all 29 value columns.
    Data are read via memmap and copied only for the requested columns."""
    all_cols = local_columns()["value_cols"]
    cidx = None if cols is None else [all_cols.index(c) for c in cols]
    arrays = []
    for ym in months:
        a = np.load(C.OUT_LOCAL / f"local_{ym}.npy", mmap_mode="r")
        arrays.append(np.asarray(a if cidx is None else a[:, :, cidx]))
    X = np.concatenate(arrays, axis=0)
    ti = _time_index(C.OUT_LOCAL, "time_index_{ym}.parquet", months)
    return _filter_days(X, ti, drop_missing_days, drop_anomalous)


def load_pop10(months, drop_missing_days: bool = False, drop_anomalous: bool = False):
    """Analysis total (ages 10+): float32 [T, N]."""
    X = np.concatenate([np.load(C.OUT_LOCAL / f"pop10_{ym}.npy", mmap_mode="r") for ym in months], axis=0)
    ti = _time_index(C.OUT_LOCAL, "time_index_{ym}.parquet", months)
    return _filter_days(X, ti, drop_missing_days, drop_anomalous)


def load_local_masked(months):
    """uint8 [T, N, 29] count of '*' source entries; zeros for months without a masked_ file."""
    out = []
    for ym in months:
        f = C.OUT_LOCAL / f"masked_{ym}.npy"
        if f.exists():
            out.append(np.load(f, mmap_mode="r"))
        else:
            n = len(pd.read_parquet(C.OUT_LOCAL / f"time_index_{ym}.parquet"))
            out.append(np.zeros((n, load_grid_meta()["n_cells"], 29), dtype=np.uint8))
    return np.concatenate(out, axis=0)


def load_foreign(kind: str, months, drop_missing_days: bool = False):
    """kind in {'long', 'temp'}. Returns (float32 [T, N] totals, time_index)."""
    X = np.concatenate([np.load(C.OUT_FORN / f"{kind}_spop_{ym}.npy", mmap_mode="r") for ym in months], axis=0)
    ti = _time_index(C.OUT_FORN, kind + "_time_index_{ym}.parquet", months)
    if drop_missing_days:
        keep = ti.available.values
        X, ti = X[keep], ti[keep].reset_index(drop=True)
    return X, ti


def load_foreign_nationalities(kind: str, months):
    """Long table with nationality detail for populated cells (t_idx is month-local; ym column added)."""
    parts = []
    for ym in months:
        d = pd.read_parquet(C.OUT_FORN / f"{kind}_nat_{ym}.parquet")
        d.insert(0, "ym", ym)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def to_lattice(values: np.ndarray, grid=None, fill=np.nan):
    """Scatter a length-N vector onto the [rows, cols] lattice for imaging (row 0 = south)."""
    g = grid if grid is not None else load_grid(with_geometry=False)
    meta = load_grid_meta()
    img = np.full(meta["lattice_shape_rows_cols"], fill, dtype=np.float32)
    img[g.row.values, g.col.values] = values
    return img
