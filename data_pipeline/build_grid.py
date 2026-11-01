"""Build the standard 250m grid: one canonical cell index, EPSG:5179 geometry,
Vs30 attribute, rook/queen adjacency, and the cell<->administrative-dong share table.

Outputs (processed/grid/):
  grid_standard.gpkg   polygons, EPSG:5179, one row per cell (cell_idx 0..N-1)
  grid_index.parquet   same attributes without geometry
  adjacency_rook.npz / adjacency_queen.npz   scipy CSR, shape (N, N), symmetric 0/1
  cell_dong_map.parquet  (CELL_ID, cell_idx, H_DNG_CD, dong_name, share)
  grid_meta.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
import config as C


def main():
    C.ensure_dirs()
    g = gpd.read_file(C.MATCH_SHP)
    assert g.crs.to_epsg() == C.CRS_STANDARD, g.crs
    v = gpd.read_file(C.VS30_SHP)[["CELL_ID", "Vs30", "geometry"]]

    # Geometry cross-check: Vs30 layer (5186) reprojected to 5179 must coincide with match.shp
    vc = v.set_index("CELL_ID").to_crs(C.CRS_STANDARD).geometry.centroid
    gc = g.set_index("CELL_ID").geometry.centroid
    common = gc.index.intersection(vc.index)
    dist = gc.loc[common].distance(vc.loc[common])
    geom_max_offset_m = float(dist.max())

    g = g.merge(v.drop(columns="geometry"), on="CELL_ID", how="left")
    # Vs30 == 0 is a missing-value placeholder, not a physical shear-wave velocity
    g["vs30_missing"] = g["Vs30"].isna() | (g["Vs30"] <= 0)
    g.loc[g["vs30_missing"], "Vs30"] = np.nan

    # Regular lattice indices from cell centroids
    xmin, ymin = g.CELL_X.min(), g.CELL_Y.min()
    g["col"] = np.rint((g.CELL_X - xmin) / C.CELL_SIZE).astype(int)
    g["row"] = np.rint((g.CELL_Y - ymin) / C.CELL_SIZE).astype(int)
    assert not g.duplicated(["row", "col"]).any()

    # Canonical order: row-major from the south-west corner
    g = g.sort_values(["row", "col"]).reset_index(drop=True)
    g.insert(0, "cell_idx", np.arange(len(g)))
    g = g[["cell_idx", "CELL_ID", "GID", "CELL_X", "CELL_Y", "row", "col", "Vs30", "vs30_missing", "geometry"]]
    N = len(g)

    # Adjacency on the lattice
    lut = {(r, c): i for i, r, c in zip(g.cell_idx, g.row, g.col)}
    rook_off = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queen_off = rook_off + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def build(offsets):
        rows, cols = [], []
        for i, r, c in zip(g.cell_idx, g.row, g.col):
            for dr, dc in offsets:
                j = lut.get((r + dr, c + dc))
                if j is not None:
                    rows.append(i); cols.append(j)
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))
        assert (A != A.T).nnz == 0
        return A

    A_rook, A_queen = build(rook_off), build(queen_off)
    sp.save_npz(C.OUT_GRID / "adjacency_rook.npz", A_rook)
    sp.save_npz(C.OUT_GRID / "adjacency_queen.npz", A_queen)

    # Cell <-> dong share table from the June 2023 monthly-average file (all 24 hours)
    avg = pd.read_csv(C.RAW_LOCAL / "2023" / "250_LOCAL_RESD_AVG_202306.csv", engine="pyarrow",
                      usecols=["H_DNG_CD", "CELL_ID", "SPOP"])
    cd = avg.groupby(["CELL_ID", "H_DNG_CD"], as_index=False).SPOP.sum()
    cd["share"] = cd.SPOP / cd.groupby("CELL_ID").SPOP.transform("sum")
    names = pd.read_csv(C.DONG_CODE_CSV, encoding="utf-8-sig").rename(columns={"행정동코드": "H_DNG_CD", "행정동명": "dong_name"})
    cd = cd.merge(names, on="H_DNG_CD", how="left").merge(g[["CELL_ID", "cell_idx"]], on="CELL_ID", how="left")
    unknown_cells = sorted(cd.loc[cd.cell_idx.isna(), "CELL_ID"].unique())
    cd = cd.dropna(subset=["cell_idx"]).astype({"cell_idx": int})
    cd = cd[["CELL_ID", "cell_idx", "H_DNG_CD", "dong_name", "share"]].sort_values(["cell_idx", "share"], ascending=[True, False])
    cd.to_parquet(C.OUT_GRID / "cell_dong_map.parquet", index=False)
    # Dominant dong per cell for convenience
    dom = cd.drop_duplicates("cell_idx")[["cell_idx", "H_DNG_CD", "dong_name"]].rename(columns={"H_DNG_CD": "dong_major", "dong_name": "dong_major_name"})
    g = g.merge(dom, on="cell_idx", how="left")

    g.to_file(C.OUT_GRID / "grid_standard.gpkg", layer="grid", driver="GPKG")
    g.drop(columns="geometry").to_parquet(C.OUT_GRID / "grid_index.parquet", index=False)

    meta = {
        "crs_epsg": C.CRS_STANDARD, "n_cells": int(N), "cell_size_m": C.CELL_SIZE,
        "lattice_shape_rows_cols": [int(g.row.max() + 1), int(g.col.max() + 1)],
        "origin_xy_sw_centroid": [float(xmin), float(ymin)],
        "vs30_missing_cells": int(g.vs30_missing.sum()),
        "vs30_geometry_max_offset_m_after_reprojection": geom_max_offset_m,
        "cells_with_population_but_not_in_grid": unknown_cells,
        "cells_with_dong_mapping": int(cd.cell_idx.nunique()),
        "cells_split_across_multiple_dongs": int((cd.groupby("cell_idx").size() > 1).sum()),
        "rook_edges": int(A_rook.nnz // 2), "queen_edges": int(A_queen.nnz // 2),
        "source_match_shp": str(C.MATCH_SHP), "source_vs30_shp": str(C.VS30_SHP),
        "cell_dong_share_source": "250_LOCAL_RESD_AVG_202306.csv (sum over 24 hours)",
    }
    (C.OUT_GRID / "grid_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
