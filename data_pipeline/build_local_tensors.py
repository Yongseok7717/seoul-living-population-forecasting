"""Convert daily 250m domestic living-population CSVs into monthly dense tensors.

For every month YYYYMM (processed/tensors/local/):
  local_YYYYMM.npy        float32 [D*24, N, 29]  SPOP + 28 sex/age groups; masked entries contribute 0
  pop10_YYYYMM.npy        float32 [D*24, N]      analysis total = SPOP - (M00+F00)   (0-9 excluded)
  masked_YYYYMM.npy       uint8   [D*24, N, 29]  number of source rows whose value was '*' (only written if any)
  time_index_YYYYMM.parquet  one row per hour: t_idx, ymd, tt, datetime, available, regime
Rows for missing days are NaN in local_/pop10_.
Cells split across several dongs are summed to one cell.
The 250_LOCAL_RESD_* folders are read first; the SE_250M_SPOP_* folders of 2023 are byte-identical duplicates
and are used only as a fallback when the 250_ file is empty (2023-03-02, 2023-03-03).
Encodings: UTF-8 with BOM everywhere except 2024-07..2024-12, which are cp949 with quoted, padded fields.

Suppression regimes found in the source (see qc/local_daily_qc.csv):
  A  2023-01 .. 2024-06 : full precision (0.01), no suppression
  B  2024-07 .. 2024-12 : values <= 3 replaced by '*'          (detectable -> masked_ file)
  C  2025-01 ..         : values <= 3 replaced by 0            (NOT distinguishable from true zeros)
Malformed CSV rows are skipped and counted (qc column bad_rows).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys, json, calendar, time
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import pyarrow.csv as pc
import config as C

COLS = C.LOCAL_VALUE_COLS
C09_IDX = [COLS.index(c) for c in C.EXCLUDED_GROUPS]
GROUP_IDX = [COLS.index(c) for c in C.GROUP_COLS]


def regime(ym: str) -> str:
    if ym <= "202406":
        return "A"
    if ym <= "202412":
        return "B"
    return "C"


def _grid_lut():
    gi = pd.read_parquet(C.OUT_GRID / "grid_index.parquet", columns=["cell_idx", "CELL_ID"])
    return dict(zip(gi.CELL_ID, gi.cell_idx)), len(gi)


def day_file(ym: str, ymd: str):
    """Prefer 250_LOCAL_RESD_*; fall back to the SE_250M_SPOP_* copy (2023 only) when the 250_ file is
    missing or empty (2023-03-02 and 2023-03-03 are header-only in the 250_ folder)."""
    primary = C.RAW_LOCAL / ym[:4] / f"250_LOCAL_RESD_{ym}" / f"250_LOCAL_RESD_{ymd}.csv"
    alt = C.RAW_LOCAL / ym[:4] / f"SE_250M_SPOP_LOCAL_RESD_{ym}" / f"SE_250M_SPOP_LOCAL_RESD_{ymd}.csv"
    p_size = primary.stat().st_size if primary.exists() else 0
    a_size = alt.stat().st_size if alt.exists() else 0
    # the SE_ copy wins when the 250_ file is empty or truncated (2023-03-02/03 empty, 2023-04-01 cut after 4 hours)
    if a_size > 1024 and a_size > 1.05 * p_size:
        return alt
    return primary


KO_HEADER = {"일자": "YMD", "시간": "TT", "행정동코드": "H_DNG_CD", "250M격자": "CELL_ID", "생활인구합계": "SPOP",
             "생활인구": "SPOP"}


def sniff_encoding(f) -> str:
    """Files of 2024-07..2024-12 are cp949; everything else is UTF-8 (with BOM)."""
    head = open(f, "rb").read(4096)
    try:
        head.decode("utf-8")
        return "utf8"
    except UnicodeDecodeError as e:
        # an error only at the very end means the 4096-byte cut split a multibyte UTF-8 character
        return "utf8" if e.start >= len(head) - 3 else "cp949"


def read_day(f):
    bad = [0]

    def handler(row):
        bad[0] += 1
        return "skip"

    enc = sniff_encoding(f)
    # Column names are resolved after reading because some months carry Korean or padded headers.
    table = pc.read_csv(
        f,
        read_options=pc.ReadOptions(encoding=enc),
        parse_options=pc.ParseOptions(invalid_row_handler=handler),
        convert_options=pc.ConvertOptions(null_values=["*"], strings_can_be_null=True,
                                          column_types={"CELL_ID": "string", "250M격자": "string",
                                                        "H_DNG_CD": "string", "행정동코드": "string"}),
    )
    df = table.to_pandas()
    df.columns = [KO_HEADER.get(c.strip(), c.strip()) for c in df.columns]
    missing = [c for c in ["TT", "CELL_ID"] + COLS if c not in df.columns]
    assert not missing, f"{f}: missing columns {missing}; header={list(df.columns)}"
    df["TT"] = pd.to_numeric(df["TT"].astype(str).str.strip()).astype(int)
    df["CELL_ID"] = df["CELL_ID"].astype(str).str.strip()
    for c in COLS:
        if df[c].dtype != np.float32:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df[["TT", "CELL_ID"] + COLS], bad[0], enc


def process_day(args):
    """Return (ymd, values[24,N,29], masked[24,N,29] or None, qc dict)."""
    ymd, lut, N = args
    f = day_file(ymd[:6], ymd)
    qc = {"ymd": ymd, "regime": regime(ymd[:6]), "available": f.exists(), "file_bytes": f.stat().st_size if f.exists() else 0,
          "source": f.parent.name if f.exists() else None}
    if not f.exists():
        return ymd, None, None, qc
    df, bad, enc = read_day(f)
    qc["encoding"] = enc
    qc["bad_rows"] = bad
    qc["n_rows"] = len(df)
    if len(df) == 0:                       # empty file (e.g. 2023-03-02, 2023-03-03): treat as a missing day
        qc["available"] = False
        return ymd, None, None, qc
    idx = df.CELL_ID.map(lut)
    unknown = idx.isna()
    qc["n_cells_unknown"] = int(df.loc[unknown, "CELL_ID"].nunique())
    qc["spop_unknown_cells"] = float(np.nansum(df.loc[unknown, "SPOP"].values))
    df = df.loc[~unknown]
    idx = idx[~unknown].astype(int).values
    tt = df.TT.values.astype(int)
    vals = df[COLS].values.astype(np.float32)
    isna = np.isnan(vals)
    n_star = int(isna.sum())
    vals = np.nan_to_num(vals, nan=0.0)

    arr = np.zeros((24, N, len(COLS)), dtype=np.float32)
    np.add.at(arr, (tt, idx), vals)                     # sums the pieces of cells split across dongs
    masked = None
    if n_star:
        masked = np.zeros((24, N, len(COLS)), dtype=np.uint8)
        np.add.at(masked, (tt, idx), isna.astype(np.uint8))

    spop = arr[:, :, 0]
    gsum = arr[:, :, GROUP_IDX].sum(axis=2)
    c09 = arr[:, :, C09_IDX].sum(axis=2)
    qc.update({
        "n_hours": int(len(np.unique(tt))),
        "n_cells": int(len(np.unique(idx))),
        "n_star_entries": n_star,
        "n_star_spop": int(isna[:, 0].sum()),
        "spop_total_daily_mean": float(spop.sum(axis=1).mean()),
        "spop_total_tt03": float(spop[3].sum()), "spop_total_tt14": float(spop[14].sum()),
        "pop10_total_daily_mean": float((spop - c09).sum(axis=1).mean()),
        "share_0_9_daily_mean": float(c09.sum() / max(spop.sum(), 1e-9)),
        "groupsum_over_spop": float(gsum.sum() / max(spop.sum(), 1e-9)),
        "max_abs_spop_minus_groupsum": float(np.abs(spop - gsum).max()),
        "max_cell_hour_spop": float(spop.max()),
    })
    return ymd, arr, masked, qc


def build_month(ym: str, lut, N, pool):
    y, m = int(ym[:4]), int(ym[4:])
    ndays = calendar.monthrange(y, m)[1]
    days = [f"{ym}{d:02d}" for d in range(1, ndays + 1)]
    out = np.full((ndays * 24, N, len(COLS)), np.nan, dtype=np.float32)
    msk = np.zeros((ndays * 24, N, len(COLS)), dtype=np.uint8)
    any_mask = False
    qcs = []
    for ymd, arr, masked, qc in pool.imap(process_day, [(d, lut, N) for d in days]):
        d = int(ymd[6:]) - 1
        if arr is not None:
            out[d * 24:(d + 1) * 24] = arr
        if masked is not None:
            msk[d * 24:(d + 1) * 24] = masked
            any_mask = True
        qcs.append(qc)
    np.save(C.OUT_LOCAL / f"local_{ym}.npy", out)
    np.save(C.OUT_LOCAL / f"pop10_{ym}.npy", out[:, :, 0] - out[:, :, C09_IDX].sum(axis=2))
    if any_mask:
        np.save(C.OUT_LOCAL / f"masked_{ym}.npy", msk)
    ti = pd.DataFrame({
        "t_idx": np.arange(ndays * 24),
        "ymd": np.repeat(days, 24), "tt": np.tile(np.arange(24), ndays),
    })
    ti["datetime"] = pd.to_datetime(ti.ymd + ti.tt.astype(str).str.zfill(2), format="%Y%m%d%H")
    ti["available"] = ti.ymd.map({q["ymd"]: q["available"] for q in qcs})
    ti["regime"] = regime(ym)
    ti.to_parquet(C.OUT_LOCAL / f"time_index_{ym}.parquet", index=False)
    return pd.DataFrame(qcs)


def main(months):
    C.ensure_dirs()
    lut, N = _grid_lut()
    (C.OUT_LOCAL / "columns.json").write_text(json.dumps({
        "value_cols": COLS, "group_cols": C.GROUP_COLS, "pop10_cols": C.POP10_COLS,
        "excluded_groups": C.EXCLUDED_GROUPS,
        "pop10_definition": "SPOP - (M00 + F00); masked pieces contribute 0",
        "layout": "[hours_in_month, cell_idx, value_cols]",
        "missing_day_fill": "NaN", "cell_order": "processed/grid/grid_index.parquet cell_idx",
        "suppression_regimes": {
            "A": "2023-01..2024-06 full precision",
            "B": "2024-07..2024-12 values<=3 -> '*' (masked_ file)",
            "C": "2025-01.. values<=3 -> 0 (indistinguishable from true zero)"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    qc_path = C.OUT_QC / "local_daily_qc.csv"
    if len(months) == len(C.LOCAL_MONTHS):
        qc_path.unlink(missing_ok=True)
    nproc = max(1, min(8, cpu_count() - 2))
    with Pool(nproc) as pool:
        for ym in months:
            t0 = time.time()
            qc = build_month(ym, lut, N, pool)
            qc.to_csv(qc_path, mode="a", header=not qc_path.exists(), index=False, encoding="utf-8-sig")
            miss = int((~qc.available).sum())
            print(f"{ym} [{regime(ym)}]: {len(qc)-miss}/{len(qc)} days, bad rows {int(qc.bad_rows.fillna(0).sum())}, "
                  f"star entries {int(qc.n_star_entries.fillna(0).sum())}, mean daily SPOP {qc.spop_total_daily_mean.mean():,.0f}, "
                  f"groupsum/SPOP {qc.groupsum_over_spop.mean():.4f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    months = sys.argv[1:] or C.LOCAL_MONTHS
    main(months)
