"""Task 1.1-1.5: canonical representation X[T, N, A, S], cleaning report, calendar features.

The monthly tensors processed/tensors/local/local_YYYYMM.npy already hold [hours, cells, 29] with
29 = SPOP + M00..M70 + F00..F70 (A = 14 age bands x S = 2 sexes). This module exposes them as
X[T, N, A, S] plus metadata, and writes the cleaning report (processed/eda/cleaning_report.md).

    from canonical import load_canonical, AGE_LABELS, SEX_LABELS, calendar_features
    X, ti, meta = load_canonical(["202306"])         # X: float32 [T, N, 14, 2] (NaN for unusable hours)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import config as C                                                     # noqa: E402
from loader import load_local, load_grid, anomalous_days, months_between   # noqa: E402
from build_baseline_atlas import daytype_of, HOLIDAY_SET               # noqa: E402

EDA = C.OUT / "eda"
AGE_LABELS = ["0-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
SEX_LABELS = ["M", "F"]
COLS = C.LOCAL_VALUE_COLS
GROUP_INDEX = np.array([[COLS.index(f"{s}{a}") for s in SEX_LABELS] for a in C.AGE_BANDS])   # [14, 2] -> column index
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def load_canonical(months, drop_age_0_9=False):
    """Return (X [T, N_all, A, S] float32 with NaN rows for unusable hours, time index, metadata)."""
    L, ti = load_local(months)
    X = L[:, :, GROUP_INDEX.reshape(-1)].reshape(L.shape[0], L.shape[1], 14, 2)
    bad = (~ti.available.values) | ti.ymd.isin(anomalous_days()).values
    X = X.copy(); X[bad] = np.nan
    if drop_age_0_9:
        X = X[:, :, 1:, :]
    meta = {"age_labels": AGE_LABELS[1:] if drop_age_0_9 else AGE_LABELS, "sex_labels": SEX_LABELS,
            "months": list(months), "regime": {m: ("A" if m <= "202406" else "B" if m <= "202412" else "C") for m in months},
            "spop_col_note": "SPOP (column 0 of local_*.npy) may differ from the group sum by rounding / suppression"}
    return X, ti, meta


def calendar_features(ti: pd.DataFrame) -> pd.DataFrame:
    dt = ti.datetime
    out = pd.DataFrame({"datetime": dt, "hour": dt.dt.hour, "dow": dt.dt.dayofweek, "is_weekend": dt.dt.dayofweek >= 5,
                        "month": dt.dt.month, "season": dt.dt.month.map(SEASON), "year": dt.dt.year,
                        "is_holiday": ti.ymd.isin(HOLIDAY_SET).values, "daytype": daytype_of(ti)})
    out["daytype_name"] = out.daytype.map({0: "weekday", 1: "saturday", 2: "sunday_holiday"})
    return out


def cleaning_report(months=None):
    """Task 1.2: duplicates, missing, negative, group-consistency checks over the requested months."""
    months = months or months_between("202301", "202601")
    EDA.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=False)
    rows = []
    for ym in months:
        L, ti = load_local([ym])
        S = L[:, :, 0]; G = L[:, :, 1:]
        avail = ti.available.values
        gs = G.sum(axis=2)
        with np.errstate(invalid="ignore"):
            rel = np.abs(S - gs) / np.maximum(S, 1)
        rows.append({
            "month": ym, "hours": len(ti), "hours_available": int(avail.sum()),
            "anomalous_days": int(ti.ymd[avail].isin(anomalous_days()).sum() // 24),
            "negative_values": int(np.nansum(L[avail] < 0)),
            "nan_inside_available_hours": int(np.isnan(L[avail]).sum()),
            "cells_ever_populated": int((np.nanmax(S[avail], axis=0) > 0).sum()) if avail.any() else 0,
            "groupsum_over_spop": float(np.nansum(gs[avail]) / max(np.nansum(S[avail]), 1)),
            "cell_hours_rel_diff_gt_5pct": int(np.nansum(rel[avail] > 0.05)),
            "duplicate_timestamps": int(ti.datetime.duplicated().sum()),
            "timestamps_continuous": bool((ti.datetime.diff().dropna() == pd.Timedelta("1h")).all()),
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(EDA / "cleaning_report.csv", index=False, encoding="utf-8-sig")
    lines = ["# Task 1.2 cleaning report", "",
             f"- grid: {len(grid)} unique CELL_IDs (row-major cell_idx), duplicates: {int(grid.CELL_ID.duplicated().sum())}",
             "- source rows of one cell split across several 행정동 were summed to one cell (build_local_tensors.py)",
             "- missing-data mask: hours with available=False (missing files) or anomalous days are NaN in the canonical tensor; "
             "in-file suppression (regime B '*', regime C zeros) is recorded per month, see processed/qc/summary.md",
             "- negative values: " + ("none" if rep.negative_values.sum() == 0 else str(int(rep.negative_values.sum()))),
             f"- timestamps continuous in every month: {bool(rep.timestamps_continuous.all())}, duplicates: {int(rep.duplicate_timestamps.sum())}",
             f"- group sum vs SPOP: regime A months {rep[rep.month <= '202406'].groupsum_over_spop.mean():.4f}, later {rep[rep.month > '202406'].groupsum_over_spop.mean():.4f} (suppression removes small group values)",
             "", rep.to_markdown(index=False), ""]
    (EDA / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    return rep


if __name__ == "__main__":
    cleaning_report()
