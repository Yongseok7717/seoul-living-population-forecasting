"""External prior channels for the extreme gate (information available before the target hour).

    python build_priors.py [--venue ../data/venue_events.csv]

Outputs processed/priors/
  prior_cal_YYYYMM.npy   [T_month, 9]      per-hour calendar priors (same for all cells): holiday, off-day (weekend/holiday),
                                          day before an off block (workday), day after, off-block length/5, position in block/5,
                                          workers' day (05-01), school vacation (Seoul approx.), year-end week (12-24..12-31)
  prior_rain_YYYYMM.npy  [T_month, N_all]  hourly rain (mm) from the gauge interpolation (2021-2024; zeros where unavailable), float16
  prior_rainok_YYYYMM.npy[T_month]         1 if rain data exists for the hour
  prior_venue_YYYYMM.npy [T_month, N_all]  scheduled-event window indicator (0/1) from the venue schedule CSV, float16
Venue CSV columns: date (YYYY-MM-DD), venue, type, start_hour, end_hour, note. Venues are mapped to cells within their radius.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import config as C
from loader import load_grid, months_between
from build_baseline_atlas import HOLIDAY_SET

OUT = C.OUT / "priors"
# venue -> (x, y in EPSG:5179, radius m); coordinates from the standard grid of the dong cells (approximate centres)
VENUES = {  # EPSG:5179 centres taken from the event-catalogue peak cells (2023-2024) and dong centroids
    "서울월드컵경기장": (947125, 1952125, 800), "잠실종합운동장": (962125, 1946375, 900), "고척스카이돔": (943765, 1944835, 650),
    "KSPO돔(올림픽공원)": (967125, 1946625, 800), "여의도한강공원": (949900, 1948300, 1100), "광화문광장": (953875, 1952875, 900),
    "여의도국회": (948400, 1947900, 900), "서울광장": (953603, 1951864, 700),
    "잠실야구장": (962650, 1946300, 700),   # east part of the 잠실종합운동장 complex (LG/두산 home, KBO)
}


def calendar_priors(ti):
    ymd = ti.ymd.astype(str).values; dt = pd.to_datetime(ti.datetime); days = pd.to_datetime(pd.Series(sorted(set(ymd))), format="%Y%m%d")
    d_all = pd.date_range(days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10), freq="D")
    off = np.array([(d.dayofweek >= 5) or (d.strftime("%Y%m%d") in HOLIDAY_SET) for d in d_all])
    hol = np.array([d.strftime("%Y%m%d") in HOLIDAY_SET for d in d_all])
    # off-block length and position
    blk_len = np.zeros(len(d_all)); blk_pos = np.zeros(len(d_all)); i = 0
    while i < len(d_all):
        if off[i]:
            j = i
            while j < len(d_all) and off[j]:
                j += 1
            blk_len[i:j] = j - i; blk_pos[i:j] = np.arange(1, j - i + 1); i = j
        else:
            i += 1
    pre = np.zeros(len(d_all), bool); post = np.zeros(len(d_all), bool)
    for k in range(1, len(d_all) - 1):
        pre[k] = (not off[k]) and off[k + 1] and blk_len[k + 1] >= 3 or ((not off[k]) and hol[k + 1])
        post[k] = (not off[k]) and off[k - 1] and blk_len[k - 1] >= 3 or ((not off[k]) and hol[k - 1])
    md = np.array([d.strftime("%m%d") for d in d_all])
    workers = md == "0501"
    vac = np.array([("0720" <= m <= "0820") or (m >= "1228") or (m <= "0228") for m in md])
    yearend = np.array([("1224" <= m <= "1231") for m in md])
    tab = pd.DataFrame({"hol": hol, "off": off, "pre": pre, "post": post, "blk_len": blk_len / 5, "blk_pos": blk_pos / 5, "workers": workers, "vac": vac, "yearend": yearend}, index=d_all.strftime("%Y%m%d"))
    F = tab.reindex(ymd).values.astype(np.float32)
    return F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--venue", default=None); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); grid = load_grid(with_geometry=False); N = len(grid)
    xy = grid[["CELL_X", "CELL_Y"]].values.astype(float)
    months = months_between("202301", "202601")
    rain = {}
    for y in (2023, 2024, 2025):
        f = C.OUT / "hazard" / f"rain_hourly_{y}.npy"
        if f.exists():
            idx = pd.read_parquet(C.OUT / "hazard" / f"rain_hourly_index_{y}.parquet"); rain[y] = (np.load(f, mmap_mode="r"), pd.to_datetime(idx.datetime))
    ven = None
    if args.venue and Path(args.venue).exists():
        ven = pd.read_csv(args.venue, encoding="utf-8-sig"); print("venue events", len(ven), ven.venue.value_counts().to_dict())
    cell_sets = {v: np.where(np.hypot(xy[:, 0] - x, xy[:, 1] - y) <= r)[0] for v, (x, y, r) in VENUES.items()}
    print({v: len(c) for v, c in cell_sets.items()})
    for m in months:
        p = C.OUT / "tensors" / "local" / f"time_index_{m}.parquet"
        if not p.exists():
            continue
        ti = pd.read_parquet(p); T = len(ti); dt = pd.to_datetime(ti.datetime)
        np.save(OUT / f"prior_cal_{m}.npy", calendar_priors(ti))
        R = np.zeros((T, N), np.float16); ok = np.zeros(T, np.float32); y = int(m[:4])
        if y in rain:
            arr, ridx = rain[y]; pos = pd.Series(np.arange(len(ridx)), index=ridx).reindex(dt).values; good = np.isfinite(pos)
            R[good] = np.nan_to_num(np.asarray(arr[pos[good].astype(int)])).astype(np.float16); ok[good] = 1
        np.save(OUT / f"prior_rain_{m}.npy", R); np.save(OUT / f"prior_rainok_{m}.npy", ok)
        V = np.zeros((T, N), np.float16)
        if ven is not None:
            sub = ven[ven.date.astype(str).str.replace("-", "").str[:6] == m]
            for e in sub.itertuples():
                cells = cell_sets.get(e.venue)
                if cells is None:
                    continue
                d0 = pd.Timestamp(e.date); s, en = int(e.start_hour), int(e.end_hour)
                rows = np.where((dt >= d0 + pd.Timedelta(hours=max(s - 2, 0))) & (dt <= d0 + pd.Timedelta(hours=min(en + 1, 23))))[0]
                V[np.ix_(rows, cells)] = 1
        np.save(OUT / f"prior_venue_{m}.npy", V)
        print(m, "cal", calendar_priors(ti).shape, "rain ok", int(ok.sum()), "venue hours", int((V.sum(1) > 0).sum()), flush=True)


if __name__ == "__main__":
    main()
