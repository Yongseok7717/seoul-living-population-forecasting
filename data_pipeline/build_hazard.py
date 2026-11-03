"""Hazard layers on the standard grid from the Seoul open-data downloads.

Inputs ([서울열린데이터 광장]/hazard/):
  OA-22715 zips : 48 Seoul rain gauges (구청·펌프장·동), 10-minute rainfall, one CSV per gauge-year: 2021-2024 (cp949) and
                  2025 (utf-8 or cp949, a title line before the header, pump stations named '...P')
  OA-15636 zips : flood-trace polygons 2022-2025 (EPSG:5179) with event dates
  OA-21172 zip  : flood-forecast depth contours (polylines, no CRS declared) -> kept as lines only

Outputs (processed/hazard/):
  rain_gauges.parquet          gauge name, matched 행정동, assigned x/y (dong centroid of the grid), cells served
  rain_hourly_YYYY.npy         float32 [hours_in_year, N_grid] hourly rainfall (mm) by nearest gauge
  rain_hourly_index_YYYY.parquet
  rain_daily_summary.csv       city-wide daily max hourly rain, mean daily total, gauges reporting
  flood_trace_cells.parquet    one row per (cell, flood polygon): year, event name, start/end date, depth, area
  flood_events.csv             events aggregated by (year, event name, start date): cells, area, dongs
  summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import io, re, zipfile, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import config as C
from loader import load_grid

SRC = C.OPEN_DIR / "hazard"
OUT = C.OUT / "hazard"
YEARS_RAIN = [2021, 2022, 2023, 2024, 2025]
RAIN_ZIPS = ["OA-22715_서울시_강우량_2021_2024.zip", "OA-22715_서울시_강우량_2025.zip"]
GAUGE_ALIAS = {"공항동P": "공항펌프장", "도림2P": "도림2펌프장"}                 # 2025 file names of two pump stations

# gauge name -> 행정동 name fragment used to place the gauge (구청 = the dong containing the district office)
GAUGE_DONG = {
    "가산2펌프장": "가산동", "갈현1동": "갈현제1동", "강남구청": "삼성1동", "강동구청": "성내1동", "강북구청": "수유1동",
    "강서구청": "화곡6동", "개봉2동": "개봉제2동", "개포2동": "개포2동", "고덕2동": "고덕제2동", "공항펌프장": "공항동",
    "관악구청": "청림동", "광진구청": "자양제1동", "구로구청": "구로제3동", "금천구청": "시흥제1동", "노원구청": "상계6·7동",
    "도림2펌프장": "신길제1동", "도봉구청": "방학제1동", "동대문구청": "용신동", "동작구청": "노량진제1동", "뚝섬펌프장": "성수1가제1동",
    "마천2동": "마천2동", "마포구청": "성산제2동", "면목펌프장": "면목제3·8동", "목동펌프장": "목1동", "반포펌프장": "반포본동",
    "봉원펌프장": "신촌동", "부암동": "부암동", "상계1동": "상계1동", "상월곡동": "월곡제2동", "서대문구청": "충현동",
    "서소문": "소공동", "서초구청": "서초2동", "성동구청": "왕십리도선동", "성북구청": "삼선동", "세곡동": "세곡동",
    "송파구청": "잠실6동", "신림펌프장": "신림동", "양천구청": "신정6동", "영등포구청": "당산제1동", "용산구청": "이태원제1동",
    "은평구청": "녹번동", "종로구청": "종로1·2·3·4가동", "중구청": "명동", "중랑구청": "신내1동", "증산펌프장": "증산동",
    "한남펌프장": "한남동", "휘경펌프장": "휘경제2동", "흑석펌프장": "흑석동",
}


def fix(n):
    try:
        return n.encode("cp437").decode("cp949")
    except Exception:
        return n


def dong_centroids(grid):
    g = grid[grid.dong_major_name.notna()]
    return g.groupby("dong_major_name")[["CELL_X", "CELL_Y"]].mean()


def place_gauges(grid):
    cents = dong_centroids(grid)
    names = list(cents.index)
    rows = []
    for gname, frag in GAUGE_DONG.items():
        hit = [n for n in names if n == frag] or [n for n in names if frag.replace("제", "") in n.replace("제", "")] \
              or [n for n in names if frag[:2] in n]
        if not hit:
            rows.append({"gauge": gname, "dong": None, "x": np.nan, "y": np.nan}); continue
        rows.append({"gauge": gname, "dong": hit[0], "x": cents.loc[hit[0], "CELL_X"], "y": cents.loc[hit[0], "CELL_Y"]})
    return pd.DataFrame(rows)


def read_rain(zpath):
    z = zipfile.ZipFile(zpath)
    series = {}
    for n in z.namelist():
        name = fix(n)
        gauge, year = name[:-8], int(name[-8:-4])
        gauge = GAUGE_ALIAS.get(gauge, gauge[:-1] + "펌프장" if gauge.endswith("P") else gauge)
        raw = z.read(n)
        try:
            txt = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            txt = raw.decode("cp949")
        lines = txt.splitlines(); h = next(i for i, l in enumerate(lines) if "," in l)       # skip a title line ('2025년')
        d = pd.read_csv(io.StringIO("\n".join(lines[h:])))
        d.columns = ["gauge", "time", "mm10"]
        t = pd.to_datetime(d["time"].str.slice(0, 16), format="%Y-%m-%d %H:%M", errors="coerce")
        v = pd.to_numeric(d["mm10"], errors="coerce")
        s = pd.Series(v.values, index=t).dropna()
        s = s[~s.index.duplicated(keep="first")]
        s = s[(s >= 0) & (s <= 60)]                                     # a 10-minute value above 60 mm is a gauge fault
        hourly = s.resample("h").sum(min_count=1)                      # mm per hour; NaN when no record in the hour
        hourly[hourly > 150] = np.nan                                   # >150 mm/h never observed in Seoul: fault
        series.setdefault(gauge, []).append(hourly)
    return {g: pd.concat(v).sort_index() for g, v in series.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=False)
    N = len(grid)
    lines = ["# Hazard layers summary", ""]

    # ---------- rain ----------
    gauges = place_gauges(grid)
    rain = {}
    for zf in RAIN_ZIPS:
        for g_, s_ in read_rain(SRC / zf).items():
            rain[g_] = pd.concat([rain[g_], s_]).sort_index() if g_ in rain else s_
    assert set(rain) <= set(GAUGE_DONG), sorted(set(rain) - set(GAUGE_DONG))
    gauges = gauges[gauges.gauge.isin(rain)].dropna(subset=["x"]).reset_index(drop=True)
    from scipy.spatial import cKDTree
    tree = cKDTree(gauges[["x", "y"]].values)
    _, nearest = tree.query(grid[["CELL_X", "CELL_Y"]].values)            # gauge index per cell
    gauges["cells_served"] = np.bincount(nearest, minlength=len(gauges))
    gauges.to_parquet(OUT / "rain_gauges.parquet", index=False)
    daily_rows = []
    for y in YEARS_RAIN:
        idx = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq="h")
        M = np.full((len(idx), len(gauges)), np.nan, np.float32)
        for j, g in enumerate(gauges.gauge):
            M[:, j] = rain[g].reindex(idx).values
        # fill a gauge's missing hours with the mean of reporting gauges (city-wide) so the grid field has no holes
        rowmean = np.nanmean(M, axis=1)
        M = np.where(np.isnan(M), rowmean[:, None], M)
        R = M[:, nearest]                                                   # [T, N]
        np.save(OUT / f"rain_hourly_{y}.npy", R.astype(np.float32))
        pd.DataFrame({"t_idx": np.arange(len(idx)), "datetime": idx}).to_parquet(OUT / f"rain_hourly_index_{y}.parquet", index=False)
        d = pd.DataFrame({"date": idx.date, "max_hourly_mm": np.nanmax(M, axis=1), "mean_mm": np.nanmean(M, axis=1),
                          "n_gauges": np.isfinite(M).sum(axis=1)})
        daily_rows.append(d.groupby("date").agg(max_hourly_mm=("max_hourly_mm", "max"), total_mm=("mean_mm", "sum"), n_gauges=("n_gauges", "min")))
    daily = pd.concat(daily_rows); daily.index.name = "date"
    daily.to_csv(OUT / "rain_daily_summary.csv", encoding="utf-8-sig", float_format="%.1f")
    lines += [f"- rain gauges used: {len(gauges)} (placed at the centroid of the 행정동 named in GAUGE_DONG; unmatched: {sorted(set(GAUGE_DONG) - set(gauges.gauge))})",
              f"- hourly grid rainfall saved for {YEARS_RAIN}; nearest-gauge assignment, city-mean fill for gauge gaps", "",
              "## Top 15 days by max hourly rainfall (mm/h, any gauge)", "",
              daily.sort_values("max_hourly_mm", ascending=False).head(15).to_markdown(floatfmt=".1f"), ""]

    # ---------- flood traces ----------
    grid_g = load_grid(with_geometry=True)[["cell_idx", "geometry", "dong_major_name"]].reset_index(drop=True)
    recs = []
    for y in (2022, 2023, 2024, 2025):
        shp = str(next((SRC / "unz" / f"trace_{y}").glob("*.shp")))   # pathlib: the folder name contains [ ]
        g = gpd.read_file(shp, encoding="cp949")
        if g.crs is None or g.crs.to_epsg() != 5179:
            g = g.set_crs(5179, allow_override=True)
        g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
        g["geometry"] = g.geometry.buffer(0)
        start = g["F_SAT_YMD"].astype(str).str.extract(r"(\d{8})")[0] if "F_SAT_YMD" in g else g["피해일시"].astype(str).str.extract(r"(\d{8})")[0]
        end = g["F_END_YMD"].astype(str).str.extract(r"(\d{8})")[0] if "F_END_YMD" in g else start
        event = g["F_DISA_NM"].astype(str) if "F_DISA_NM" in g else g.get("TYPE2", pd.Series("", index=g.index)).astype(str)
        depth = pd.to_numeric(g.get("F_SHIM"), errors="coerce")
        area = pd.to_numeric(g.get("F_AREA"), errors="coerce")
        cause = g["F_RSN_DTL"].astype(str) if "F_RSN_DTL" in g else ""
        g2 = gpd.GeoDataFrame({"year": y, "event": event.values, "start": start.values, "end": end.values, "depth_m": depth.values,
                               "area_m2": area.values, "cause": cause if isinstance(cause, str) else cause.values,
                               "zone": g.get("F_ZONE_NM", pd.Series("", index=g.index)).astype(str).values}, geometry=g.geometry.values, crs=5179)
        j = gpd.sjoin(g2, grid_g, how="inner", predicate="intersects")
        recs.append(pd.DataFrame(j.drop(columns="geometry")))
        lines.append(f"- flood traces {y}: {len(g)} polygons, {j.cell_idx.nunique()} cells touched")
    ft = pd.concat(recs, ignore_index=True)
    ft.to_parquet(OUT / "flood_trace_cells.parquet", index=False)
    ev = ft.groupby(["year", "event", "start"]).agg(n_polygons=("cell_idx", "size"), n_cells=("cell_idx", "nunique"),
                                                     area_m2=("area_m2", "sum"), max_depth_m=("depth_m", "max"),
                                                     dongs=("dong_major_name", lambda s: ", ".join(pd.unique(s.dropna())[:8]))).reset_index()
    ev = ev.sort_values(["year", "n_cells"], ascending=[True, False])
    ev.to_csv(OUT / "flood_events.csv", index=False, encoding="utf-8-sig")
    lines += ["", "## Flood events (year, event, start date) with the most cells", "",
              ev.sort_values("n_cells", ascending=False).head(20).to_markdown(index=False, floatfmt=".0f"), ""]

    # ---------- flood forecast contours ----------
    fc = []
    for shp in sorted(str(p) for p in (SRC / "unz" / "forecast").glob("*.shp")):
        g = gpd.read_file(shp)
        cls = re.search(r"DS_FLOODING_(\d)", shp).group(1)
        b = g.total_bounds
        fc.append({"file": shp.split("\\")[-1].split("/")[-1], "class": cls, "n": len(g), "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    fcd = pd.DataFrame(fc)
    # bounds (x 182-216 km, y 537-566 km) identify EPSG:5186; save the contours with that CRS for later use
    parts = []
    for shp in sorted(str(p) for p in (SRC / "unz" / "forecast").glob("*.shp")):
        g = gpd.read_file(shp).set_crs(5186, allow_override=True).to_crs(5179)
        g["depth_class"] = int(re.search(r"DS_FLOODING_(\d)", shp).group(1))
        parts.append(g[["depth_class", "geometry"]])
    gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=5179).to_file(OUT / "flood_forecast_contours.gpkg", layer="contours", driver="GPKG")
    lines.append("- flood-forecast contours saved as processed/hazard/flood_forecast_contours.gpkg (EPSG:5186 assumed, reprojected to 5179; depth_class 1..6 = <0.5m .. >3m)")
    lines += ["## Flood-forecast contour files (no CRS declared; bounds shown to identify the projection)", "", fcd.to_markdown(index=False, floatfmt=".0f"), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
