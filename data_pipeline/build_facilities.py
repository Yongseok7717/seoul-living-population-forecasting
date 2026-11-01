"""Harmonise the downloaded facility files and attach them to the standard 250m grid.

Inputs  (생활인구/[서울열린데이터 광장]/시설/):
  OA-21063_지진옥외대피소.csv            lon/lat (WGS84)             -> shelter_eq_outdoor
  OA-21060_지진대피소현황.csv            lon/lat                     -> shelter_eq_indoor
  민방위대피시설_자치구/OA-200xx_*.csv   좌표정보(X)=lat, (Y)=lon     -> shelter_civil_defense   (사용중 only)
  OA-16479_병원인허가.csv                좌표정보 EPSG:5174          -> hospital                (영업/정상 only)
  OA-16480_의원인허가.csv                좌표정보 EPSG:5174, 병상수  -> clinic                  (영업/정상 only)
  datago_15021148_...csv                 lat/lon, 시도교육청명=서울   -> school_elem / school_mid / school_high (운영 only)
  datago_15114136_...csv                 lat/lon, 주소 서울          -> senior_center (경로당)
  OA-22198_노인의료복지시설현황.xlsx     address only (no coords)    -> nursing_home (cell_idx NaN until geocoded)

Outputs (processed/facilities/):
  facilities.parquet / facilities.gpkg   one row per facility: ftype, subtype, name, address, x, y (EPSG:5179), cell_idx, capacity fields
  facility_counts_by_cell.parquet        cell_idx x ftype counts, plus shelter area, clinic beds, nursing capacity
  facilities_summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
import config as C
from loader import load_grid, load_grid_meta

SRC = C.OPEN_DIR / "시설"
OUT = C.OUT / "facilities"
T_WGS = Transformer.from_crs("EPSG:4326", f"EPSG:{C.CRS_STANDARD}", always_xy=True)
T_5174 = Transformer.from_crs("EPSG:5174", f"EPSG:{C.CRS_STANDARD}", always_xy=True)


def rd(name, enc="cp949"):
    return pd.read_csv(SRC / name, encoding=enc, dtype=str)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def frame(ftype, subtype, name, address, x, y, **extra):
    d = pd.DataFrame({"ftype": ftype, "subtype": subtype, "name": name, "address": address, "x": x, "y": y})
    for k, v in extra.items():
        d[k] = v
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts, notes = [], []

    a = rd("OA-21063_지진옥외대피소.csv")
    x, y = T_WGS.transform(num(a["경도"]).values, num(a["위도"]).values)
    parts.append(frame("shelter_eq_outdoor", "지진옥외대피장소", a["수용시설명"], a["상세주소"], x, y,
                       area_m2=num(a["시설면적"]), source="OA-21063"))

    b = rd("OA-21060_지진대피소현황.csv")
    x, y = T_WGS.transform(num(b["경도"]).values, num(b["위도"]).values)
    parts.append(frame("shelter_eq_indoor", b["구분명"], b["시설명"], b["상세주소"], x, y,
                       area_m2=num(b["시설면적"]), source="OA-21060"))

    # note: the folder name contains [ ], so glob.glob on the full path would treat it as a character class
    cd = pd.concat([pd.read_csv(p, encoding="cp949", dtype=str).assign(src=p.name)
                    for p in sorted((SRC / "민방위대피시설_자치구").glob("*.csv"))], ignore_index=True)
    notes.append(f"민방위대피시설 raw {len(cd)} rows, status {cd['영업상태명'].value_counts().to_dict()}")
    cd = cd[cd["영업상태명"] == "사용중"]
    x, y = T_WGS.transform(num(cd["좌표정보(Y)"]).values, num(cd["좌표정보(X)"]).values)   # columns are swapped in source
    parts.append(frame("shelter_civil_defense", "민방위대피시설", cd["사업장명"], cd["도로명주소"], x, y,
                       source=cd["src"].str.slice(0, 8)))

    h = rd("OA-16479_병원인허가.csv")
    notes.append(f"병원 raw {len(h)} rows, status {h['영업상태명'].value_counts().to_dict()}, coords missing {int(num(h['좌표정보(X)']).isna().sum())}")
    h = h[h["영업상태명"] == "영업/정상"]
    x, y = T_5174.transform(num(h["좌표정보(X)"]).values, num(h["좌표정보(Y)"]).values)
    parts.append(frame("hospital", h["업태구분명"], h["사업장명"], h["도로명주소"], x, y,
                       area_m2=num(h["소재지면적"]), source="OA-16479"))

    c = rd("OA-16480_의원인허가.csv")
    notes.append(f"의원 raw {len(c)} rows, status {c['영업상태명'].value_counts().to_dict()}, coords missing {int(num(c['좌표정보(X)']).isna().sum())}")
    c = c[c["영업상태명"] == "영업/정상"]
    x, y = T_5174.transform(num(c["좌표정보(X)"]).values, num(c["좌표정보(Y)"]).values)
    parts.append(frame("clinic", c["업태구분명"], c["사업장명"], c["도로명주소"], x, y,
                       beds=num(c["병상수"]), staff_medical=num(c["의료인수"]), departments=c["진료과목내용명"], source="OA-16480"))

    s = rd("datago_15021148_전국초중등학교위치표준데이터.csv", "utf-8-sig")
    s = s[(s["시도교육청명"].str.contains("서울", na=False)) & (s["운영상태"] == "운영")]
    x, y = T_WGS.transform(num(s["경도"]).values, num(s["위도"]).values)
    ft = s["학교급구분"].map({"초등학교": "school_elem", "중학교": "school_mid", "고등학교": "school_high"}).fillna("school_other")
    parts.append(frame(ft.values, s["학교급구분"], s["학교명"], s["소재지도로명주소"], x, y,
                       school_id=s["학교ID"], source="datago_15021148"))

    # senior centres (경로당): the Seoul-wide list OA-15052 (3,644, addresses only) is the master; coordinates are
    # borrowed from the national standard data (10 districts) by name or road address, the rest go to geocoding
    k = rd("datago_15114136_전국마을회관및경로당표준데이터.csv", "utf-8-sig")
    k = k[k["소재지도로명주소"].fillna(k["소재지지번주소"]).str.startswith("서울", na=False) & (k["영업상태명"].fillna("영업") != "폐업")].copy()
    full = rd("OA-15052_서울시경로당정보.csv")
    def norm(x):
        return x.fillna("").str.replace(r"\s+|\(.*?\)|경로당|아파트|apt|APT|\.|,", "", regex=True).str.lower()
    def norm_addr(x):
        return x.fillna("").str.replace(r"\s+", "", regex=True).str.replace("서울특별시", "").str.replace("서울", "")
    k["gu"] = k["소재지도로명주소"].str.split().str[1]
    k["key_name"] = k["gu"] + "|" + norm(k["시설명"]); k["key_addr"] = norm_addr(k["소재지도로명주소"])
    full["key_name"] = full["시군구명"].fillna("") + "|" + norm(full["시설명(경로당명)"]); full["key_addr"] = norm_addr(full["주소(도로명)"])
    m1 = full.merge(k[["key_name", "위도", "경도"]].drop_duplicates("key_name"), on="key_name", how="left")
    m2 = full.merge(k[["key_addr", "위도", "경도"]].drop_duplicates("key_addr"), on="key_addr", how="left")
    lat = num(m1["위도"]).fillna(num(m2["위도"])); lon = num(m1["경도"]).fillna(num(m2["경도"]))
    x, y = T_WGS.transform(lon.values, lat.values)
    parts.append(frame("senior_center", "경로당", full["시설명(경로당명)"], "서울특별시 " + full["주소(도로명)"].fillna(""), x, y,
                       district=full["시군구명"], source="OA-15052+datago_15114136"))
    notes.append(f"경로당: master list OA-15052 {len(full)} rows; coordinates matched from the national standard data for {int(np.isfinite(x).sum())}, the rest need geocoding")

    w = rd("OA-20412_노인여가복지시설목록.csv")
    w = w[w["시설종류명(시설유형)"].str.contains("노인복지관", na=False)]
    parts.append(frame("senior_welfare_center", "노인복지관", w["시설명"], w["시설주소"], np.nan, np.nan, district=w["시군구명"], source="OA-20412"))
    notes.append(f"노인복지관 (OA-20412): {len(w)} rows, addresses only -> geocoding")

    # transport (static features / mobility hubs)
    st = rd("../교통/OA-21232_서울시역사마스터정보.csv")
    x, y = T_WGS.transform(num(st["경도"]).values, num(st["위도"]).values)
    parts.append(frame("subway_station", st["호선"], st["역사명"], "", x, y, source="OA-21232"))
    bs = rd("../교통/OA-15067_서울시버스정류소위치정보.csv")
    x, y = T_WGS.transform(num(bs["X좌표"]).values, num(bs["Y좌표"]).values)
    parts.append(frame("bus_stop", bs["정류소 타입"], bs["정류소명"], "", x, y, source="OA-15067"))

    # emergency medical institutions from the data.go.kr API (optional; produced by fetch_emergency_api.py)
    em_path = SRC / "datago_emergency_seoul.csv"
    if em_path.exists():
        em = pd.read_csv(em_path, dtype=str, encoding="utf-8-sig")
        x, y = T_WGS.transform(num(em["wgs84Lon"]).values, num(em["wgs84Lat"]).values)
        parts.append(frame("emergency", em.get("dutyEmclsName", pd.Series("", index=em.index)).fillna(""), em["dutyName"], em["dutyAddr"], x, y,
                           er_beds=num(em.get("hperyn")), beds=num(em.get("hpbdn")), hpid=em["hpid"], source="datago_B552657"))
        notes.append(f"응급의료기관 (API): {len(em)} rows, ER beds known for {int(num(em.get('hperyn')).notna().sum())}")
    else:
        notes.append("응급의료기관: datago_emergency_seoul.csv not found (run fetch_emergency_api.py with DATAGO_KEY)")

    # nursing homes: xlsx with merged header rows; addresses only
    nh = []
    for sheet, sub in (("시트1 노인요양시설(241)", "노인요양시설"), ("시트2 노인요양공동생활가정시설(246)", "노인요양공동생활가정")):
        d = pd.read_excel(SRC / "OA-22198_노인의료복지시설현황.xlsx", sheet_name=sheet, header=None, skiprows=4, dtype=str)
        d = d[d[3].notna() & d[0].notna()]
        nh.append(frame("nursing_home", sub, d[3], d[32], np.nan, np.nan,
                        capacity=num(d[8]), occupancy=num(d[9]), district=d[1], source="OA-22198"))
    parts += nh
    # subway stations outside the grid (other cities on the same lines) are dropped by the grid snap below

    fac = pd.concat(parts, ignore_index=True)
    fac["x"] = num(fac["x"]); fac["y"] = num(fac["y"])
    fac["gkey"] = fac.ftype + "|" + fac.name.fillna("").astype(str) + "|" + fac.address.fillna("").astype(str)
    geo_path = OUT / "geocoded.csv"
    if geo_path.exists():
        geo = pd.read_csv(geo_path, dtype=str).dropna(subset=["lon", "lat"]).drop_duplicates("key")
        gx, gy = T_5174.transform(np.zeros(0), np.zeros(0)) if False else T_WGS.transform(num(geo.lon).values, num(geo.lat).values)
        lut_g = dict(zip(geo.key, zip(gx, gy)))
        need = fac.x.isna() & fac.gkey.isin(lut_g)
        fac.loc[need, "x"] = [lut_g[k][0] for k in fac.loc[need, "gkey"]]
        fac.loc[need, "y"] = [lut_g[k][1] for k in fac.loc[need, "gkey"]]
        notes.append(f"geocoded.csv applied to {int(need.sum())} facilities")
    todo = fac[fac.x.isna() & fac.address.fillna("").astype(str).str.strip().ne("")][["gkey", "ftype", "name", "address"]].rename(columns={"gkey": "key"})
    todo.drop_duplicates("key").to_csv(OUT / "to_geocode.csv", index=False, encoding="utf-8-sig")
    notes.append(f"addresses still needing geocoding: {todo.key.nunique()} -> processed/facilities/to_geocode.csv (run geocode_addresses.py with a VWorld key)")

    # snap to the lattice
    meta = load_grid_meta()
    grid = load_grid(with_geometry=False)
    x0, y0 = meta["origin_xy_sw_centroid"]
    col = np.rint((fac.x - x0) / C.CELL_SIZE); row = np.rint((fac.y - y0) / C.CELL_SIZE)
    lut = {(r, c): i for i, r, c in zip(grid.cell_idx, grid.row, grid.col)}
    fac["cell_idx"] = [lut.get((int(r), int(c))) if np.isfinite(r) and np.isfinite(c) else None for r, c in zip(row, col)]
    fac["cell_idx"] = fac["cell_idx"].astype("Int64")
    fac["has_coords"] = fac.x.notna()
    fac["in_grid"] = fac.cell_idx.notna()
    fac["dong_major_name"] = fac.cell_idx.map(grid.dong_major_name)
    fac.insert(0, "fid", np.arange(len(fac)))
    fac.to_parquet(OUT / "facilities.parquet", index=False)
    g = gpd.GeoDataFrame(fac[fac.has_coords], geometry=gpd.points_from_xy(fac.loc[fac.has_coords, "x"], fac.loc[fac.has_coords, "y"]),
                         crs=f"EPSG:{C.CRS_STANDARD}")
    g.to_file(OUT / "facilities.gpkg", layer="facilities", driver="GPKG", index=False)

    # per-cell counts and capacities
    f = fac[fac.in_grid]
    counts = f.pivot_table(index="cell_idx", columns="ftype", values="fid", aggfunc="count", fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]
    extra = pd.DataFrame({
        "shelter_eq_outdoor_area_m2": f[f.ftype == "shelter_eq_outdoor"].groupby("cell_idx").area_m2.sum(),
        "shelter_eq_indoor_area_m2": f[f.ftype == "shelter_eq_indoor"].groupby("cell_idx").area_m2.sum(),
        "clinic_beds": f[f.ftype == "clinic"].groupby("cell_idx").beds.sum(),
        "n_hospital_general": f[(f.ftype == "hospital") & (f.subtype == "종합병원")].groupby("cell_idx").fid.count(),
        "n_hospital_nursing": f[(f.ftype == "hospital") & f.subtype.str.startswith("요양병원")].groupby("cell_idx").fid.count(),
        "er_beds": f[f.ftype == "emergency"].groupby("cell_idx").er_beds.sum() if "er_beds" in f else pd.Series(dtype=float),
    })
    cell = pd.DataFrame({"cell_idx": grid.cell_idx.values}).set_index("cell_idx").join(counts).join(extra).fillna(0).reset_index()
    cell.to_parquet(OUT / "facility_counts_by_cell.parquet", index=False)

    # sanity: well-known hospitals should land in the right dong
    chk = fac[fac.name.fillna("").str.contains("서울대학교병원|연세대학교의과대학세브란스병원|삼성서울병원|서울아산병원", regex=True) & (fac.ftype == "hospital")]
    lines = ["# Facilities summary", "", "## Counts by type (all / with coordinates / inside grid)", "",
             "| ftype | all | coords | in grid |", "|---|---|---|---|"]
    for t, gdf in fac.groupby("ftype"):
        lines.append(f"| {t} | {len(gdf)} | {int(gdf.has_coords.sum())} | {int(gdf.in_grid.sum())} |")
    lines += ["", "## Subtypes", ""]
    for t in ["hospital", "clinic", "emergency", "shelter_eq_indoor", "senior_center", "senior_welfare_center", "nursing_home", "subway_station", "bus_stop"]:
        lines.append(f"- {t}: {fac[fac.ftype == t].subtype.value_counts().head(8).to_dict()}")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    lines += ["", "## Sanity check: well-known hospitals -> dominant dong of the snapped cell", "",
              chk[["name", "subtype", "dong_major_name"]].to_markdown(index=False) if len(chk) else "(none matched)", "",
              "## Cells with at least one facility", "",
              f"- {int((cell.filter(like='n_').sum(axis=1) > 0).sum())} of {len(cell)} cells", ""]
    (OUT / "facilities_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
