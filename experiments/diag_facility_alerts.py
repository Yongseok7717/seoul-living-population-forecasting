"""Data-only check: how does the population of facility cells change during emergency alerts?

For every alert event (data/disaster_events_seoul.csv; window = [first message, last message + 12 h]; cells of the named districts, or all
cells for city-wide alerts) we compute the mean relative residual (X - B) / B over the window for several cell classes:
all cells of the alerted districts, cells with an emergency room, a hospital, a clinic, an earthquake shelter (indoor/outdoor) or a
civil-defence shelter.  The same quantity is computed on placebo windows shifted by +-1..4 weeks (same weekday, same hours) and the
difference event - placebo is reported by alert type.  Crowd/rally alerts are included as a positive control.
Outputs results/diag_propagation/facility_alerts.csv (+ per-event rows in facility_alerts_events.csv)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "diag_propagation"; sys.path.insert(0, str(SRC))
import config as C                                   # noqa: E402
from loader import load_pop10, months_between, load_grid   # noqa: E402

GROUPS = {"폭염 (heat)": {"폭염", "기타:폭염"}, "호우·홍수·산사태 (rain, flood)": {"호우", "홍수", "산사태"}, "대설·한파 (snow, cold)": {"대설", "한파", "기타:결빙·한파"}, "강풍·미세먼지 (wind, dust)": {"강풍", "미세먼지"},
          "화재·산불 (fire)": {"화재", "산불"}, "집회·인파 (rally, crowd; control)": {"기타:집회·인파"}, "교통 (traffic)": {"기타:교통·열차", "기타:도로통제", "교통사고", "교통통제", "교통"}}
GU = {"11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구", "11410": "서대문구",
      "11440": "마포구", "11470": "양천구", "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구", "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구"}


def main():
    months = months_between("202309", "202512"); X, ti = load_pop10(months); X = X.astype(np.float32); T, N = X.shape
    B = np.concatenate([np.load(C.OUT / "residual" / f"baseline_{m}.npy") for m in months]).astype(np.float32); assert B.shape == X.shape
    tt = pd.to_datetime(ti.datetime).values; rel = np.where(B > 20, (X - B) / np.maximum(B, 1), np.nan)      # relative residual, cells with a meaningful baseline
    dmap = pd.read_parquet(C.OUT / "grid" / "cell_dong_map.parquet").sort_values("share").groupby("cell_idx").last(); gu_of = dmap["H_DNG_CD"].astype(str).str[:5].map(GU).reindex(range(N)).fillna("").values
    fc = pd.read_parquet(C.OUT / "facilities" / "facility_counts_by_cell.parquet").set_index("cell_idx").reindex(range(N)).fillna(0)
    classes = {"all cells": np.ones(N, bool), "emergency room": fc.n_emergency.values > 0, "hospital": fc.n_hospital.values > 0, "clinic": fc.n_clinic.values > 0, "earthquake shelter": (fc.n_shelter_eq_indoor.values + fc.n_shelter_eq_outdoor.values) > 0, "civil-defence shelter": fc.n_shelter_civil_defense.values > 0}
    ev = pd.read_csv(C.TABLES / "disaster_events_seoul.csv", encoding="utf-8-sig"); ev["t0"] = pd.to_datetime(ev.first_ts).dt.floor("h"); ev["t1"] = pd.to_datetime(ev.last_ts) + pd.Timedelta(hours=12)
    rows = []
    for e in ev.itertuples():
        grp = next((g for g, s in GROUPS.items() if e.type in s), None)
        if grp is None:
            continue
        h = np.where((tt >= np.datetime64(e.t0)) & (tt <= np.datetime64(e.t1)))[0]
        if len(h) == 0:
            continue
        cells = np.arange(N) if e.gu == "ALL" else np.where(np.isin(gu_of, str(e.gu).split(",")))[0]
        if len(cells) == 0:
            continue
        for cname, cm in classes.items():
            cc = cells[cm[cells]]
            if len(cc) == 0:
                continue
            v_ev = np.nanmean(rel[np.ix_(h, cc)]); pl = []
            for sh in (-28, -21, -14, -7, 7, 14, 21, 28):
                hh = h + sh * 24; hh = hh[(hh >= 0) & (hh < T)]
                if len(hh) == len(h):
                    pl.append(np.nanmean(rel[np.ix_(hh, cc)]))
            if pl:
                rows.append({"group": grp, "type": e.type, "date": e.date, "gu": e.gu, "citywide": e.gu == "ALL", "hours": len(h), "class": cname, "n_cells": len(cc), "rel_event": v_ev, "rel_placebo": float(np.mean(pl)), "diff": v_ev - float(np.mean(pl))})
    d = pd.DataFrame(rows); d.to_csv(OUT / "facility_alerts_events.csv", index=False, encoding="utf-8-sig")
    g = d.groupby(["group", "class"]).agg(n_events=("date", "nunique"), rel_event=("rel_event", "mean"), rel_placebo=("rel_placebo", "mean"), diff_mean=("diff", "mean"), diff_median=("diff", "median"), share_positive=("diff", lambda s: float((s > 0).mean()))).reset_index()
    g.to_csv(OUT / "facility_alerts.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 220); print(g.round(4).to_string()); print("ALLDONE")


if __name__ == "__main__":
    main()
