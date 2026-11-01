"""External validation of the top-40 localized events of 2025 (events_top200_localized_2025.csv), automatic part:
calendar (holiday / bridge / year-end), venue schedule (data/venue_events.csv, venue -> nearby dongs), disaster messages
(집회·인파 / 호우 etc. on the same day and gu), flood traces.  Writes processed/eda/events/events_validation_2025_auto.csv
with an 'auto_category' + evidence; the remaining 'unknown' rows are checked by web search (events_validation_2025.md)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys
from pathlib import Path
import pandas as pd
import config as C   # noqa: E402
from build_baseline_atlas import HOLIDAY_SET   # noqa: E402

EV = C.OUT / "eda" / "events"; DATA = C.TABLES
VENUE_DONGS = {"잠실야구장": {"잠실2동", "잠실3동", "잠실6동", "잠실7동"}, "잠실종합운동장": {"잠실2동", "잠실3동", "잠실6동", "잠실7동"}, "KSPO돔(올림픽공원)": {"방이2동", "오륜동", "방이1동", "잠실4동"},
               "서울월드컵경기장": {"성산제2동", "상암동"}, "광화문광장": {"사직동", "종로1·2·3·4가동", "청운효자동"}, "여의도한강공원": {"여의동"}, "여의도국회": {"여의동", "영등포동"}}
BRIDGE = {"20250131": "설 연휴(1/27 임시공휴일~1/30) 다음날 금요일: 업무지구 휴가", "20251226": "성탄절 다음날 금요일: 업무지구 휴가", "20251231": "연말 마지막 영업일", "20251010": "개천절·한글날 사이 금요일",
          "20250502": "근로자의 날·어린이날 사이", "20251006": "추석 연휴", "20251007": "추석 연휴", "20251003": "개천절", "20251009": "한글날", "20251008": "추석 대체공휴일", "20250606": "현충일", "20250815": "광복절", "20250301": "삼일절", "20250303": "삼일절 대체공휴일", "20250505": "어린이날·부처님오신날", "20250506": "대체공휴일", "20251225": "성탄절", "20250603": "대통령 선거일"}


def main():
    e = pd.read_csv(EV / "events_top200_localized_2025.csv", encoding="utf-8-sig").head(40).copy()
    ven = pd.read_csv(DATA / "venue_events.csv", encoding="utf-8-sig"); dm = pd.read_csv(DATA / "disaster_events_seoul.csv")
    fl = pd.read_csv(C.OUT / "hazard" / "flood_events.csv", encoding="utf-8-sig") if (C.OUT / "hazard" / "flood_events.csv").exists() else pd.DataFrame()
    rows = []
    for r in e.itertuples():
        ymd = r.date.replace("-", ""); dongs = {d.strip() for d in str(r.dongs).split(",")} | {r.dong}; cat, ev = "unknown", ""
        if ymd in HOLIDAY_SET or ymd in BRIDGE:
            cat = "holiday/calendar"; ev = BRIDGE.get(ymd, "공휴일")
            if r.direction == "decrease":
                cat = "calendar (drop in office district)"
        v = ven[ven.date == r.date]; hits = [f"{x.venue}: {x.note} ({x.type}, {x.start_hour}-{x.end_hour}h)" for x in v.itertuples() if VENUE_DONGS.get(x.venue, set()) & dongs]
        if hits:
            cat = "venue schedule" if cat == "unknown" else cat + " + venue"; ev = (ev + "; " if ev else "") + " | ".join(hits[:3])
        d = dm[(dm.date == r.date) & (dm.type.isin(["기타:집회·인파", "교통통제", "기타:도로통제", "호우", "대설", "태풍", "폭염"]))]
        d = d[d.gu.apply(lambda g: g == "ALL" or r.gu in str(g))]
        if len(d):
            ev = (ev + "; " if ev else "") + "재난문자: " + "; ".join(f"{x.type} {str(x.first_msg)[:40]}" for x in d.head(2).itertuples())
            if cat == "unknown" and d.type.str.contains("집회|통제").any():
                cat = "protest/traffic (disaster msg)"
        if len(fl) and (fl.start.astype(str) == ymd).any():
            ev = (ev + "; " if ev else "") + "침수흔적 사건일"; cat = "flood" if cat == "unknown" else cat
        rows.append({"rank": r.Index + 1, "event_id": r.event_id, "date": r.date, "area": f"{r.gu} {r.dong} ({r.dongs})", "dir": "+" if r.direction == "increase" else "−", "excess_k": round(r.sum_excess / 1e3), "n_cells": r.n_cells, "peak": r.peak_time[-5:], "auto_category": cat, "evidence": ev})
    out = pd.DataFrame(rows); out.to_csv(EV / "events_validation_2025_auto.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(max_colwidth=70)); print("\nunknown:", (out.auto_category == "unknown").sum())


if __name__ == "__main__":
    main()
