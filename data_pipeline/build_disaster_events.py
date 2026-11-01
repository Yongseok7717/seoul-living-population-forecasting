"""Seoul 긴급재난문자 -> event label table for the incident catalogue.

Input  data/disaster_msgs_seoul.csv (fetch_disaster_msgs.py)
Output data/disaster_events_seoul.csv  one row per (date, disaster type) with first/last message time, number of
       messages, 자치구 list ('ALL' = city-wide), share of 위급/긴급 (non-안전안내) steps, first message text.
       Missing-person notices (DST_SE_NM 기타 with 찾습니다/실종) are dropped: they are not population-relevant events.
       data/disaster_events_seoul_summary.md  counts by type and by year, top event days by message count.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import re
import pandas as pd
import config as C

DATA = C.TABLES
s = pd.read_csv(DATA / "disaster_msgs_seoul.csv")
miss = s.MSG_CN.str.contains("찾습니다|실종", na=False) & (s.DST_SE_NM == "기타")
s = s[~miss].copy(); s["ts"] = pd.to_datetime(s.ts)
# sub-type for 기타 by keywords (집회/인파, 풍선, 교통/열차, 안전수칙 등)
def sub(m):
    m = str(m)
    for pat, lab in [("집회|시위|인파|무정차", "집회·인파"), ("풍선", "대남풍선"), ("열차|지하철|호선|운행", "교통·열차"), ("차단|통제|고속도로", "도로통제"),
                     ("결빙|한파|기온", "결빙·한파"), ("폭염|무더위|온열", "폭염"), ("물놀이|수상", "수상안전"), ("정전|단수|가스", "기반시설")]:
        if re.search(pat, m):
            return lab
    return "기타"
s["type"] = s.apply(lambda r: r.DST_SE_NM if r.DST_SE_NM != "기타" else "기타:" + sub(r.MSG_CN), axis=1)
g = s.groupby(["date", "type"])
ev = g.agg(first_ts=("ts", "min"), last_ts=("ts", "max"), n_msgs=("ts", "size"),
           n_citywide=("gu", lambda x: int((x == "ALL").sum())),
           gu=("gu", lambda x: "ALL" if (x == "ALL").any() else ",".join(sorted({q for v in x for q in str(v).split(",")}))),
           urgent_share=("EMRG_STEP_NM", lambda x: round(float((x != "안전안내").mean()), 3)),
           first_msg=("MSG_CN", "first")).reset_index()
ev["n_gu"] = ev.gu.apply(lambda x: 25 if x == "ALL" else len(x.split(",")))
ev = ev.sort_values(["date", "first_ts"]); ev.to_csv(DATA / "disaster_events_seoul.csv", index=False, encoding="utf-8-sig")
L = ["# 서울 재난문자 사건 라벨 표 (실종안내 제외)", "", f"- 문자 {len(s):,}건 → 사건(날짜×유형) {len(ev):,}건, 기간 {ev.date.min()}~{ev.date.max()}", "",
     "## 유형별 사건 수", "", "| 유형 | 사건 수 | 문자 수 | 시 전체 발령 사건 |", "|---|---|---|---|"]
for t, d in ev.groupby("type"):
    L.append(f"| {t} | {len(d)} | {d.n_msgs.sum()} | {(d.gu == 'ALL').sum()} |")
L += ["", "## 연도별 사건 수", "", "| 연도 | 사건 수 | 문자 수 |", "|---|---|---|"] + [f"| {y} | {len(d)} | {d.n_msgs.sum()} |" for y, d in ev.groupby(ev.date.str[:4])]
top = ev.sort_values("n_msgs", ascending=False).head(25)
L += ["", "## 문자 수 상위 사건일", "", "| 날짜 | 유형 | 문자 | 구 수 | 첫 발령 | 첫 문자 |", "|---|---|---|---|---|---|"]
L += [f"| {r.date} | {r.type} | {r.n_msgs} | {r.n_gu} | {r.first_ts:%H:%M} | {str(r.first_msg)[:60].replace('|', '/')} |" for r in top.itertuples()]
(DATA / "disaster_events_seoul_summary.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
