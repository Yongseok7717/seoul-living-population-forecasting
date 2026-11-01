"""Link the 긴급재난문자 event labels (data/disaster_events_seoul.csv) to the grid/time index as a prior channel.

    python build_disaster_prior.py

Output processed/priors/prior_dmsg_YYYYMM.npy  [T_month, N_all, 3] float16, 1 inside a message window:
  channel 0  any message (window = [first message, last message + 12 h]) for cells in the named 자치구 (all cells if 서울 전체)
  channel 1  weather types (호우·대설·한파·폭염·강풍·홍수·산사태·미세먼지·결빙)
  channel 2  crowd / protest / traffic types (집회·인파, 교통·열차, 도로통제, 교통통제, 교통사고)
Also processed/priors/disaster_prior_summary.md (coverage per year). Cells -> 자치구 via the cell-dong map (H_DNG_CD prefix).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import numpy as np
import pandas as pd
import config as C
from loader import load_grid, months_between

OUT = C.OUT / "priors"; OUT.mkdir(parents=True, exist_ok=True)
GU = {"11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구",
      "11320": "도봉구", "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구", "11530": "구로구", "11545": "금천구",
      "11560": "영등포구", "11590": "동작구", "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구"}
WEATHER = {"호우", "대설", "한파", "폭염", "강풍", "홍수", "산사태", "기타:폭염", "기타:결빙·한파", "미세먼지"}
CROWD = {"기타:집회·인파", "기타:교통·열차", "기타:도로통제", "교통통제", "교통사고", "교통"}


def main():
    grid = load_grid(with_geometry=False); N = len(grid)
    dmap = pd.read_parquet(C.OUT / "grid" / "cell_dong_map.parquet").sort_values("share").groupby("cell_idx").last()
    gu_of = dmap["H_DNG_CD"].astype(str).str[:5].map(GU).reindex(range(N)).fillna("").values
    cells_by_gu = {g: np.where(gu_of == g)[0] for g in GU.values()}
    ev = pd.read_csv(C.TABLES / "disaster_events_seoul.csv"); ev["t0"] = pd.to_datetime(ev.first_ts); ev["t1"] = pd.to_datetime(ev.last_ts) + pd.Timedelta(hours=12)
    ev["ch"] = np.where(ev.type.isin(WEATHER), 1, np.where(ev.type.isin(CROWD), 2, 0))
    rows = []
    for m in months_between("202301", "202601"):
        p = C.OUT / "tensors" / "local" / f"time_index_{m}.parquet"
        if not p.exists():
            continue
        ti = pd.read_parquet(p); tt = pd.to_datetime(ti.datetime).values; P = np.zeros((len(ti), N, 3), np.float16)
        sub = ev[(ev.t1 >= tt[0]) & (ev.t0 <= tt[-1])]
        for e in sub.itertuples():
            h = (tt >= np.datetime64(e.t0.floor("h"))) & (tt <= np.datetime64(e.t1))
            cells = np.arange(N) if e.gu == "ALL" else np.concatenate([cells_by_gu.get(g, np.array([], int)) for g in str(e.gu).split(",")])
            if h.any() and len(cells):
                P[np.ix_(np.where(h)[0], cells, [0])] = 1
                if e.ch:
                    P[np.ix_(np.where(h)[0], cells, [e.ch])] = 1
        np.save(OUT / f"prior_dmsg_{m}.npy", P)
        rows.append({"month": m, "events": len(sub), "hours_any": int((P[:, :, 0].sum(1) > 0).sum()), "cell_hour_share_any": float(P[:, :, 0].mean()), "share_weather": float(P[:, :, 1].mean()), "share_crowd": float(P[:, :, 2].mean())})
        print(m, rows[-1], flush=True)
    df = pd.DataFrame(rows); df["year"] = df.month.str[:4]
    L = ["# 긴급재난문자 사전 채널 (prior_dmsg_YYYYMM.npy, [T, N_all, 3])", "", "- 창 = [첫 발령 시각, 마지막 발령 + 12 h]; 셀 = 문자에 명시된 자치구의 격자(시 전체 발령은 전 격자). 채널: 0 전체, 1 기상 유형, 2 집회·인파·교통 유형.", "",
         "| 연도 | 사건 수 | 창이 있는 시간 비율 | 셀-시간 비율(전체) | 기상 | 집회·교통 |", "|---|---|---|---|---|---|"]
    for y, d in df.groupby("year"):
        L.append(f"| {y} | {int(d.events.sum())} | {d.hours_any.sum() / (len(d) * 720):.3f} | {d.cell_hour_share_any.mean():.4f} | {d.share_weather.mean():.4f} | {d.share_crowd.mean():.4f} |")
    (OUT / "disaster_prior_summary.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
