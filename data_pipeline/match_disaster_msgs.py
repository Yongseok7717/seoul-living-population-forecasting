"""Match the 긴급재난문자 event labels (data/disaster_events_seoul.csv) against the anomaly-event catalogue
(processed/eda/events/events_all_<years>.csv, |z| > 3 connected components) in both directions.

A. message event -> catalogue: for each (date, type) message event, the strongest catalogue event whose gu is among the
   named 자치구 (any gu if city-wide) and whose span overlaps [first message - 3 h, last message + 12 h].
   "hit" = that event's sum_abs_excess >= the catalogue's top-1% threshold (q99); "hit90" = >= q90.
   Placebo: the same gu / hour window shifted by +-7, 14, 21, 28 days (same weekday, 8 placebo windows).
   Paired statistics: base rate = placebo hit rate; percentile = share of the 8 placebo values below the real value
   (null mean 0.5) for the window maximum |excess|, the city day total |excess| and the day's drop share.
   Lag = catalogue peak time - first message time (hours; negative = the anomaly peaked before the first message).
B. city day level: same percentiles pooled by weather / non-weather message types.
C. catalogue -> messages: top-200 events (2023-09 onward) and the externally validated top-40 (categories from
   events_validation_2023_2024.md): any Seoul message (non missing-person) within +-12 h of the event span in the
   same gu or city-wide?  Coverage vs the same placebo shifts.
Outputs: data/disaster_events_seoul_matched.csv, processed/eda/events/disaster_match_summary.md, fig_disaster_match.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import config as C   # noqa: E402

EV = C.OUT / "eda" / "events"; DATA = C.TABLES
SHIFTS = [-28, -21, -14, -7, 7, 14, 21, 28]
WEATHER = {"호우", "대설", "한파", "폭염", "강풍", "홍수", "산사태", "기타:폭염", "기타:결빙·한파", "미세먼지"}


def load_catalogue():
    parts = []
    for f in sorted(EV.glob("events_all_*.csv")):
        e = pd.read_csv(f, low_memory=False, encoding="utf-8-sig"); e["src"] = f.stem.replace("events_all_", ""); parts.append(e)
    e = pd.concat(parts, ignore_index=True)
    e["t0"] = pd.to_datetime(e.date + " " + e.start); e["t1"] = pd.to_datetime(e.end_date + " " + e.end); e["tpk"] = pd.to_datetime(e.peak_time)
    return e


class ByGu:
    def __init__(self, df):
        df = df.sort_values("t0"); self.df = df; self.t0v = df.t0.values


def best_in_window(cat_by_gu, gus, w0, w1):
    """largest-|excess| catalogue event overlapping [w0, w1] in any of gus (None = all)"""
    best = None
    for g in (gus if gus is not None else cat_by_gu.keys()):
        c = cat_by_gu.get(g)
        if c is None:
            continue
        lo, hi = np.searchsorted(c.t0v, np.datetime64(w0 - pd.Timedelta(hours=24))), np.searchsorted(c.t0v, np.datetime64(w1), side="right")
        s = c.df.iloc[lo:hi]; s = s[(s.t0 <= w1) & (s.t1 >= w0)]
        if len(s):
            r = s.iloc[s.sum_abs_excess.values.argmax()]
            if best is None or r.sum_abs_excess > best.sum_abs_excess:
                best = r
    return best


def pct(x, pl):
    pl = [p for p in pl if not (isinstance(p, float) and np.isnan(p))]
    return float(np.mean([p < x for p in pl])) if pl and not np.isnan(x) else np.nan


def main():
    cat = load_catalogue(); years = sorted(cat.src.unique()); q99, q90 = cat.sum_abs_excess.quantile([.99, .90])
    days_cat = set(cat.date.unique()); cat_by_gu = {g: ByGu(d) for g, d in cat.groupby("gu")}
    daily = cat.groupby("date").agg(tot=("sum_abs_excess", "sum"), dropx=("sum_excess", lambda s: -s[s < 0].sum()))
    daily["dshare"] = daily.dropx / daily.tot.clip(lower=1)
    de = pd.read_csv(DATA / "disaster_events_seoul.csv"); de["first_ts"] = pd.to_datetime(de.first_ts); de["last_ts"] = pd.to_datetime(de.last_ts)
    de = de[(de.date >= min(days_cat)) & (de.date <= max(days_cat))].copy()
    print(f"catalogue {len(cat):,} events ({', '.join(years)}), q99 |excess| {q99:,.0f}, q90 {q90:,.0f}; message events in range {len(de)}")

    # ---- A: message -> catalogue, with placebo shifts ----
    rows = []
    for r in de.itertuples():
        gus = None if r.gu == "ALL" else r.gu.split(",")
        rec = {"date": r.date, "type": r.type, "gu": r.gu, "n_gu": r.n_gu, "n_msgs": r.n_msgs, "first_ts": r.first_ts, "cat_excess": np.nan, "hit": np.nan, "hit90": np.nan}
        pl = {"hit": [], "hit90": [], "ex": [], "tot": [], "dsh": []}
        for sh in [0] + SHIFTS:
            w0, w1 = r.first_ts + pd.Timedelta(days=sh) - pd.Timedelta(hours=3), r.last_ts + pd.Timedelta(days=sh) + pd.Timedelta(hours=12)
            d = (r.first_ts + pd.Timedelta(days=sh)).strftime("%Y-%m-%d")
            if d not in days_cat:            # day without catalogue coverage (missing / anomalous day)
                continue
            b = best_in_window(cat_by_gu, gus, w0, w1); ex = float(b.sum_abs_excess) if b is not None else 0.0
            if sh == 0:
                rec.update(cat_excess=ex, hit=float(ex >= q99), hit90=float(ex >= q90), cat_event_id=(int(b.event_id) if b is not None else -1),
                           cat_dir=(b.direction if b is not None else ""), cat_gu=(b.gu if b is not None else ""), cat_dong=(b.dong if b is not None else ""),
                           cat_peak_z=(float(b.peak_z) if b is not None else np.nan), cat_peak_time=(b.tpk if b is not None else pd.NaT),
                           lag_h=((b.tpk - r.first_ts) / pd.Timedelta(hours=1) if b is not None else np.nan),
                           day_tot=daily.tot.get(r.date, np.nan), day_drop_share=daily.dshare.get(r.date, np.nan))
            else:
                pl["hit"].append(float(ex >= q99)); pl["hit90"].append(float(ex >= q90)); pl["ex"].append(ex); pl["tot"].append(daily.tot.get(d, np.nan)); pl["dsh"].append(daily.dshare.get(d, np.nan))
        rec.update(n_placebo=len(pl["ex"]), pl_hit=(np.mean(pl["hit"]) if pl["hit"] else np.nan), pl_hit90=(np.mean(pl["hit90"]) if pl["hit90"] else np.nan),
                   pl_excess_med=(np.median(pl["ex"]) if pl["ex"] else np.nan), pct_excess=pct(rec["cat_excess"], pl["ex"]),
                   pct_day_tot=pct(rec.get("day_tot", np.nan), pl["tot"]), pct_day_drop=pct(rec.get("day_drop_share", np.nan), pl["dsh"]))
        rows.append(rec)
    M = pd.DataFrame(rows); M["weather"] = M.type.isin(WEATHER)
    M.to_csv(DATA / "disaster_events_seoul_matched.csv", index=False, encoding="utf-8-sig")

    def agg(d):
        d = d.dropna(subset=["hit"]); h = d[d.hit == 1]
        return pd.Series({"n": len(d), "hit": d.hit.mean(), "base": d.pl_hit.mean(), "hit90": d.hit90.mean(), "base90": d.pl_hit90.mean(),
                          "pct_excess": d.pct_excess.mean(), "pct_day_tot": d.pct_day_tot.mean(), "pct_day_drop": d.pct_day_drop.mean(),
                          "drop_share_hits": (h.cat_dir == "decrease").mean() if len(h) else np.nan, "lag_med_h": h.lag_h.median() if len(h) else np.nan,
                          "day_tot_M": d.day_tot.mean() / 1e6})
    byt = M.groupby("type").apply(agg).sort_values("n", ascending=False); byt = byt[byt.n >= 4]
    tot = agg(M); wx = agg(M[M.weather]); nwx = agg(M[~M.weather]); city = agg(M[M.gu == "ALL"]); loc_ = agg(M[M.gu != "ALL"]); locwx = agg(M[(M.gu != "ALL") & M.weather])
    se = lambda s: 0.289 / np.sqrt(max(s.n, 1))     # sd of a uniform percentile / sqrt(n): null standard error of the mean percentile

    # ---- C: catalogue -> messages ----
    msgs = pd.read_csv(DATA / "disaster_msgs_seoul.csv"); msgs = msgs[~(msgs.MSG_CN.str.contains("찾습니다|실종", na=False) & (msgs.DST_SE_NM == "기타"))].copy()
    msgs["ts"] = pd.to_datetime(msgs.ts); msgs = msgs.sort_values("ts"); mts = msgs.ts.values
    def msg_near(t0, t1, gu):
        lo, hi = np.searchsorted(mts, np.datetime64(t0 - pd.Timedelta(hours=12))), np.searchsorted(mts, np.datetime64(t1 + pd.Timedelta(hours=12)), side="right")
        s = msgs.iloc[lo:hi]; s = s[(s.gu == "ALL") | s.gu.str.contains(gu, na=False)]
        return "; ".join(sorted(set(s.DST_SE_NM))) if len(s) else ""
    def cover(df):
        df = df.copy(); df["msg_types"] = [msg_near(r.t0, r.t1, r.gu) for r in df.itertuples()]; df["covered"] = df.msg_types != ""
        df["pl_covered"] = [np.mean([msg_near(r.t0 + pd.Timedelta(days=s), r.t1 + pd.Timedelta(days=s), r.gu) != "" for s in SHIFTS]) for r in df.itertuples()]
        return df
    top = cover(cat[cat.date >= "2023-09-01"].sort_values("sum_abs_excess", ascending=False).head(200))
    val = []
    vf = EV / "events_validation_2023_2024.md"
    if vf.exists():
        for line in vf.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d-]+)\s*\|\s*(.*?)\s*\|\s*([+−-])\s*\|\s*(-?[\d−]+)\s*\|\s*([\w\- ]+?)\s*\|", line)
            if m:
                val.append({"rank": int(m.group(1)), "event_id": int(m.group(2)), "vdate": m.group(3), "area": m.group(4), "category": m.group(7).strip()})
    V = pd.DataFrame(val)
    if len(V):
        V = V.merge(cat[cat.src == "2023_2024"][["event_id", "gu", "t0", "t1", "direction", "sum_abs_excess"]], on="event_id", how="inner"); V = cover(V); V = V[V.vdate >= "2023-09-01"]

    # ---- report ----
    f3 = lambda x: "–" if pd.isna(x) else f"{x:.3f}"; f1 = lambda x: "–" if pd.isna(x) else f"{x:.1f}"; f2 = lambda x: "–" if pd.isna(x) else f"{x:.2f}"
    L = [f"# 재난문자 사건 라벨 × 이상 사건 카탈로그 매칭 (카탈로그 연도 {', '.join(years)}; 겹치는 기간 {M.date.min()}~{M.date.max()})", "",
         f"- 카탈로그 {len(cat):,}개 사건(|z|>3 연결 성분, 2023-09 이후 {int((cat.date >= '2023-09-01').sum()):,}개), 상위 1% |초과량| 임계 {q99/1e3:,.1f}k 인·시, 상위 10% {q90/1e3:,.1f}k. 재난문자 사건(날짜×유형) {len(M)}건 중 카탈로그 일자가 있는 {int(M.hit.notna().sum())}건 평가.",
         "- 히트 = 문자에 명시된 자치구(시 전체면 모든 구) 안에서 [첫 발령−3h, 마지막 발령+12h]와 겹치는 카탈로그 사건 중 최대 |초과량|이 상위 1% 임계 이상. 기준율 = 같은 구·시간창을 ±1~4주 같은 요일로 옮긴 플라시보 8개의 히트율.",
         "- 백분위 = 실제 값보다 작은 플라시보 값의 비율(귀무 평균 0.5, 표준오차 ≈ 0.29/√n). 창 최대 |초과량|, 도시 하루 총 |초과량|, 하루 감소 비중 세 가지.",
         "- 지연 = 카탈로그 사건 피크 시각 − 첫 발령 시각 (음수면 문자보다 먼저 피크).", "",
         "## A. 재난문자 사건 → 카탈로그 (유형별)", "",
         "| 유형 | n | 히트율(상위1%) | 기준율 | 히트율(상위10%) | 기준율 | 창 최대 |초과량| 백분위 | 도시 일 |초과량| 백분위 | 감소 비중 백분위 | 히트 중 감소 비율 | 지연 중앙 h | 귀무 SE |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(name, s):
        return (f"| {name} | {int(s.n)} | {f3(s.hit)} | {f3(s.base)} | {f3(s.hit90)} | {f3(s.base90)} | {f2(s.pct_excess)} | {f2(s.pct_day_tot)} | {f2(s.pct_day_drop)} | "
                f"{f3(s.drop_share_hits)} | {f1(s.lag_med_h)} | {se(s):.2f} |")
    L += [row("전체", tot), row("기상 유형 합계", wx), row("비기상 유형 합계", nwx), row("시 전체 발령", city), row("자치구 지정 발령", loc_), row("자치구 지정 · 기상", locwx), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [row(t, s) for t, s in byt.iterrows()]
    L += ["", "### 히트 사례 (실제 |초과량| 상위 25)", "", "| 날짜 | 유형 | 문자 구 | 첫 발령 | 카탈로그 사건 | 구·동 | 방향 | |초과량| k | 백분위 | peak z | 지연 h |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in M[M.hit == 1].sort_values("cat_excess", ascending=False).head(25).itertuples():
        L.append(f"| {r.date} | {r.type} | {r.gu if r.gu == 'ALL' else r.gu[:30]} | {r.first_ts:%H:%M} | {r.cat_event_id} | {r.cat_gu} {r.cat_dong} | {r.cat_dir} | {r.cat_excess/1e3:.0f} | {f2(r.pct_excess)} | {r.cat_peak_z:.0f} | {r.lag_h:.0f} |")
    L += ["", "## B. 도시 일 단위 (백분위, 귀무 0.5)", "",
          f"- 기상 유형 발령 {int(wx.n)}건: 도시 총 |초과량| 백분위 {wx.pct_day_tot:.2f} (SE {se(wx):.2f}), 감소 비중 백분위 {wx.pct_day_drop:.2f}; 평균 총 |초과량| {wx.day_tot_M:.2f}M 인·시.",
          f"- 비기상 유형 발령 {int(nwx.n)}건: {nwx.pct_day_tot:.2f} (SE {se(nwx):.2f}), 감소 비중 {nwx.pct_day_drop:.2f}; 평균 {nwx.day_tot_M:.2f}M.", "",
          "## C. 카탈로그 → 재난문자 (상위 200 사건, 2023-09 이후; ±12 h, 같은 구 또는 시 전체 문자)", "",
          f"- 문자 있는 사건 {int(top.covered.sum())}/{len(top)} = {top.covered.mean():.2f} vs 플라시보(±1~4주) {top.pl_covered.mean():.2f}; 증가 사건 {top[top.direction=='increase'].covered.mean():.2f} vs {top[top.direction=='increase'].pl_covered.mean():.2f}, 감소 사건 {top[top.direction=='decrease'].covered.mean():.2f} vs {top[top.direction=='decrease'].pl_covered.mean():.2f}.",
          "", "| 문자 유형 조합 | 사건 수 | 증가 | 감소 |", "|---|---|---|---|"]
    for k, d in top.groupby("msg_types"):
        L.append(f"| {k if k else '(문자 없음)'} | {len(d)} | {(d.direction=='increase').sum()} | {(d.direction=='decrease').sum()} |")
    if len(V):
        L += ["", f"### 외부 검증 상위 40 중 2023-09 이후 {len(V)}건 (events_validation_2023_2024.md)", "", "| 범주 | 사건 수 | 문자 있음 | 플라시보 | 문자 유형 |", "|---|---|---|---|---|"]
        for k, d in V.groupby("category"):
            L.append(f"| {k} | {len(d)} | {int(d.covered.sum())} | {d.pl_covered.mean():.2f} | {'; '.join(sorted({t for t in d.msg_types if t}))} |")
        L += ["", "| # | 날짜 | 지역 | 범주 | 문자 유형(±12 h, 같은 구/시 전체) |", "|---|---|---|---|---|"]
        for r in V.sort_values("rank").itertuples():
            L.append(f"| {r.rank} | {r.vdate} | {r.area} | {r.category} | {r.msg_types or '–'} |")
    (EV / "disaster_match_summary.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    b = byt.sort_values("hit"); y = np.arange(len(b))
    ax[0].barh(y - .2, b.hit, .4, label="문자 창 안 히트율"); ax[0].barh(y + .2, b.base, .4, label="플라시보(±1~4주) 기준율", color="0.6")
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{t} (n={int(s.n)})" for t, s in b.iterrows()], fontsize=8); ax[0].set_xlabel("상위 1% 카탈로그 사건 존재 비율"); ax[0].legend(fontsize=8)
    b2 = byt.sort_values("pct_excess"); y2 = np.arange(len(b2))
    ax[1].barh(y2 - .2, b2.pct_excess, .4, label="창 최대 |초과량|"); ax[1].barh(y2 + .2, b2.pct_day_tot, .4, label="도시 하루 총 |초과량|", color="tab:green")
    ax[1].axvline(.5, color="k", lw=.8); ax[1].set_yticks(y2); ax[1].set_yticklabels(b2.index, fontsize=8); ax[1].set_xlabel("플라시보 대비 백분위 (귀무 0.5)"); ax[1].legend(fontsize=8)
    h = M[(M.hit == 1) & M.lag_h.notna()]
    ax[2].hist([h[h.weather].lag_h, h[~h.weather].lag_h], bins=np.arange(-27, 40, 3), stacked=True, label=["기상", "비기상"], color=["tab:blue", "tab:orange"])
    ax[2].axvline(0, color="k", lw=.8); ax[2].set_xlabel("카탈로그 피크 - 첫 발령 (h)"); ax[2].set_ylabel("히트 사건 수"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(EV / "fig_disaster_match.png", dpi=120)


if __name__ == "__main__":
    main()
