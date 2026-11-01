"""Fetch 긴급재난문자 (emergency disaster text alerts) 2023-01 .. 2025-12 from 재난안전데이터공유플랫폼 (safetydata.go.kr,
DSSP-IF-00247 = data.go.kr 15134001 LINK) and keep the Seoul ones.

Key: SAFETYDATA_KEY in pipeline/.env (platform-issued 16-char key, IP-bound; a data.go.kr key answers
"SERVICE KEY IS NOT REGISTERED"). Never printed.  Daily quota of the issued key: 100 calls -> the script
  1. probes once whether crtDt is optional and whether rgnNm filters server-side (2-3 calls, cached in state.json),
  2. pages the cheapest complete query (numOfRows 1000) and caches every page in data/disaster_msgs_cache/<mode>_p<page>.json,
     stopping at --budget calls (default 95),
  3. re-runs resume from the cache (nothing is re-fetched), so a multi-day collection just means running it again tomorrow.
Falls back to per-day queries (crtDt) only if the server refuses a query without crtDt.

    python fetch_disaster_msgs.py [--start 20230101 --end 20251231] [--budget 95] [--probe] [--summarise]

Outputs (written whenever at least one page is cached; complete when the log says COMPLETE)
  data/disaster_msgs_all.csv     every fetched message (all fields)
  data/disaster_msgs_seoul.csv   reception region includes 서울: + gu (comma list of 자치구, 'ALL' for 서울 전체), ts, date, hour
  data/disaster_msgs_summary.md  counts by year / type / step / gu / hour; coverage and completeness flag
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import argparse, csv, json, os, re, sys, time, urllib.parse, urllib.request
from collections import Counter
from datetime import datetime, timedelta
import config as C

BASE = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00247"
DATA = C.TABLES; CACHE = DATA / "disaster_msgs_cache"; STATE = CACHE / "state.json"
GU = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구",
      "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
CALLS = 0


def key():
    k = os.environ.get("SAFETYDATA_KEY") or ""
    if not k:
        sys.exit("SAFETYDATA_KEY is not set in pipeline/.env (add a line SAFETYDATA_KEY=<key>)")
    return k if "%" in k else urllib.parse.quote(k, safe="")


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def call(params, k, budget):
    global CALLS
    if CALLS >= budget:
        raise RuntimeError("BUDGET")
    url = f"{BASE}?serviceKey={k}&" + urllib.parse.urlencode(params)
    for attempt in range(3):
        CALLS += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            h = j.get("header", {}) or {}
            if str(h.get("resultCode")) not in ("0", "00"):
                raise RuntimeError(f"resultCode={h.get('resultCode')} {h.get('resultMsg')}")
            return j
        except Exception as e:
            s = str(e)
            if attempt == 2 or "SERVICE KEY" in s or "resultCode" in s:
                raise
            time.sleep(3 * (attempt + 1))


def field(rows, *cands):
    cols = {c.upper(): c for r in rows[:5] for c in r}
    for c in cands:
        if c.upper() in cols:
            return cols[c.upper()]
    return None


def parse_ts(s):
    s = (s or "").strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def seoul_gu(region):
    parts = [p.strip() for p in re.split(r"[,\n;]", region or "") if "서울" in p]
    if not parts:
        return ""
    gus = sorted({g for p in parts for g in GU if g in p})
    if not gus or any("전체" in p or p.replace(" ", "") in ("서울특별시", "서울시", "서울") for p in parts):
        return "ALL"
    return ",".join(gus)


def probe(k, budget):
    """Is crtDt optional? Does rgnNm filter server-side? Cached in state.json (costs 2 calls once)."""
    st = load_state()
    if "probe" in st:
        return st["probe"]
    p = {}
    try:
        j = call({"returnType": "json", "pageNo": 1, "numOfRows": 3, "rgnNm": "서울특별시"}, k, budget)
        p["no_crtdt_ok"] = True; p["total_rgn"] = int(j.get("totalCount") or 0); p["sample"] = (j.get("body") or [])[:3]
    except RuntimeError as e:
        p["no_crtdt_ok"] = False; p["err"] = str(e)[:120]
    if p.get("no_crtdt_ok"):
        j = call({"returnType": "json", "pageNo": 1, "numOfRows": 3}, k, budget)
        p["total_all"] = int(j.get("totalCount") or 0); p["rgn_filters"] = 0 < p["total_rgn"] < p["total_all"]
    st["probe"] = p; save_state(st)
    return p


def load_cache():
    rows = {}
    for f in sorted(CACHE.glob("*.json")):
        if f.name == "state.json":
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            rows[str(r.get("SN") or r.get("sn") or json.dumps(r, sort_keys=True, ensure_ascii=False))] = r
    return rows


def collect(k, budget, start, end, p):
    """Page the cheapest complete query until the cache covers [start, end]. Returns True when complete."""
    st = load_state()
    if st.get("complete"):
        return True
    if p.get("no_crtdt_ok"):
        base = {"returnType": "json", "numOfRows": 1000}; mode = "all"
        if p.get("rgn_filters"):
            base["rgnNm"] = "서울특별시"; mode = "rgn"
        total = p["total_rgn"] if mode == "rgn" else p["total_all"]; n_pages = (total + 999) // 1000
        print(f"mode {mode}: total {total} rows -> {n_pages} pages of 1000 (budget {budget} calls/run)", flush=True)
        page = st.get("next_page", 1); tk = None
        while page <= n_pages:
            f = CACHE / f"{mode}_p{page:04d}.json"
            if not f.exists():
                try:
                    j = call({**base, "pageNo": page}, k, budget)
                except RuntimeError as e:
                    if str(e) == "BUDGET":
                        print(f"call budget reached at page {page}/{n_pages}; run again tomorrow to continue", flush=True)
                        st["next_page"] = page; save_state(st); return False
                    raise
                body = j.get("body") or []
                f.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                if not body:
                    break
                tk = tk or field(body, "CRT_DT", "CREATE_DATE", "CRTDT")
                tss = [t for t in (parse_ts(r.get(tk)) for r in body) if t]
                print(f"page {page}/{n_pages}: {len(body)} rows" + (f", {min(tss):%Y-%m-%d} .. {max(tss):%Y-%m-%d}" if tss else ""), flush=True)
                if tss and max(tss) < start:          # newest-first order: everything after this precedes the start date
                    print("all rows on this page precede the start date; stopping", flush=True); page += 1; break
                time.sleep(0.3)
            page += 1; st["next_page"] = page; save_state(st)
        st["complete"] = True; save_state(st); return True
    print("server requires crtDt: per-day mode (1 call/day, 100 calls/day quota -> ~11 runs)", flush=True)
    d = datetime.strptime(st.get("next_day", f"{start:%Y%m%d}"), "%Y%m%d")
    while d <= end:
        f = CACHE / f"day_{d:%Y%m%d}.json"
        if not f.exists():
            try:
                j = call({"returnType": "json", "pageNo": 1, "numOfRows": 1000, "crtDt": f"{d:%Y%m%d}"}, k, budget)
            except RuntimeError as e:
                if str(e) == "BUDGET":
                    st["next_day"] = f"{d:%Y%m%d}"; save_state(st); print(f"call budget reached at {d:%Y-%m-%d}; run again tomorrow", flush=True); return False
                raise
            f.write_text(json.dumps(j.get("body") or [], ensure_ascii=False), encoding="utf-8"); time.sleep(0.3)
        d += timedelta(days=1)
    st["complete"] = True; save_state(st); return True


def summarise(start, end, complete):
    rows = list(load_cache().values())
    if not rows:
        print("nothing cached yet"); return
    cols = sorted({c for r in rows for c in r})
    rk = field(rows, "RCPTN_RGN_NM", "LOCATION_NAME", "RGN_NM") or "RCPTN_RGN_NM"; tk = field(rows, "CRT_DT", "CREATE_DATE", "CRTDT") or "CRT_DT"
    typ = field(rows, "DST_SE_NM", "DISASTER_TYPE"); step = field(rows, "EMRG_STEP_NM", "EMERGENCY_STEP")
    with open(DATA / "disaster_msgs_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    seoul = []
    for r in rows:
        gu = seoul_gu(r.get(rk, ""))
        if not gu:
            continue
        ts = parse_ts(r.get(tk))
        if ts is None or not (start <= ts < end + timedelta(days=1)):
            continue
        rr = dict(r); rr.update(gu=gu, ts=ts.isoformat(), date=ts.date().isoformat(), hour=ts.hour); seoul.append(rr)
    seoul.sort(key=lambda r: r["ts"])
    with open(DATA / "disaster_msgs_seoul.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "date", "hour", "gu"] + cols); w.writeheader(); w.writerows(seoul)
    tss = [t for t in (parse_ts(r.get(tk)) for r in rows) if t]

    def tab(key_fn, head):
        c = Counter(key_fn(r) for r in seoul)
        return [f"| {head} | 건수 |", "|---|---|"] + [f"| {kk} | {v} |" for kk, v in sorted(c.items(), key=lambda x: -x[1])] + [""]
    L = [f"# 긴급재난문자 수집 요약 (safetydata.go.kr DSSP-IF-00247, 목표 기간 {start:%Y-%m-%d}~{end:%Y-%m-%d})", "",
         f"- 수집 상태: {'COMPLETE' if complete else 'PARTIAL (일일 호출 한도, 내일 재실행)'}; 캐시 {len(rows):,}건" + (f", 시각 범위 {min(tss):%Y-%m-%d} .. {max(tss):%Y-%m-%d}" if tss else ""),
         f"- 서울 수신 {len(seoul):,}건 (서울 전체 발령 {sum(r['gu'] == 'ALL' for r in seoul):,}건, 자치구 지정 {sum(r['gu'] != 'ALL' for r in seoul):,}건)", "",
         "## 연도별 (서울)", ""] + tab(lambda r: r["date"][:4], "연도")
    if typ:
        L += ["## 재난 유형별 (서울)", ""] + tab(lambda r: r.get(typ, ""), "유형")
    if step:
        L += ["## 송출 단계별 (서울)", ""] + tab(lambda r: r.get(step, ""), "단계")
    gc = Counter(g for r in seoul for g in (GU if r["gu"] == "ALL" else r["gu"].split(",")))
    L += ["## 자치구별 (서울 전체 발령은 25개 구 모두에 가산)", "", "| 자치구 | 건수 |", "|---|---|"] + [f"| {g} | {gc[g]} |" for g in GU] + ["", "## 시간대별 (서울)", ""] + tab(lambda r: r["hour"], "시")
    (DATA / "disaster_msgs_summary.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L[:6]))
    print(f"wrote disaster_msgs_all.csv ({len(rows)}), disaster_msgs_seoul.csv ({len(seoul)}), calls used this run: {CALLS}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="20230101"); ap.add_argument("--end", default="20251231"); ap.add_argument("--budget", type=int, default=95)
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--summarise", action="store_true"); a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True); start, end = datetime.strptime(a.start, "%Y%m%d"), datetime.strptime(a.end, "%Y%m%d")
    if a.summarise:
        summarise(start, end, load_state().get("complete", False)); return
    k = key(); p = probe(k, a.budget)
    print("probe:", json.dumps({kk: v for kk, v in p.items() if kk != "sample"}, ensure_ascii=False)); print("sample:", json.dumps(p.get("sample", [])[:2], ensure_ascii=False)[:1500])
    if a.probe:
        return
    complete = collect(k, a.budget, start, end, p); summarise(start, end, complete)
    print("COMPLETE" if complete else "PARTIAL - rerun tomorrow (cache is kept)")


if __name__ == "__main__":
    main()
