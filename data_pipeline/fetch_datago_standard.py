"""Download a data.go.kr 표준데이터 set the same way the portal's CSV button does (JSON API, no login).

    python fetch_datago_standard.py 15021148 out.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import json, sys, csv, time, urllib.request, urllib.parse

BASE = "https://www.data.go.kr"
HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/javascript, */*; q=0.01",
       "X-Requested-With": "XMLHttpRequest"}


def get_json(url):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main(pk, out):
    HDR["Referer"] = f"{BASE}/data/{pk}/standard.do"
    header = get_json(f"{BASE}/download/columList.json?pk={pk}&ext=CSV")
    total = int(header["totalCount"])
    cols_en = [c["columCode"] for c in header["columList"]]
    cols_kr = [c["columNm"] for c in header["columList"]]
    col_nm_list = header["tableVO"]["colNmList"]
    svc = header["tableVO"]["svcTableNm"]
    print(f"pk={pk} file={header.get('fileName')} total={total} cols={len(cols_en)} svcTable={svc}")
    per = 10000
    rows = []
    page = 1                                   # the portal's generateCsv() counts pages from 1
    while (page - 1) * per < total:
        q = [("publicDataPk", pk), ("totalCount", total), ("svcTableNm", svc), ("perPage", per), ("page", page)]
        q += [("colNmList", c) for c in col_nm_list]
        data = get_json(f"{BASE}/download/standard.json?" + urllib.parse.urlencode(q))
        if page == 1:
            print("response type:", list(data.keys())[:10] if isinstance(data, dict) else type(data))
        recs = data
        if isinstance(data, dict):
            for k in ("data", "list", "records", "result", "resultList", "dataList"):
                if k in data and isinstance(data[k], list):
                    recs = data[k]; break
            else:
                recs = next((v for v in data.values() if isinstance(v, list)), [])
        rows.extend(recs)
        print(f"  page {page}: +{len(recs)} (cum {len(rows)})", flush=True)
        if not recs:
            break
        page += 1
        time.sleep(0.5)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols_kr)
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get(c, r.get(c.lower(), r.get(c.upper(), ""))) for c in cols_en])
            else:
                w.writerow(r)
    if rows and isinstance(rows[0], dict):
        print("sample keys:", list(rows[0].keys())[:12])
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
