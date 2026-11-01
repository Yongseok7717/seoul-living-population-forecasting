"""Fetch Seoul emergency medical institutions from 국립중앙의료원 응급의료정보 조회 서비스 (data.go.kr B552657).

The service key is read from the environment variable DATAGO_KEY (paste the '일반 인증키' exactly as shown on
the portal; a key that is already percent-encoded is used as-is).

    $env:DATAGO_KEY="..."; python fetch_emergency_api.py

Steps
  1. getEgytListInfoInqire  Q0=서울특별시  -> list: hpid, name, address, class (dutyEmclsName), lon/lat
  2. getEgytBassInfoInqire  per hpid       -> basic info: hperyn (응급실 병상), hpbdn (병상), hpicuyn/hpccuyn etc.
Output: [서울열린데이터 광장]/시설/datago_emergency_seoul.csv  (utf-8-sig)
Then re-run build_facilities.py (adds ftype 'emergency') and build_accessibility.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import os, sys, time, csv, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
import config as C

BASE = "https://apis.data.go.kr/B552657/ErmctInfoInqireService"
OUT = C.OPEN_DIR / "시설" / "datago_emergency_seoul.csv"
KEY = os.environ.get("DATAGO_KEY", "")


def call(op, **params):
    key = KEY if "%" in KEY else urllib.parse.quote(KEY, safe="")
    q = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in params.items())
    url = f"{BASE}/{op}?serviceKey={key}&{q}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                xml = r.read().decode("utf-8", "replace")
            root = ET.fromstring(xml)
            code = root.findtext(".//resultCode")
            if code not in (None, "00", "0"):
                raise RuntimeError(f"{op}: resultCode={code} {root.findtext('.//resultMsg')}")
            return root
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def items(root):
    return [{c.tag: (c.text or "").strip() for c in it} for it in root.iter("item")]


def main():
    if not KEY:
        sys.exit("DATAGO_KEY is not set")
    rows, page = [], 1
    while True:
        root = call("getEgytListInfoInqire", Q0="서울특별시", pageNo=page, numOfRows=100)
        its = items(root)
        rows += its
        total = int(root.findtext(".//totalCount") or 0)
        print(f"list page {page}: +{len(its)} (total {total})", flush=True)
        if page * 100 >= total or not its:
            break
        page += 1
    print("fetching basic info per institution ...", flush=True)
    for i, r in enumerate(rows):
        try:
            b = items(call("getEgytBassInfoInqire", HPID=r["hpid"], pageNo=1, numOfRows=1))
            if b:
                for k in ("hperyn", "hpbdn", "hpicuyn", "hpccuyn", "hpcuyn", "hpgryn", "hpopyn", "dutyEryn", "dutyHayn", "dutyHano", "dutyInf", "dutyEmcls", "dutyEmclsName", "wgs84Lon", "wgs84Lat", "dutyAddr", "dutyName", "dutyDivNam"):
                    if k in b[0] and b[0][k]:
                        r[k] = b[0][k]
        except Exception as e:
            r["basic_error"] = str(e)[:80]
        if i % 20 == 0:
            print(i, r.get("dutyName"), r.get("dutyEmclsName"), "ER beds", r.get("hperyn"), flush=True)
        time.sleep(0.15)
    cols = sorted({k for r in rows for k in r})
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} institutions")


if __name__ == "__main__":
    main()
