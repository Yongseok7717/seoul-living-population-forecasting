"""Download datasets from 서울 열린데이터광장 without login, the way the site's own buttons do.

    python fetch_seoul.py sheet OA-15052 3644 out.csv           # SHEET tab -> CSV (cp949)
    python fetch_seoul.py files OA-15636                        # list files on the FILE tab (seq, name, MB)
    python fetch_seoul.py file  OA-15636 <seq> out.zip          # download one file by seq
    python fetch_seoul.py file  OA-15636 "2023년" out.zip       # or by a substring of the file name
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import re, sys, html, urllib.request, urllib.parse

UA = {"User-Agent": "Mozilla/5.0"}


def sheet(inf_id, total, out):
    data = urllib.parse.urlencode({"srvType": "S", "infId": inf_id, "serviceKind": "1", "pageNo": "1", "gridTotalCnt": str(total),
                                   "ssUserId": "SAMPLE_VIEW", "strWhere": "", "strOrderby": "", "txtFilter": "",
                                   "filterCol": "필터선택", "sheetDownFileType1": "csv"}).encode()
    req = urllib.request.Request("https://datafile.seoul.go.kr/bigfile/iot/sheet/csv/download.do", data=data,
                                 headers={**UA, "Referer": f"https://data.seoul.go.kr/dataList/{inf_id}/S/1/datasetView.do"})
    with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
        body = r.read(); f.write(body)
        n_lines = body.count(b"\n")
        print(f"{inf_id} sheet -> {out}: {len(body):,} bytes, {n_lines:,} lines, {r.headers.get('Content-Type')}")


import http.cookiejar
_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_JAR))
_INF_SEQ = {}


def list_files(inf_id):
    req = urllib.request.Request(f"https://data.seoul.go.kr/dataList/{inf_id}/F/1/datasetView.do", headers=UA)
    page = _OPENER.open(req, timeout=120).read().decode("utf-8", "replace")
    frm = re.search(r'<form name="frmFile".*?</form>', page, re.S)
    if frm:
        m = re.search(r'name="infSeq"\s+value="(\d+)"', frm.group(0))
        _INF_SEQ[inf_id] = m.group(1) if m else "1"
    rows = []
    for m in re.finditer(r'<tr id="fileTr_\d+">(.*?)</tr>', page, re.S):
        tr = m.group(1)
        seq = re.search(r"downloadFile\('(\d+)'\)", tr)
        tds = [html.unescape(re.sub(r"<[^>]+>", " ", td)).strip() for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        tds = [re.sub(r"\s+", " ", t) for t in tds]
        if seq:
            rows.append((seq.group(1), tds))
    return rows


def file(inf_id, which, out):
    rows = list_files(inf_id)
    seq = None
    for s, tds in rows:
        if which == s or any(which in t for t in tds):
            seq = s; break
    if seq is None:
        raise SystemExit(f"no file matching {which!r}; available: {rows}")
    data = urllib.parse.urlencode({"infId": inf_id, "seqNo": "", "seq": seq, "infSeq": _INF_SEQ.get(inf_id, "1")}).encode()
    req = urllib.request.Request("https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false", data=data,
                                 headers={**UA, "Referer": f"https://data.seoul.go.kr/dataList/{inf_id}/F/1/datasetView.do"})
    with _OPENER.open(req, timeout=1800) as r, open(out, "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); n += len(chunk)
        print(f"{inf_id} file seq {seq} -> {out}: {n:,} bytes, {r.headers.get('Content-Disposition')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sheet":
        sheet(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "files":
        for s, tds in list_files(sys.argv[2]):
            print(s, tds)
    elif cmd == "file":
        file(sys.argv[2], sys.argv[3], sys.argv[4])
