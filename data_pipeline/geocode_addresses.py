"""Geocode facility addresses with the VWorld geocoder (free key from https://www.vworld.kr, 40,000 calls/day).

The key is read from the environment variable VWORLD_KEY; nothing is stored in this file.

    set VWORLD_KEY=your-key            (PowerShell: $env:VWORLD_KEY="your-key")
    python geocode_addresses.py                                              # processed/facilities/to_geocode.csv
    python geocode_addresses.py --input ../processed/facilities/to_geocode_retry.csv   # normalised retry list (prepare_geocode_retry.py);
                                                                             # an optional 'alt' column holds extra candidates separated by ' || '

Input : processed/facilities/to_geocode.csv  (written by build_facilities.py: key, ftype, name, address)
Output: processed/facilities/geocoded.csv    (key, address, lon, lat, status) - resumable, appends as it goes
Then re-run build_facilities.py; it picks up geocoded.csv automatically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import csv, os, re, sys, time, json, urllib.parse, urllib.request
from pathlib import Path
import config as C

IN = C.OUT / "facilities" / "to_geocode.csv"
OUT = C.OUT / "facilities" / "geocoded.csv"
KEY = os.environ.get("VWORLD_KEY")


def vworld(addr, typ):
    q = urllib.parse.urlencode({"service": "address", "request": "getcoord", "version": "2.0", "crs": "epsg:4326",
                                "address": addr, "refine": "true", "simple": "false", "format": "json", "type": typ, "key": KEY})
    with urllib.request.urlopen("https://api.vworld.kr/req/address?" + q, timeout=30) as r:
        j = json.loads(r.read().decode("utf-8"))
    res = j.get("response", {})
    if res.get("status") == "OK":
        p = res["result"]["point"]
        return float(p["x"]), float(p["y"]), "OK"
    return None, None, res.get("status", "ERROR")


def vworld_place(query):
    """VWorld search API (type=place): POI name search, returns the first hit's coordinates."""
    q = urllib.parse.urlencode({"service": "search", "request": "search", "version": "2.0", "crs": "EPSG:4326", "size": 5, "page": 1,
                                "query": query, "type": "place", "format": "json", "key": KEY})
    with urllib.request.urlopen("https://api.vworld.kr/req/search?" + q, timeout=30) as r:
        j = json.loads(r.read().decode("utf-8"))
    res = j.get("response", {})
    if res.get("status") == "OK" and res.get("result", {}).get("items"):
        it = res["result"]["items"][0]; pt = it.get("point", {})
        return float(pt["x"]), float(pt["y"]), "OK_PLACE:" + str(it.get("title", ""))[:40]
    return None, None, res.get("status", "ERROR")


def main():
    global IN
    if len(sys.argv) >= 3 and sys.argv[1] == "--input":
        IN = Path(sys.argv[2]).resolve()
    if not KEY:
        sys.exit("VWORLD_KEY is not set")
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8-sig") as f:
            done = {r["key"] for r in csv.DictReader(f) if r["status"].startswith("OK")}   # retry anything that failed
    rows = [r for r in csv.DictReader(open(IN, encoding="utf-8-sig")) if r["key"] not in done]
    print(f"{len(rows)} addresses to geocode ({len(done)} already done)")
    new = not OUT.exists()
    with open(OUT, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "address", "lon", "lat", "status"])
        for i, r in enumerate(rows):
            addr = r["address"].strip()
            lon = lat = None; status = "EMPTY"
            # VWorld rejects building names / floors: try the raw address, then the cleaned road address
            # ("서울특별시 송파구 정의로8길 4, KDG70타워 2층 (문정동)" -> "서울특별시 송파구 정의로8길 4"), then the 지번 in ( )
            cands = []
            if addr:
                cands.append((addr, "road"))
                base = re.split(r"[,，]", re.sub(r"\(.*?\)", "", addr))[0].strip()
                base = re.sub(r"\s+(지하)?\d+(층|호)$", "", base).strip()
                if base and base != addr:
                    cands.append((base, "road"))
                m = re.search(r"\(([^()]*?(동|가|리)\s*[\d-]+[^()]*)\)", addr)
                if m:
                    gu = " ".join(base.split()[:2]) if len(base.split()) >= 2 else ""
                    cands.append((f"{gu} {m.group(1)}".strip(), "parcel"))
                cands.append((base or addr, "parcel"))
                for alt in [x.strip() for x in (r.get("alt") or "").split("||") if x.strip()]:
                    if alt.startswith("place:"):
                        cands.append((alt, "place"))
                    else:
                        cands.append((alt, "road")); cands.append((alt, "parcel"))
            for a, typ in cands:
                try:
                    if a.startswith("place:"):
                        lon, lat, status = vworld_place(a[6:].strip())
                    else:
                        lon, lat, status = vworld(a, typ)
                except Exception as e:
                    status = f"EXC {e}"
                if lon is not None:
                    break
                time.sleep(0.1)
            w.writerow([r["key"], addr, lon, lat, status]); f.flush()
            if i % 100 == 0:
                print(i, addr[:40], status, flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    main()
