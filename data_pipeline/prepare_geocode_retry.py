"""Normalise the addresses that VWorld could not geocode and write a retry list.

    python prepare_geocode_retry.py            -> processed/facilities/to_geocode_retry.csv (+ review printed)
    set VWORLD_KEY=...; python geocode_addresses.py --input ../processed/facilities/to_geocode_retry.csv
    python build_facilities.py                 (picks up geocoded.csv by key)

Rules: fix typos (중랄구, 도옵구, 가어구, 서울시, doubled 서울특별시), drop parentheses / floors / rooms / building names,
re-space road names ("동일로 237다길16" -> "동일로237다길 16", "오패산로98" -> "오패산로 98"), infer the missing 구 from the
road name using the 19k already-geocoded facility addresses (majority 구 per road; ambiguous roads keep every candidate as
an alternate), and add the jibun (parcel) address from the parenthesis as another alternate. Keys are unchanged so
build_facilities.py joins the result back.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import re, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import config as C

FAC = C.OUT / "facilities"
# verified by web search (2026-09-06): road names that were mangled in the source
OVERRIDE = {"효원노인복지센터": "서울특별시 중랑구 상봉중앙로1길 12", "다원요양": "서울특별시 관악구 조원중앙로 24"}
TYPO = {"중랄구": "중랑구", "도옵구": "도봉구", "가어구": "강서구", "서울시": "서울특별시", "서울특별시특별시": "서울특별시"}
GU = "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 관악구 서초구 강남구 송파구 강동구".split()
ROAD = re.compile(r"([가-힣A-Za-z0-9·]+?(?:대로|로|길))\s*(\d+[가-힣]?길)?\s*(\d+(?:-\d+)?)(?![가-힣\d])")


def basic(a):
    a = str(a).strip()
    for k, v in TYPO.items():
        a = a.replace(k, v)
    a = re.sub(r"(서울특별시\s*)+", "서울특별시 ", a)
    return re.sub(r"\s+", " ", a).strip()


def parse(a):
    """-> (gu or None, road, number, parcel or None)"""
    a = basic(a)
    paren = re.findall(r"\(([^)]*)\)", a); body = re.sub(r"\([^)]*\)", " ", a)
    gu = next((g for g in GU if re.search(r"\b" + g, body)), None)
    body = re.sub(r"\b" + str(gu), " ", body) if gu else body
    body = re.sub(r"서울특별시", " ", body)
    body = re.sub(r"(로|길)\s+(\d+[가-힣]?길)", r"\1\2", body)          # "동일로 237다길" -> "동일로237다길"
    m = ROAD.search(body)
    if not m:
        return gu, None, None, None, None
    road = m.group(1) + (m.group(2) or ""); num = m.group(3)
    pre = re.search(r"([가-힣]{2,})\s+$", body[:m.start()])                 # a token glued in front of the road ("조원 중앙로") -> joined variant
    joined = (pre.group(1) + road) if pre else None
    parcel = None
    for ptxt in paren:
        pm = re.search(r"([가-힣0-9]+(?:동|가))\s*(\d+(?:-\d+)?)", ptxt)
        if pm:
            parcel = (pm.group(1), pm.group(2)); break
    return gu, road, num, parcel, joined


def main():
    fac = pd.read_parquet(FAC / "facilities.parquet")
    src = fac[fac.has_coords & fac.address.notna() & fac.address.str.contains("서울")]
    road_gu = defaultdict(Counter)
    for a in src.address.astype(str):
        gu, road, num, _, _ = parse(a)
        if gu and road:
            road_gu[road][gu] += 1
    dm = pd.read_parquet(C.OUT / "grid" / "cell_dong_map.parquet"); gu_names = dict(zip(["11110","11140","11170","11200","11215","11230","11260","11290","11305","11320","11350","11380","11410","11440","11470","11500","11530","11545","11560","11590","11620","11650","11680","11710","11740"], GU))
    dong_gu = {}
    for code, name in zip(dm.H_DNG_CD.astype(str), dm.dong_name.astype(str)):
        g = gu_names.get(code[:5])
        if g:
            dong_gu[name] = g; dong_gu[re.sub(r"제(\d)", r"", name)] = g; dong_gu[re.sub(r"[·].*", "", re.sub(r"제?\d+동$", "동", name))] = g
    t = pd.read_csv(FAC / "to_geocode.csv", encoding="utf-8-sig")
    rows = []
    for r in t.itertuples():
        gu, road, num, parcel, joined = parse(r.address)
        cands, note = [], ""
        if road is None:
            note = "road not parsed"; primary = basic(r.address)
        else:
            gus = []
            if gu:
                gus = [gu]
            else:
                base = re.sub(r"\d+[가-힣]?길$", "", road)                       # "도림로63길" -> "도림로"
                key = road if road in road_gu else (base if base in road_gu else None)
                pd_ = None
                if parcel:
                    pd_ = dong_gu.get(parcel[0]) or dong_gu.get(re.sub(r"\d+가$", "", parcel[0])) or dong_gu.get(re.sub(r"\d+동$", "동", parcel[0]))
                if key:
                    cnt = road_gu[key]; tot = sum(cnt.values()); top = cnt.most_common(3)
                    gus = [g for g, c in top]; note = f"gu inferred from {tot} facilities on {key}: " + ", ".join(f"{g} {c}" for g, c in top)
                    if top[0][1] / tot < 0.8:
                        note = "AMBIGUOUS " + note
                        if pd_:
                            gus = [pd_] + [g for g in gus if g != pd_]; note = f"gu from the parcel dong {parcel[0]} ({pd_}); road {key} is ambiguous"
                elif pd_:
                    gus = [pd_]; note = f"gu inferred from the parcel dong {parcel[0]}"
                else:
                    note = "gu unknown (road not in facility data)"
            primary = f"서울특별시 {gus[0] + ' ' if gus else ''}{road} {num}"
            for g in gus[1:]:
                cands.append(f"서울특별시 {g} {road} {num}")
            if joined:
                cands.insert(0, f"서울특별시 {gus[0] + ' ' if gus else ''}{joined} {num}"); note += " | joined-road variant added"
            cands.append(f"서울특별시 {road} {num}")
            if parcel:
                cands.append(f"서울특별시 {gus[0] + ' ' if gus else ''}{parcel[0]} {parcel[1]}")
        for k, v in OVERRIDE.items():
            if k in str(r.name):
                cands.insert(0, primary); primary = v; note = "verified road name (web search)"
        cands = [c for c in dict.fromkeys(cands) if c != primary]
        rows.append({"key": r.key, "ftype": r.ftype, "name": r.name, "address": primary, "alt": " || ".join(cands), "original": r.address, "note": note})
    out = pd.DataFrame(rows); out.to_csv(FAC / "to_geocode_retry.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 100)
    print(out[["ftype", "name", "address", "alt", "note"]].to_string(index=False))
    print(f"\n{len(out)} rows, {out.address.nunique()} unique primary addresses; ambiguous/unknown gu: {int(out.note.str.contains('AMBIGUOUS|unknown|not parsed').sum())}")


if __name__ == "__main__":
    main()
