"""Merge the venue schedule sources into data/venue_events.csv (columns date, venue, type, start_hour, end_hour, note).

    python build_venue_events.py

Sources
  data/venue_events_manual.csv   hand-collected rows (FC서울 home matches, A-matches, 상암·잠실 concerts, fireworks, rallies)
  data/kbo_jamsil_raw.txt        KBO official schedule (koreabaseball.com) filtered to 잠실: year|MM.DD(dow)|HH:MM|game|series|note
                                 cancelled games (note != '-') are dropped; window = first pitch .. +3 h
  data/kspo_dome_raw.txt         올림픽공원 official event list (ksponco.or.kr) filtered to KSPO DOME: begin|end|time text|place|title
                                 multi-day ranges: every day of a range <= 3 days; longer ranges keep only the weekdays named in the time text
                                 (금/토/일/월 ...), or all days if none are named; start hour parsed from the first time in the text (default 18)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import re, sys
from pathlib import Path
import pandas as pd

import config as C
DATA = C.TABLES
DOW = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}


def parse_hour(txt):
    """first clock time in a Korean/English time string -> hour (int)"""
    t = txt.replace("；", ":").replace(";", ":")
    m = re.search(r"(\d{1,2})\s*:\s*(\d{2})\s*(PM|AM)?", t, re.I)
    if m:
        h = int(m.group(1)); ap = (m.group(3) or "").upper()
        if ap == "PM" and h < 12:
            h += 12
        return h
    m = re.search(r"(\d{1,2})\s*(PM|AM)", t, re.I)
    if m:
        h = int(m.group(1)); return h + 12 if m.group(2).upper() == "PM" and h < 12 else h
    m = re.search(r"오후\s*(\d{1,2})\s*시", t)
    if m:
        h = int(m.group(1)); return h + 12 if h < 12 else h
    m = re.search(r"오전\s*(\d{1,2})\s*시", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2})\s*시", t)
    if m:
        return int(m.group(1))
    return 18


def main():
    manual = pd.read_csv(DATA / "venue_events_manual.csv", encoding="utf-8-sig")
    rows = []
    # ---- KBO 잠실 ----
    for line in (DATA / "kbo_jamsil_raw.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yr, md, tm, game, series, note = line.split("|")
        if note.strip() != "-":
            continue                                             # cancelled (우천취소, 미세먼지취소, 폭염취소, 그라운드사정, 기타)
        d = f"{yr}-{md[:2]}-{md[3:5]}"; h = int(tm[:2])
        rows.append(dict(date=d, venue="잠실야구장", type="baseball", start_hour=h, end_hour=min(h + 3, 23), note=f"KBO {series} {game}"))
    # ---- KSPO DOME ----
    for line in (DATA / "kspo_dome_raw.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b, e, tm, plc, title = line.split("|")
        b, e = pd.Timestamp(b), pd.Timestamp(e); days = pd.date_range(b, e, freq="D"); h = parse_hour(tm)
        if len(days) > 3:
            named = {DOW[c] for c in re.findall(r"\(([월화수목금토일])\)", tm)} | ({DOW[c] for c in re.findall(r"(?<![\w가-힣])([월화수목금토일])(?=\s*(?:오후|오전|\d|/|,|$))", tm)})
            if "토일" in tm:
                named |= {5, 6}
            if named:
                days = [d for d in days if d.dayofweek in named]
        for d in days:
            if d.year < 2023:
                continue
            rows.append(dict(date=d.strftime("%Y-%m-%d"), venue="KSPO돔(올림픽공원)", type="concert", start_hour=h, end_hour=min(h + 3, 23), note=title.strip()[:60]))
    add = pd.DataFrame(rows)
    out = pd.concat([manual, add], ignore_index=True).drop_duplicates(subset=["date", "venue", "start_hour"]).sort_values(["date", "venue"])
    out.to_csv(DATA / "venue_events.csv", index=False, encoding="utf-8-sig")
    print(out.groupby(["type", "venue"]).size()); print(out.groupby(out.date.str[:4]).size()); print("total", len(out))


if __name__ == "__main__":
    main()
