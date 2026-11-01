"""Experiment 26 - validation of the venue rule (EC where a fitted profile exists, else ZA) without using the scored events:
the choice between B1+P (EC), B1+ZA (ZA) and B1 is made per venue-type on the 2025 events of January-June and scored on July-December
(weights were fitted on 2024 in exp21, so only the choice is selected here).  Also reports the reverse split and the exp21 rule.
Outputs results/exp26_rule_validation/summary.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "exp26_rule_validation"; OUT.mkdir(parents=True, exist_ok=True)
CANDS = ["B1", "B1+P", "B1+ZA"]


def main():
    e = pd.read_csv(ROOT / "results" / "exp21_event_blend" / "events.csv"); e["date"] = pd.to_datetime(e.date); e = e[e.cand.isin(CANDS)]
    e["half"] = (e.date.dt.month > 6).map({False: "H1", True: "H2"}); rows = []
    for origin in ("day-ahead", "short"):
        for sel_half, score_half in (("H1", "H2"), ("H2", "H1")):
            s = e[(e.origin == origin)]; pick = s[s.half == sel_half].groupby(["venue", "cand"]).wmape_area.mean().unstack().idxmin(1)
            sc = s[s.half == score_half]; chosen = sc[sc.apply(lambda r: pick.get(r.venue, "B1") == r.cand, axis=1)]
            base = sc[sc.cand == "B1"]; ec = sc[sc.cand == "B1+P"]; za = sc[sc.cand == "B1+ZA"]
            rows.append({"origin": origin, "selected_on": sel_half, "scored_on": score_half, "rule_wMAPE": chosen.wmape_area.mean(), "B1_wMAPE": base.wmape_area.mean(), "EC_wMAPE": ec.wmape_area.mean(), "ZA_wMAPE": za.wmape_area.mean(), "choice": dict(pick), "n_events": chosen.date.nunique()})
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig"); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
