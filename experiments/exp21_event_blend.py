"""Experiment 21 - event-aware blend: combine B1 (0.9 M2 + 0.1 R6) with the venue event profile P (exp19) and the analog-context zero-shot
forecast ZA (exp20) on scheduled event days, with weights fitted on the 2024 events (per venue-type where >= 3 events, pooled otherwise).

Candidates on 2025 event windows (cells within 500 m): M2, B1, ZA, EC (a M2 + b P), B1+P (a B1 + b P), B1+ZA (a B1 + c ZA),
B1+ZA+P (a B1 + c ZA + b P), M2+ZA+P.  Non-event days are untouched (the operational forecast stays B1).
Outputs results/exp21_event_blend/{events.csv, summary.csv, weights.csv}.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "exp21_event_blend"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments"))
from exp18_venue import VEN_EN  # noqa: E402
import exp19_event_profile as e19  # noqa: E402
from exp20_chronos_analog import NBLK, H, MODEL  # noqa: E402

MINFIT = 3
COMBOS = {"EC": ("M2", "P"), "B1+P": ("B1", "P"), "B1+ZA": ("B1", "ZA"), "B1+ZA+P": ("B1", "ZA", "P"), "M2+ZA+P": ("M2", "ZA", "P"), "ZA+P": ("ZA", "P")}


def main():
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
    dt, hidx, X, B, Rser, fc, vc, cpos = e19.load(); ev = e19.events_table(); P = e19.profiles(ev, dt, hidx, Rser, vc, cpos)
    prev = {key: list(grp.itertuples()) for key, grp in ev.groupby("key")}

    def analog(e, kind, o, cc, h0, n_h):
        earlier = [p for p in prev[e.key] if p.date < e.date][-NBLK:]
        if len(earlier) < 2:
            return None
        hrs = []
        for p in earlier:
            hrs += list(pd.date_range(p.date - pd.Timedelta(days=1), p.date + pd.Timedelta(hours=23), freq="h"))
        hrs += list(pd.date_range(e.date - pd.Timedelta(days=1), o, freq="h")); pos = hidx.reindex(hrs).values
        if not np.isfinite(pos).all():
            return None
        q, _ = pipe.predict_quantiles(torch.tensor(np.nan_to_num(Rser[pos.astype(int)][:, cc].T), dtype=torch.float32), prediction_length=H, quantile_levels=[0.5])
        return q[..., 0].float().cpu().numpy().T[h0:h0 + n_h]

    def components(e, kind, split):
        w = e19.window_forecasts(e, kind, split, dt, hidx, X, B, fc[split], vc, cpos, P)
        if w is None or w["P"] is None:
            return None
        M2 = w["M2f"] if split == "val" else w["M2r"]
        if np.isnan(M2).any():
            return None
        date, s = e.date, e.start_hour; w0 = max(s - 1, 0)
        o = date - pd.Timedelta(hours=1) if kind == "day-ahead" else [date + pd.Timedelta(hours=h) for h in (5, 11, 17) if h < w0][-1]
        h0 = int((date + pd.Timedelta(hours=w0) - o) / pd.Timedelta(hours=1)) - 1; cc = cpos[vc[e.venue]["500"]].values
        ZA = analog(e, kind, o, cc, h0, w["X"].shape[0])
        if ZA is None:
            return None
        return {"w": w, "M2": M2, "R6": w["R6"], "B1": 0.9 * M2 + 0.1 * w["R6"], "P": w["P"], "ZA": ZA}

    # ---- fit on 2024
    t0 = time.time(); samples = {}
    for e in ev[ev.date.dt.year == 2024].itertuples():
        for kind in ("day-ahead", "short"):
            c = components(e, kind, "val")
            if c is None:
                continue
            samples.setdefault((e.key, kind), []).append(c)
    print("2024 samples", {k: len(v) for k, v in samples.items()}, round((time.time() - t0) / 60, 1), "min", flush=True)

    def fit(cs, names):
        A_ = np.concatenate([np.stack([c[n].ravel() for n in names], 1) for c in cs]); y_ = np.concatenate([c["w"]["R"].ravel() for c in cs])
        coef, *_ = np.linalg.lstsq(A_, y_, rcond=None); return np.clip(coef, 0, 1.5)

    Wt = {}; wrows = []
    for combo, names in COMBOS.items():
        for kind in ("day-ahead", "short"):
            pooled = [c for (k, kd), cs in samples.items() if kd == kind for c in cs]; wp = fit(pooled, names) if pooled else np.array([1.0] + [0.0] * (len(names) - 1))
            Wt[(combo, "pooled", kind)] = wp; wrows.append({"combo": combo, "key": "pooled", "origin": kind, "n": len(pooled), **{f"w_{n}": float(v) for n, v in zip(names, wp)}})
            for (key, kd), cs in samples.items():
                if kd != kind:
                    continue
                if len(cs) >= MINFIT:
                    wk = fit(cs, names); Wt[(combo, key, kind)] = wk; wrows.append({"combo": combo, "key": key, "origin": kind, "n": len(cs), **{f"w_{n}": float(v) for n, v in zip(names, wk)}})
    pd.DataFrame(wrows).to_csv(OUT / "weights.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(wrows).round(2).to_string(), flush=True)

    # ---- evaluate 2025
    rows = []
    for e in ev[ev.date.dt.year == 2025].itertuples():
        for kind in ("day-ahead", "short"):
            c = components(e, kind, "test")
            if c is None:
                continue
            cands = {"M2": c["M2"], "B1": c["B1"], "ZA": c["ZA"], "P": c["P"]}
            for combo, names in COMBOS.items():
                wk = Wt.get((combo, e.key, kind), Wt[(combo, "pooled", kind)]); cands[combo] = sum(w_ * c[n] for w_, n in zip(wk, names))
                cands[combo + " (pooled)"] = sum(w_ * c[n] for w_, n in zip(Wt[(combo, "pooled", kind)], names))
            for cname, Rh in cands.items():
                rows.append({"venue": VEN_EN[e.venue], "type": e.type, "date": e.date.date(), "origin": kind, "cand": cname, "fitted": "venue" if (("EC", e.key, kind) in Wt) else "pooled", **e19.metrics(Rh, c["w"])})
    df = pd.DataFrame(rows); df.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    g = df.groupby(["venue", "type", "origin", "cand"]).agg(n=("date", "nunique"), wmape_area=("wmape_area", "mean"), peak_err=("peak_err", "mean"), retention=("retention", "median")).reset_index()
    g.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    cols = ["M2", "B1", "ZA", "EC", "B1+P", "B1+ZA", "B1+ZA+P", "M2+ZA+P", "ZA+P", "B1+ZA+P (pooled)"]
    for val in ("wmape_area", "peak_err", "retention"):
        print(val); print(g.pivot_table(index=["venue", "origin"], columns="cand", values=val)[cols].round(2).to_string(), flush=True)
    # event-weighted overall (all 2025 event windows pooled by sum of absolute errors / sum of population)
    ew = df.groupby(["origin", "cand"]).wmape_area.mean().unstack()[cols].round(3); print("mean over events\n", ew.to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
