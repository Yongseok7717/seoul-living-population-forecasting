"""Experiment 20 - zero-shot forecasting with an 'analog context': for a scheduled event at a venue, Chronos-Bolt receives, instead of the last
72 h, the concatenation of the last NBLK previous events of the same venue and type (48-h blocks: the day before and the event day, residual
or raw series of each cell) followed by the current history from 00:00 of the previous day to the origin.  Evaluated on the 2025 events as in
exp18/exp19 (day-ahead and short-lead origins, cells within 500 m).  Candidates: R6 (72 h), R6-analog, R5-analog, plus the exp19 M2-based
fitted combination for reference.  Outputs results/exp20_chronos_analog/{events.csv, summary.csv}.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "exp20_chronos_analog"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments"))
from exp18_venue import VENUES, VEN_EN  # noqa: E402
import exp19_event_profile as e19  # noqa: E402

NBLK, H, MODEL = 10, 24, "amazon/chronos-bolt-small"; MINPREV = 2


def main():
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
    dt, hidx, X, B, Rser, fc, vc, cpos = e19.load(); ev = e19.events_table(); P = e19.profiles(ev, dt, hidx, Rser, vc, cpos)
    e19_ev = pd.read_csv(ROOT / "results" / "exp19_event_profile" / "events.csv"); e19_ev["date"] = pd.to_datetime(e19_ev.date)
    prev = {key: list(grp.itertuples()) for key, grp in ev.groupby("key")}

    def predict(series_ctx):
        """series_ctx [cells, L] -> median forecast [H, cells]"""
        q, _ = pipe.predict_quantiles(torch.tensor(series_ctx, dtype=torch.float32), prediction_length=H, quantile_levels=[0.5])
        return q[..., 0].float().cpu().numpy().T

    rows = []; t0 = time.time()
    for e in ev[ev.date.dt.year == 2025].itertuples():
        earlier = [p for p in prev[e.key] if p.date < e.date][-NBLK:]
        if len(earlier) < MINPREV:
            continue
        for kind in ("day-ahead", "short"):
            w = e19.window_forecasts(e, kind, "test", dt, hidx, X, B, fc["test"], vc, cpos, P)
            if w is None or np.isnan(w["M2r"]).any():
                continue
            date = e.date; w0 = max(e.start_hour - 1, 0)
            o = date - pd.Timedelta(hours=1) if kind == "day-ahead" else [date + pd.Timedelta(hours=h) for h in (5, 11, 17) if h < w0][-1]
            cc = cpos[vc[e.venue]["500"]].values
            # blocks of previous events: [d_j - 1 day 00:00, d_j 23:00]; current: [date - 1 day 00:00, o]
            hrs = []
            for p in earlier:
                hrs += list(pd.date_range(p.date - pd.Timedelta(days=1), p.date + pd.Timedelta(hours=23), freq="h"))
            hrs += list(pd.date_range(date - pd.Timedelta(days=1), o, freq="h"))
            pos = hidx.reindex(hrs).values
            if not np.isfinite(pos).all():
                continue
            pos = pos.astype(int); ctxR = np.nan_to_num(Rser[pos][:, cc].T); ctxX = np.nan_to_num(X[pos][:, cc].T)
            h0 = int((date + pd.Timedelta(hours=w0) - o) / pd.Timedelta(hours=1)) - 1; h1 = h0 + w["X"].shape[0] - 1
            fR = predict(ctxR)[h0:h1 + 1]                                   # residual forecast in people
            Bw = w["B"]; fX = np.maximum(predict(ctxX)[h0:h1 + 1], 0) - Bw    # raw forecast -> residual-equivalent
            sub = e19_ev[(e19_ev.venue == VEN_EN[e.venue]) & (e19_ev.date == date) & (e19_ev.origin == kind)]
            cands = {"R6-analog": fR, "R5-analog": fX, "R6": w["R6"], "M2": w["M2r"]}
            for cname, Rh in cands.items():
                rows.append({"venue": VEN_EN[e.venue], "type": e.type, "date": date.date(), "origin": kind, "cand": cname, "n_blocks": len(earlier), "ctx_len": len(hrs), **e19.metrics(Rh, w)})
            for cname in ("fit", "fitR6"):
                r = sub[sub.cand == cname]
                if len(r):
                    rows.append({"venue": VEN_EN[e.venue], "type": e.type, "date": date.date(), "origin": kind, "cand": cname, "n_blocks": len(earlier), "ctx_len": len(hrs), "wmape_area": float(r.wmape_area.iloc[0]), "peak_err": float(r.peak_err.iloc[0]), "retention": float(r.retention.iloc[0])})
        if len(rows) % 200 < 8:
            print(len(rows), round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    g = df.groupby(["venue", "type", "origin", "cand"]).agg(n=("date", "nunique"), wmape_area=("wmape_area", "mean"), peak_err=("peak_err", "mean"), retention=("retention", "median")).reset_index()
    g.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    for val in ("wmape_area", "peak_err", "retention"):
        print(val); print(g.pivot_table(index=["venue", "origin"], columns="cand", values=val)[["M2", "R6", "R6-analog", "R5-analog", "fit", "fitR6"]].round(2).to_string(), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
