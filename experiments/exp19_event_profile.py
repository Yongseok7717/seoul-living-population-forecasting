"""Experiment 19 - event-conditional forecasting at fixed venues: learn, per (venue, event type), the residual profile of past event days
and combine it with the model forecast.

Profile P[c, r]: causal mean of the residual R = X - B on the last 12 previous events of the same venue and type, for cells c within 1 km of
the venue and relative hours r = hour - start_hour in [-6, +9].  Candidates on the event-window hours of the venue cells:
    M2      model alone (rolling M2 in 2025, fixed M2 in 2024)          R6      zero-shot alone
    P       B + P (profile alone)                                       M2+P    B + R_M2 + P          R6+P
    fit     B + a R_M2 + b P with (a, b) fitted by least squares on the 2024 events (per venue-type and origin), clipped to [0, 1.5]
Evaluation as in exp18 (day-ahead origin 23:00, short lead = latest six-hourly origin before the window; area wMAPE, peak error, retention).
Outputs results/exp19_event_profile/{summary.csv, events.csv, weights.csv} and figures/paper/F11_venue_profile.{pdf,png}.
Usage: python exp19_event_profile.py [--worker]
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp19_event_profile"; FIG = ROOT / "results" / "figures"
sys.path.insert(0, str(ROOT / "src")); import config as C   # noqa: E402
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
E17 = ROOT / "results" / "exp17_ctx72"; E18 = ROOT / "results" / "exp18_venue"; ORIG_CKPT = ROOT / "results" / "base" / "ckpt"
sys.path.insert(0, str(ROOT / "experiments"))
from exp18_venue import VENUES, VEN_EN, venue_cells  # noqa: E402

H, K = 24, 72; RMIN, RMAX = -6, 9; NPREV = 12; MINPREV = 2
CANDS = ["M2", "R6", "B1", "P", "M2+P", "R6+P", "B1+P", "fit", "fitB1", "fitR6"]


def worker():
    """fixed M2 (seed 0) forecasts on the venue cells for val (2024) and test (2025), the R6 forecasts, and the full hourly X, B series of the venue cells."""
    sys.path.insert(0, str(SRC))
    import torch
    from train_moe import MoEData
    from train_cap import build as build_cap
    dev = torch.device("cuda"); md = MoEData(K, H, "q95"); d = md.d
    vc = venue_cells(d.cells); sel = np.unique(np.concatenate([c[1000] for c in vc.values()])); S = len(sel)
    sd = torch.load(ORIG_CKPT / "t16c_B_A5_lr0.003_wd0.pt", map_location=dev); K_cl = int(d.cluster.max()) + 1
    m = build_cap(sd["arch"], K, H, K_cl, persist=sd.get("persist", True), step_cal=sd.get("step_cal", False), future_b=sd.get("future_b", True), cond=sd.get("cond", "func_x_cal")).to(dev)
    m.load_state_dict(sd["state"]); m.eval(); step_cal = sd.get("step_cal", False); cl = torch.tensor(d.cluster.astype(np.int64), device=dev)[sel]; T = lambda v: torch.tensor(v, device=dev)
    R6all = np.load(E17 / "chronos_std_residual.npz"); out = {"sel": sel, "cells": d.cells[sel]}
    for split in ("val", "test"):
        ts = d.windows(split, 6); Z = np.zeros((len(ts), H, S), np.float32)
        for i in range(0, len(ts), 8):
            tb = ts[i:i + 8]; xh, th, tfu, bf, y, dtf = d.batch(split, tb); B = xh.shape[0]
            xs = xh[:, :, sel].transpose(0, 2, 1, 3).astype(np.float32); bfn = bf[:, :, sel].transpose(0, 2, 1).astype(np.float32)
            cs = np.concatenate([th[:, -1, :11], tfu[:, :, :11].mean(1)], -1).astype(np.float32); tff = tfu[:, :, :11].reshape(B, -1).astype(np.float32)
            with torch.no_grad():
                Z[i:i + B] = m(T(xs), T(bfn), T(cs), T(tff), cl, T(th[:, :, :11].astype(np.float32)) if step_cal else None).permute(0, 2, 1).cpu().numpy()
        R6 = R6all[split].astype(np.float32); assert len(R6) == len(ts); out[f"RM2_{split}"] = Z * d.r_std[sel][None, None, :]; out[f"R6_{split}"] = R6[:, :, sel]
        out[f"origin_{split}"] = pd.to_datetime(d.ti[split].datetime.values)[ts].astype("datetime64[s]").astype(str); print(split, len(ts), flush=True)
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = d.X[split][:, sel].astype(np.float32); out[f"B_{split}"] = d.B[split][:, sel].astype(np.float32); out[f"dt_{split}"] = pd.to_datetime(d.ti[split].datetime.values).astype("datetime64[s]").astype(str)
    np.savez_compressed(OUT / "fc_std.npz", **out); print("saved", S, flush=True)


def load():
    f = np.load(OUT / "fc_std.npz", allow_pickle=True); cells = f["cells"]; sel = f["sel"]
    dt = pd.to_datetime(np.concatenate([f[f"dt_{s}"] for s in ("train", "val", "test")])); X = np.concatenate([f[f"X_{s}"] for s in ("train", "val", "test")]); B = np.concatenate([f[f"B_{s}"] for s in ("train", "val", "test")])
    ser = pd.DataFrame(index=dt); Rser = X - B; hidx = pd.Series(np.arange(len(dt)), index=dt)
    fc = {}
    for s in ("val", "test"):
        fc[s] = {"origin": pd.Series(np.arange(len(f[f"origin_{s}"])), index=pd.to_datetime(f[f"origin_{s}"])), "M2f": f[f"RM2_{s}"], "R6": f[f"R6_{s}"]}
    g = np.load(E18 / "fc_std.npz", allow_pickle=True); assert (g["sel"] == sel).all()
    parts = [np.load(E18 / f"fc_{q}.npz", allow_pickle=True) for q in ("Q1", "Q2", "Q3", "Q4")]
    ro = pd.to_datetime(np.concatenate([p["origin"] for p in parts])); rM = np.concatenate([p["RM2"] for p in parts])
    idx = pd.Series(np.arange(len(ro)), index=ro).reindex(fc["test"]["origin"].index); ok = idx.notna().values
    M2r = np.full_like(fc["test"]["M2f"], np.nan); M2r[ok] = rM[idx.values[ok].astype(int)]; fc["test"]["M2r"] = M2r
    vc = json.load(open(E18 / "venue_cells.json", encoding="utf-8")); cpos = pd.Series(np.arange(len(cells)), index=cells)
    return dt, hidx, X, B, Rser, fc, vc, cpos


def events_table():
    ev = pd.read_csv(C.TABLES / "venue_events.csv", encoding="utf-8-sig"); ev = ev[ev.venue.isin(VENUES)].copy()
    ev["date"] = pd.to_datetime(ev.date); ev["start_hour"] = ev.start_hour.astype(int); ev["end_hour"] = ev.end_hour.astype(int)
    ev = ev[(ev.date >= "2023-03-01") & (ev.date <= "2025-12-31")].sort_values(["venue", "type", "date"]).reset_index(drop=True)
    ev["key"] = ev.venue + "|" + ev.type
    return ev


def profiles(ev, dt, hidx, Rser, vc, cpos):
    """causal profile for every event: mean residual [cells, RMAX-RMIN+1] over the previous NPREV events of the same key (NaN where < MINPREV)."""
    mats = {}; P = {}
    for key, grp in ev.groupby("key"):
        v = key.split("|")[0]; cc = cpos[vc[v]["1000"]].values; hist = []
        for e in grp.itertuples():
            t0 = e.date + pd.Timedelta(hours=e.start_hour); hrs = [t0 + pd.Timedelta(hours=r) for r in range(RMIN, RMAX + 1)]
            pos = hidx.reindex(hrs).values
            M = np.full((len(cc), RMAX - RMIN + 1), np.nan, np.float32)
            good = np.isfinite(pos)
            if good.any():
                M[:, good] = Rser[pos[good].astype(int)][:, cc].T
            P[e.Index] = np.nanmean(np.stack(hist[-NPREV:]), 0) if len(hist) >= MINPREV else None
            if good.all() and not np.isnan(M).any():
                hist.append(M)
            mats[e.Index] = M
    return P


def window_forecasts(e, origin_hour_offset_kind, fc_split, dt, hidx, X, B, fcd, vc, cpos, P, radius="500"):
    """returns dict of residual forecasts [hours, cells] for the event window under one origin, plus truth; None if the origin is unavailable"""
    date, s, en = e.date, e.start_hour, e.end_hour; w0, w1 = max(s - 1, 0), min(en + 1, 23)
    if origin_hour_offset_kind == "day-ahead":
        o = date - pd.Timedelta(hours=1)
    else:
        short = [date + pd.Timedelta(hours=h) for h in (5, 11, 17) if h < w0]
        if not short:
            return None
        o = short[-1]
    if o not in fcd["origin"].index:
        return None
    i = int(fcd["origin"][o]); h0 = int((date + pd.Timedelta(hours=w0) - o) / pd.Timedelta(hours=1)) - 1; h1 = int((date + pd.Timedelta(hours=w1) - o) / pd.Timedelta(hours=1)) - 1
    if h0 < 0 or h1 >= H:
        return None
    cc = cpos[vc[e.venue][radius]].values; cc1 = cpos[vc[e.venue]["1000"]].values; pos_in_1k = pd.Series(np.arange(len(cc1)), index=cc1)[cc].values
    hrs = [o + pd.Timedelta(hours=h + 1) for h in range(h0, h1 + 1)]; pos = hidx.reindex(hrs).values.astype(int)
    Xw = X[pos][:, cc]; Bw = B[pos][:, cc]
    out = {"X": Xw, "B": Bw, "R": Xw - Bw, "lead": (date + pd.Timedelta(hours=s) - o) / pd.Timedelta(hours=1)}
    for mname in ("M2f", "M2r", "R6"):
        if mname in fcd:
            out[mname] = fcd[mname][i, h0:h1 + 1][:, cc]
    Pm = P.get(e.Index)
    if Pm is None:
        out["P"] = None
    else:
        rel = np.array([int((h - (date + pd.Timedelta(hours=s))) / pd.Timedelta(hours=1)) for h in hrs]); Pw = np.zeros((len(hrs), len(cc)), np.float32)
        ok = (rel >= RMIN) & (rel <= RMAX); Pw[ok] = np.nan_to_num(Pm[pos_in_1k][:, rel[ok] - RMIN].T); out["P"] = Pw
    return out


def metrics(Rh, w):
    A = w["X"].sum(1); Ah = (w["B"] + Rh).sum(1); Rs = w["R"].sum()
    return {"wmape_area": float(np.abs(Ah - A).sum() / A.sum()), "peak_err": float((Ah.max() - A.max()) / A.max()), "retention": float(Rh.sum() / Rs) if abs(Rs) > 0 else np.nan}


def main():
    if not (OUT / "fc_std.npz").exists():
        env = dict(os.environ, PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=str(SRC), env=env)
        if rc != 0:
            print("worker failed", rc); return
    dt, hidx, X, B, Rser, fc, vc, cpos = load(); ev = events_table(); P = profiles(ev, dt, hidx, Rser, vc, cpos)
    print("events", ev.groupby("key").size().to_dict(), "with profile:", sum(v is not None for v in P.values()), flush=True)
    # ---- fit (a, b) on 2024 events: R ~ a * R_M2f + b * P, per key and origin kind
    W = {}; WB = {}; WR = {}; wrows = []
    for key, grp in ev[ev.date.dt.year == 2024].groupby("key"):
        for kind in ("day-ahead", "short"):
            xs, ys, xb, xr = [], [], [], []
            for e in grp.itertuples():
                w = window_forecasts(e, kind, "val", dt, hidx, X, B, fc["val"], vc, cpos, P)
                if w is None or w["P"] is None:
                    continue
                xs.append(np.stack([w["M2f"].ravel(), w["P"].ravel()], 1)); ys.append(w["R"].ravel()); xb.append(np.stack([(0.9 * w["M2f"] + 0.1 * w["R6"]).ravel(), w["P"].ravel()], 1)); xr.append(np.stack([w["R6"].ravel(), w["P"].ravel()], 1))
            if xs:
                A_ = np.concatenate(xs); y_ = np.concatenate(ys); coef, *_ = np.linalg.lstsq(A_, y_, rcond=None); a, b = np.clip(coef, 0, 1.5)
                coefb, *_ = np.linalg.lstsq(np.concatenate(xb), y_, rcond=None); ab, bb = np.clip(coefb, 0, 1.5)
                coefr, *_ = np.linalg.lstsq(np.concatenate(xr), y_, rcond=None); ar, br = np.clip(coefr, 0, 1.5)
            else:
                a, b = 1.0, 0.0; ab, bb = 1.0, 0.0; ar, br = 1.0, 0.0
            WB[(key, kind)] = (float(ab), float(bb)); WR[(key, kind)] = (float(ar), float(br))
            W[(key, kind)] = (float(a), float(b)); wrows.append({"key": key, "venue": VEN_EN[key.split("|")[0]], "type": key.split("|")[1], "origin": kind, "a_model": a, "b_profile": b, "a_blend": ab, "b_profile_blend": bb, "a_r6": ar, "b_profile_r6": br, "n_events_2024": len(xs)})
    pd.DataFrame(wrows).to_csv(OUT / "weights.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(wrows).round(2).to_string(), flush=True)
    # ---- evaluate 2025 events
    rows = []
    for e in ev[ev.date.dt.year == 2025].itertuples():
        for kind in ("day-ahead", "short"):
            w = window_forecasts(e, kind, "test", dt, hidx, X, B, fc["test"], vc, cpos, P)
            if w is None or w["P"] is None or np.isnan(w["M2r"]).any():
                continue
            a, b = W.get((e.key, kind), (1.0, 0.0)); ab, bb = WB.get((e.key, kind), (1.0, 0.0)); ar, br = WR.get((e.key, kind), (1.0, 0.0)); Pw = w["P"]; B1 = 0.9 * w["M2r"] + 0.1 * w["R6"]
            cands = {"M2": w["M2r"], "R6": w["R6"], "B1": B1, "P": Pw, "M2+P": w["M2r"] + Pw, "R6+P": w["R6"] + Pw, "B1+P": B1 + Pw, "fit": a * w["M2r"] + b * Pw, "fitB1": ab * B1 + bb * Pw, "fitR6": ar * w["R6"] + br * Pw}
            for cname, Rh in cands.items():
                rows.append({"venue": VEN_EN[e.venue], "type": e.type, "date": e.date.date(), "origin": kind, "lead_h": w["lead"], "cand": cname, "peak_obs": float(w["X"].sum(1).max()), "excess_obs": float(w["R"].sum()), **metrics(Rh, w)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    g = df.groupby(["venue", "type", "origin", "cand"]).agg(n=("date", "nunique"), lead_h=("lead_h", "mean"), wmape_area=("wmape_area", "mean"), peak_err=("peak_err", "mean"), abs_peak_err=("peak_err", lambda s: s.abs().mean()), retention=("retention", "median")).reset_index()
    g.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    for val in ("wmape_area", "peak_err", "retention"):
        print(val); print(g.pivot_table(index=["venue", "origin"], columns="cand", values=val)[CANDS].round(2).to_string(), flush=True)
    # ---- figure: the three example days of exp18 with the event-conditional forecasts
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ex = pd.read_csv(E18 / "examples.csv"); fig, axes = plt.subplots(1, len(ex), figsize=(4.3 * len(ex), 4.4)); axes = np.atleast_1d(axes)
    short = {"World Cup Stadium": "World Cup Stadium", "KSPO Dome": "KSPO Dome", "Yeouido Hangang Park": "Yeouido"}
    for ax, p in zip(axes, ex.itertuples()):
        ven_kr = [k for k, vv in VEN_EN.items() if vv == p.venue][0]; e = ev[(ev.venue == ven_kr) & (ev.date == pd.Timestamp(p.date))].iloc[0]; e = next(ev.loc[[e.name]].itertuples())
        date = pd.Timestamp(p.date); o = date - pd.Timedelta(hours=1); i = int(fc["test"]["origin"][o]); cc = cpos[vc[ven_kr]["500"]].values
        hrs = [o + pd.Timedelta(hours=h + 1) for h in range(H)]; pos = hidx.reindex(hrs).values.astype(int); A = X[pos][:, cc].sum(1) / 1000; Bs = B[pos][:, cc].sum(1) / 1000
        Pm = P.get(e.Index); rel = np.array([int((h - (date + pd.Timedelta(hours=e.start_hour))) / pd.Timedelta(hours=1)) for h in hrs]); Pw = np.zeros((H, len(cc)), np.float32)
        cc1 = cpos[vc[ven_kr]["1000"]].values; pin = pd.Series(np.arange(len(cc1)), index=cc1)[cc].values
        if Pm is not None:
            ok = (rel >= RMIN) & (rel <= RMAX); Pw[ok] = np.nan_to_num(Pm[pin][:, rel[ok] - RMIN].T)
        a, b = W.get((e.key, "day-ahead"), (1.0, 0.0)); M2 = fc["test"]["M2r"][i][:, cc]
        ax.plot(np.arange(H), A, "k-", lw=3.2, label="observed"); ax.plot(np.arange(H), Bs, color="0.55", ls="--", lw=2.4, label="baseline (R0)")
        ax.plot(np.arange(H), Bs + M2.sum(1) / 1000, color="tab:green", lw=2.4, label="M2, rolling")
        ax.plot(np.arange(H), Bs + (M2 + Pw).sum(1) / 1000, color="tab:orange", lw=2.4, label="M2 + event profile")
        ax.plot(np.arange(H), Bs + (a * M2 + b * Pw).sum(1) / 1000, color="tab:purple", lw=2.4, ls="-.", label="fitted combination")
        ax.set_title(f"{short[p.venue]}, {date.strftime('%d %b')}", fontsize=20); ax.set_xlabel("hour of the event day", fontsize=19); ax.tick_params(labelsize=17); ax.set_xlim(0, 23); ax.grid(alpha=0.3)
    axes[0].set_ylabel("population (thousands)", fontsize=19); axes[0].legend(fontsize=14, loc="upper left", handlelength=1.6, borderaxespad=0.3)
    fig.tight_layout(); fig.savefig(FIG / "F11_venue_profile.pdf"); fig.savefig(FIG / "F11_venue_profile.png", dpi=160); print("ALLDONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        main()
