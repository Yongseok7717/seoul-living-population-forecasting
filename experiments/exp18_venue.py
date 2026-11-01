"""Experiment 18 - scheduled-event case study: forecast accuracy around venues (stadiums, concert hall, rally square, fireworks) on
event days versus matched non-event days, 2025 test year, all models with the same 72 h history.

Models: R0 baseline (R_hat = 0), M2 fixed (seed 0, t16c_B_A5_lr0.003_wd0), M2 rolling (roll_Q*_A5), R6 zero-shot residual (exp17 npz, 72 h
context), B1 = 0.9 M2 + 0.1 R6 (both protocols, constant weight).  Origins: day-ahead (23:00 of the previous day) and the latest 6-hourly origin
before the event window ('short lead').  Metrics per event on the cells within 500 m / 1 km of the venue: area wMAPE over the event window
(start-1 .. end+1 h), signed peak error of the area population, and residual retention sum(R_hat)/sum(R).  Controls: same weekday +-1, +-2 weeks
without an event at that venue.

Usage:  python exp18_venue.py            (driver: runs the five forecast workers, then the evaluation)
        python exp18_venue.py --worker std | Q1 | Q2 | Q3 | Q4
Outputs results/exp18_venue/{fc_*.npz, events.csv, summary.csv, examples.csv} and results/figures/F10_venue.{pdf,png}.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp18_venue"; LOGS = OUT / "logs"; FIG = ROOT / "results" / "figures"
sys.path.insert(0, str(ROOT / "src")); import config as C   # noqa: E402
for d_ in (OUT, LOGS, FIG):
    d_.mkdir(parents=True, exist_ok=True)
E17 = ROOT / "results" / "exp17_ctx72"; ROLL_CKPT = ROOT / "results" / "exp01_rolling" / "ckpt"; ORIG_CKPT = ROOT / "results" / "base" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}
VENUES = {"서울월드컵경기장": (947125, 1952125), "KSPO돔(올림픽공원)": (967125, 1946625), "여의도한강공원": (949900, 1948300), "광화문광장": (953875, 1952875), "잠실야구장": (962650, 1946300)}
VEN_EN = {"서울월드컵경기장": "World Cup Stadium", "KSPO돔(올림픽공원)": "KSPO Dome", "여의도한강공원": "Yeouido Hangang Park", "광화문광장": "Gwanghwamun Square", "잠실야구장": "Jamsil Baseball Stadium"}
RADII = (500, 1000); H = 24; K = 72
MODELS = ["R0", "M2f", "R6f", "B1f", "M2r", "R6r", "B1r"]


def venue_cells(cells):
    from loader import load_grid
    g = load_grid(with_geometry=False); xy = g.loc[cells, ["CELL_X", "CELL_Y"]].values.astype(float)
    return {v: {r: np.where(np.hypot(xy[:, 0] - x, xy[:, 1] - y) <= r)[0] for r in RADII} for v, (x, y) in VENUES.items()}


def worker(tag):
    sys.path.insert(0, str(SRC))
    import torch
    from train_moe import MoEData
    from train_cap import build as build_cap
    dev = torch.device("cuda"); md = MoEData(K, H, "q95"); d = md.d; ts = d.windows("test", 6)
    vc = venue_cells(d.cells); sel = np.unique(np.concatenate([c[1000] for c in vc.values()])); S = len(sel)
    ck = (ORIG_CKPT / "t16c_B_A5_lr0.003_wd0.pt") if tag == "std" else (ROLL_CKPT / f"roll_{tag}_A5.pt")
    sd = torch.load(ck, map_location=dev); K_cl = int(d.cluster.max()) + 1
    m = build_cap(sd["arch"], K, H, K_cl, persist=sd.get("persist", True), step_cal=sd.get("step_cal", False), future_b=sd.get("future_b", True), cond=sd.get("cond", "func_x_cal")).to(dev)
    m.load_state_dict(sd["state"]); m.eval(); step_cal = sd.get("step_cal", False); cl = torch.tensor(d.cluster.astype(np.int64), device=dev)[sel]
    T = lambda v: torch.tensor(v, device=dev)
    Z = np.zeros((len(ts), H, S), np.float32); t0 = time.time()
    for i in range(0, len(ts), 8):
        tb = ts[i:i + 8]; xh, th, tfu, bf, y, dtf = d.batch("test", tb); B = xh.shape[0]
        xs = xh[:, :, sel].transpose(0, 2, 1, 3).astype(np.float32); bfn = bf[:, :, sel].transpose(0, 2, 1).astype(np.float32)
        cs = np.concatenate([th[:, -1, :11], tfu[:, :, :11].mean(1)], -1).astype(np.float32); tff = tfu[:, :, :11].reshape(B, -1).astype(np.float32)
        with torch.no_grad():
            z = m(T(xs), T(bfn), T(cs), T(tff), cl, T(th[:, :, :11].astype(np.float32)) if step_cal else None).permute(0, 2, 1)
        Z[i:i + B] = z.cpu().numpy()
        if i % 400 == 0:
            print(tag, i, len(ts), round((time.time() - t0) / 60, 1), "min", flush=True)
    RM2 = Z * d.r_std[sel][None, None, :]
    npz = E17 / ("chronos_std_residual.npz" if tag == "std" else f"chronos_{tag}_roll_residual.npz"); R6 = np.load(npz)["test"].astype(np.float32)
    assert len(R6) == len(ts), (len(R6), len(ts)); R6 = R6[:, :, sel]
    X = np.stack([d.X["test"][t + 1:t + 1 + H][:, sel] for t in ts]); Bt = np.stack([d.B["test"][t + 1:t + 1 + H][:, sel] for t in ts])
    dt = pd.to_datetime(d.ti["test"].datetime.values)[ts].astype("datetime64[s]").astype(str)
    np.savez_compressed(OUT / f"fc_{tag}.npz", ts=ts, sel=sel, cells=d.cells[sel], origin=dt, RM2=RM2, R6=R6, X=X, B=Bt)
    json.dump({v: {str(r): d.cells[c].tolist() for r, c in vc[v].items()} for v in vc}, open(OUT / "venue_cells.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", tag, len(ts), S, flush=True)


def call(tag, splits):
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    if splits:
        env["RD_SPLITS"] = ",".join(splits)
    print(time.strftime("%m-%d %H:%M"), "RUN", tag, flush=True)
    with open(LOGS / f"fc_{tag}.log", "w", encoding="utf-8") as lg:
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker", tag], stdout=lg, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)


def load_forecasts():
    """dict origin datetime -> index; arrays [T,H,S] per model in people (residual); X, B truth. Rolling = concatenation of the four folds."""
    f = np.load(OUT / "fc_std.npz", allow_pickle=True); sel = f["sel"]; cells = f["cells"]
    fixed = {"origin": pd.to_datetime(f["origin"]), "RM2": f["RM2"], "R6": f["R6"], "X": f["X"], "B": f["B"]}
    parts = []
    for q in FOLDS:
        g = np.load(OUT / f"fc_{q}.npz", allow_pickle=True); assert (g["sel"] == sel).all(); parts.append(g)
    roll = {"origin": pd.to_datetime(np.concatenate([g["origin"] for g in parts])), "RM2": np.concatenate([g["RM2"] for g in parts]), "R6": np.concatenate([g["R6"] for g in parts])}
    # rolling folds may miss a few origins at fold starts; align everything on the fixed origins
    idx = pd.Series(np.arange(len(roll["origin"])), index=roll["origin"]).reindex(fixed["origin"])
    ok = idx.notna().values; ii = idx.values[ok].astype(int)
    R = {"R0": np.zeros_like(fixed["RM2"]), "M2f": fixed["RM2"], "R6f": fixed["R6"], "M2r": np.full_like(fixed["RM2"], np.nan), "R6r": np.full_like(fixed["RM2"], np.nan)}
    R["M2r"][ok] = roll["RM2"][ii]; R["R6r"][ok] = roll["R6"][ii]
    R["B1f"] = 0.9 * R["M2f"] + 0.1 * R["R6f"]; R["B1r"] = 0.9 * R["M2r"] + 0.1 * R["R6r"]
    return fixed["origin"], cells, R, fixed["X"], fixed["B"]


def evaluate():
    origin, cells, R, X, Bt = load_forecasts(); oidx = pd.Series(np.arange(len(origin)), index=origin)
    vc = json.load(open(OUT / "venue_cells.json", encoding="utf-8")); cpos = pd.Series(np.arange(len(cells)), index=cells)
    ev = pd.read_csv(C.TABLES / "venue_events.csv", encoding="utf-8-sig"); ev = ev[(ev.date >= "2025-01-01") & (ev.date <= "2025-12-31") & ev.venue.isin(VENUES)].copy()
    ev["date"] = pd.to_datetime(ev.date); ev["start_hour"] = ev.start_hour.astype(int); ev["end_hour"] = ev.end_hour.astype(int)
    busy = {v: set(ev[ev.venue == v].date.dt.normalize()) for v in VENUES}
    rows = []

    def score(v, date, s, e, role, event_id, etype):
        """metrics for one (venue, day, window) under both origins, all models and radii"""
        w0, w1 = max(s - 1, 0), min(e + 1, 23)                                  # event window hours of the day
        cands = [("day-ahead", date - pd.Timedelta(hours=1))]                   # 23:00 of the previous day
        short = [date + pd.Timedelta(hours=h) for h in (5, 11, 17) if h < w0]
        if short:
            cands.append(("short", short[-1]))
        out = []
        for lead_name, o in cands:
            if o not in oidx.index:
                continue
            i = int(oidx[o]); h0 = int((date + pd.Timedelta(hours=w0) - o) / pd.Timedelta(hours=1)) - 1; h1 = int((date + pd.Timedelta(hours=w1) - o) / pd.Timedelta(hours=1)) - 1
            if h0 < 0 or h1 >= H:
                continue
            lead = (date + pd.Timedelta(hours=s) - o) / pd.Timedelta(hours=1)
            for r in RADII:
                cc = cpos[vc[v][str(r)]].values; Xw = X[i, h0:h1 + 1][:, cc]; Bw = Bt[i, h0:h1 + 1][:, cc]; Rw = Xw - Bw
                A = Xw.sum(1); base = {"venue": VEN_EN[v], "type": etype, "role": role, "event_id": event_id, "date": date.date(), "lead": lead_name, "lead_h": lead, "radius": r,
                                       "n_cells": len(cc), "hours": h1 - h0 + 1, "peak_obs": float(A.max()), "excess_obs": float(Rw.sum()), "base_excess_ratio": float(Rw.sum() / max(Bw.sum(), 1))}
                for mname in MODELS:
                    Rh = R[mname][i, h0:h1 + 1][:, cc]
                    if np.isnan(Rh).any():
                        continue
                    Ah = (Bw + Rh).sum(1)
                    out.append({**base, "model": mname, "wmape_area": float(np.abs(Ah - A).sum() / A.sum()), "wmape_cell": float(np.abs(Rh - Rw).sum() / Xw.sum()),
                                "peak_err": float((Ah.max() - A.max()) / A.max()), "retention": float(Rh.sum() / Rw.sum()) if abs(Rw.sum()) > 0 else np.nan})
        return out

    for k, e in enumerate(ev.sort_values(["venue", "date"]).itertuples()):
        rows += score(e.venue, e.date, e.start_hour, e.end_hour, "event", k, e.type)
        for wk in (-1, 1, -2, 2):                                                 # matched controls: same weekday, no event at the venue
            c = e.date + pd.Timedelta(weeks=wk)
            if c.year == 2025 and c not in busy[e.venue]:
                rows += score(e.venue, c, e.start_hour, e.end_hour, "control", k, e.type)
    df = pd.DataFrame(rows); df.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    # summary: per venue/type, lead, radius, model: number of events, mean metrics on events and on controls
    g = df.groupby(["venue", "type", "lead", "radius", "role", "model"]).agg(n=("event_id", "nunique"), wmape_area=("wmape_area", "mean"), wmape_cell=("wmape_cell", "mean"),
                                                                             peak_err=("peak_err", "mean"), abs_peak_err=("peak_err", lambda s: s.abs().mean()), retention=("retention", "median"),
                                                                             excess_obs=("excess_obs", "mean"), base_excess_ratio=("base_excess_ratio", "mean")).reset_index()
    g.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    piv = g[(g.radius == 500)].pivot_table(index=["venue", "lead", "role"], columns="model", values="wmape_area").round(3)
    print(piv.to_string(), flush=True)
    piv2 = g[(g.radius == 500) & (g.role == "event")].pivot_table(index=["venue", "lead"], columns="model", values="retention").round(2); print(piv2.to_string(), flush=True)
    return df, origin, cells, R, X, Bt, vc


def figure(df, origin, cells, R, X, Bt, vc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    oidx = pd.Series(np.arange(len(origin)), index=origin); cpos = pd.Series(np.arange(len(cells)), index=cells)
    ex = df[(df.role == "event") & (df.lead == "day-ahead") & (df.radius == 500) & (df.model == "R0")]
    picks = []
    for v in ("World Cup Stadium", "KSPO Dome", "Yeouido Hangang Park"):
        s = ex[ex.venue == v]
        if len(s):
            picks.append(s.sort_values("excess_obs").iloc[-1])
    fig, axes = plt.subplots(1, len(picks), figsize=(4.3 * len(picks), 4.4)); axes = np.atleast_1d(axes); rec = []
    short = {"World Cup Stadium": "World Cup Stadium", "KSPO Dome": "KSPO Dome", "Yeouido Hangang Park": "Yeouido"}
    for ax, p in zip(axes, picks):
        date = pd.Timestamp(p.date); o = date - pd.Timedelta(hours=1); i = int(oidx[o]); ven_kr = [k for k, vv in VEN_EN.items() if vv == p.venue][0]; cc = cpos[vc[ven_kr]["500"]].values
        A = X[i][:, cc].sum(1) / 1000; Bs = Bt[i][:, cc].sum(1) / 1000; hrs = np.arange(24)
        ax.plot(hrs, A, "k-", lw=3.2, label="observed"); ax.plot(hrs, Bs, color="0.55", ls="--", lw=2.4, label="baseline (R0)")
        for mname, col, lab in (("M2f", "tab:blue", "M2, fixed"), ("M2r", "tab:green", "M2, rolling"), ("R6f", "tab:red", "R6, zero-shot")):
            ax.plot(hrs, Bs + R[mname][i][:, cc].sum(1) / 1000, color=col, lw=2.4, label=lab)
        ax.set_title(f"{short[p.venue]}, {date.strftime('%d %b')}", fontsize=20); ax.set_xlabel("hour of the event day", fontsize=19); ax.tick_params(labelsize=17); ax.set_xlim(0, 23); ax.grid(alpha=0.3)
        rec.append({"venue": p.venue, "date": str(date.date()), "type": p.type, "peak_obs_k": float(A.max()), "peak_base_k": float(Bs.max())})
    axes[0].set_ylabel("population (thousands)", fontsize=19); axes[0].legend(fontsize=15, loc="upper left", handlelength=1.6, borderaxespad=0.3)
    fig.tight_layout(); fig.savefig(FIG / "F10_venue.pdf"); fig.savefig(FIG / "F10_venue.png", dpi=160); pd.DataFrame(rec).to_csv(OUT / "examples.csv", index=False, encoding="utf-8-sig"); print(rec, flush=True)


def main():
    for tag, fold in [("std", None)] + list(FOLDS.items()):
        if not (OUT / f"fc_{tag}.npz").exists():
            rc = call(tag, fold)
            if rc != 0 or not (OUT / f"fc_{tag}.npz").exists():
                print("FAILED", tag, rc, flush=True); return
    res = evaluate(); figure(*res); print("ALLDONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
