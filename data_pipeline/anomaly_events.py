"""F3-F6: standardized local anomaly scores, extreme-event extraction and catalog, GRU residual vs anomalies.

    python anomaly_events.py --years 2023,2024 --tau 3 --top 200

z_{t,i} = (X_{t,i} - mu_{i,h,d}) / (sigma_{i,h,d} + sqrt(mu + 1)), with mu = causal rolling median (8 weeks, hour x daytype)
and sigma = the matching rolling MAD (build_anomaly_scale.py). Events: |z| > tau, positive (surge) / negative (drop).
Catalog: connected components of flagged cell-hours (same cell in consecutive hours, or cells within 500 m in the same
hour) -> start/end, duration, cells, peak z / cell / time, direction, excess people, dong / gu.
F5: Corr(|R_GRU|, |z|), overlap of the top-1% GRU residuals with events (needs ml/predict_residual.py outputs).
Outputs: processed/eda/events/{z_<years>.npy (float16), events_all.csv, events_top<K>.csv/.md, z_stats.md, figures}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import config as C                                                              # noqa: E402
from loader import load_pop10, load_grid, load_adjacency, months_between, anomalous_days, to_lattice   # noqa: E402
from build_baseline_atlas import HOLIDAY_SET                                    # noqa: E402
from residual_data import RES_DIR, calendar                                     # noqa: E402

OUT = C.OUT / "eda" / "events"; REL = C.OUT / "relations"; ML_RES = C.BASE / "results"   # optional GRU predictions for the F5 comparison
GU = {"11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구",
      "11320": "도봉구", "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구", "11530": "구로구", "11545": "금천구",
      "11560": "영등포구", "11590": "동작구", "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구"}


def load_sparse(path, key):
    z = np.load(path, allow_pickle=True)
    return sp.csr_matrix((z[f"{key}__data"], z[f"{key}__indices"], z[f"{key}__indptr"]), shape=tuple(z[f"{key}__shape"]))


def components(flag, W, dt_hours):
    """flag [T, N] bool; W [N, N] symmetric adjacency (csr); dt_hours [T-1] hour gaps between consecutive rows -> labels [T, N] (-1 = none)"""
    T, N = flag.shape; idx = np.full((T, N), -1, np.int64); n = int(flag.sum()); idx[flag] = np.arange(n)
    rows, cols = [], []
    both = flag[:-1] & flag[1:] & (dt_hours == 1)[:, None]
    a = idx[:-1][both]; b = idx[1:][both]; rows.append(a); cols.append(b)
    Wc = W.tocsr()
    for t in np.where(flag.any(1))[0]:
        f = np.where(flag[t])[0]
        if len(f) < 2:
            continue
        sub = Wc[f][:, f].tocoo()
        if sub.nnz:
            rows.append(idx[t, f[sub.row]]); cols.append(idx[t, f[sub.col]])
    rows = np.concatenate(rows) if rows else np.array([], np.int64); cols = np.concatenate(cols) if cols else np.array([], np.int64)
    G = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n)).tocsr()
    _, lab = connected_components(G, directed=False)
    out = np.full((T, N), -1, np.int64); out[flag] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2023,2024"); ap.add_argument("--tau", type=float, default=3.0); ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--pred", default="r_k24_none"); ap.add_argument("--min_cellhours", type=int, default=2); ap.add_argument("--min_excess", type=float, default=30.0); ap.add_argument("--adjacency", default="rook")
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    years = args.years.split(","); ytag = "_".join(years)
    grid = load_grid(with_geometry=False)
    lab = pd.read_parquet(C.OUT / "eda" / "cluster_labels.parquet"); cells = lab.cell_idx.values; N = len(cells)
    months = [m for y in years for m in months_between(f"{y}01", f"{y}12")]
    X, ti = load_pop10(months); X = X.astype(np.float32)[:, cells]
    B = np.concatenate([np.load(RES_DIR / f"baseline_{m}.npy") for m in months])[:, cells].astype(np.float32)
    SC = np.concatenate([np.load(RES_DIR / f"scale_{m}.npy") for m in months])[:, cells].astype(np.float32)
    valid = ti.available.values & ~ti.ymd.isin(anomalous_days()).values & ~np.isnan(X[:, 0]) & ~np.isnan(B[:, 0]) & ~np.isnan(SC[:, 0])
    X, B, SC, tiv = X[valid], B[valid], SC[valid], ti[valid].reset_index(drop=True); T = len(tiv)
    dtime = pd.to_datetime(tiv.datetime); dt_hours = (np.diff(dtime.values).astype("timedelta64[h]").astype(int))
    z = ((X - B) / (SC + np.sqrt(B + 1))).astype(np.float32); excess = X - B
    np.save(OUT / f"z_{ytag}.npy", z.astype(np.float16)); tiv[["ymd", "tt", "datetime"]].to_csv(OUT / f"time_{ytag}.csv", index=False); np.save(OUT / f"cells_{ytag}.npy", cells)
    _, dtype_ = calendar(tiv)
    lines = [f"# F3: standardized local anomaly score z, years {args.years}", "",
             f"{T} valid hours x {N} cells. z = (X - mu)/(sigma + sqrt(mu+1)); mu = rolling median (8 w, hour x daytype), sigma = rolling MAD.", "",
             "## distribution", "", "| statistic | value |", "|---|---|"]
    q = lambda p: float(np.quantile(z, p))
    lines += [f"| mean / std of z | {z.mean():.3f} / {z.std():.3f} |", f"| quantiles 0.1% / 1% / 50% / 99% / 99.9% | {q(.001):.2f} / {q(.01):.2f} / {q(.5):.2f} / {q(.99):.2f} / {q(.999):.2f} |",
              f"| share |z| > 2 / 3 / 5 | {(np.abs(z) > 2).mean():.4f} / {(np.abs(z) > 3).mean():.4f} / {(np.abs(z) > 5).mean():.4f} |",
              f"| share z > 3 (surge) / z < -3 (drop) | {(z > 3).mean():.4f} / {(z < -3).mean():.4f} |",
              f"| top-1% / top-0.1% threshold of |z| | {np.quantile(np.abs(z), .99):.2f} / {np.quantile(np.abs(z), .999):.2f} |", ""]
    # is the score free of the 'always-busy cell' problem? compare cell concentration of top-1% |z| vs top-1% X
    for name, sc in (("|z|", np.abs(z)), ("X (absolute level)", X)):
        m = sc >= np.quantile(sc, .99); cnt = m.sum(0)
        lines.append(f"- top-1% {name}: {(cnt > 0).mean():.1%} of cells ever flagged; top-5% cells hold {np.sort(cnt)[::-1][:int(.05*N)].sum()/m.sum():.1%} of flags; cells flagged > 500 h: {(cnt > 500).sum()}")
    flag_raw = np.abs(z) > args.tau
    # event membership: |z| > tau AND material excess AND not a persistent shift (cell flagged > 50% of the previous 168 h)
    cs = np.concatenate([np.zeros((1, N), np.int32), np.cumsum(flag_raw, 0, dtype=np.int32)])
    past = np.zeros((T, N), np.float32); t0 = np.maximum(np.arange(T) - 168, 0); past[:] = (cs[np.arange(T)] - cs[t0]) / np.maximum(np.arange(T) - t0, 1)[:, None]
    persistent = past > 0.5; material = np.abs(excess) >= args.min_excess
    flag = flag_raw & material & ~persistent
    lines += ["", f"event membership: |z| > {args.tau:g} ({flag_raw.mean():.3%}) and |X - mu| >= {args.min_excess:g} people ({(flag_raw & material).mean():.3%}) and not persistent "
              f"(cell flagged > 50% of the previous 168 h; removes {(flag_raw & material & persistent).mean():.3%}) -> {flag.mean():.3%} of cell-hours"]
    byd = pd.DataFrame({"daytype": dtype_, "hour": tiv.tt.values, "pos": (z > args.tau).sum(1), "neg": (z < -args.tau).sum(1)})
    lines += ["", f"## flagged cell-hours (|z| > {args.tau:g}) by daytype (0 weekday, 1 saturday, 2 sunday/holiday): mean per hour", "",
              byd.groupby("daytype")[["pos", "neg"]].mean().round(1).to_markdown(), "", "by hour of day (mean flagged cells):", "",
              byd.groupby("hour")[["pos", "neg"]].mean().round(1).T.to_markdown(), ""]
    # ---- F6: catalog ----
    W = load_adjacency(args.adjacency) if args.adjacency in ("rook", "queen") else load_sparse(REL / f"spatial_{args.adjacency}.npz", "W")
    W = W[cells][:, cells]; W = ((W + W.T) > 0).astype(np.int8).tocsr()
    same_day = (tiv.ymd.values[1:] == tiv.ymd.values[:-1])
    links = np.where(same_day, dt_hours, 0)                                          # temporal links only within a calendar day
    labs_p = components(flag & (z > 0), W, links); labs_n = components(flag & (z < 0), W, links)   # surges and drops are separate events
    labs = np.where(labs_p >= 0, labs_p, np.where(labs_n >= 0, labs_n + (labs_p.max() + 1), -1))
    day_share = pd.Series((flag_raw & material).mean(1), index=tiv.ymd.values).groupby(level=0).mean()   # share of cells flagged, daily mean
    dmap = pd.read_parquet(C.OUT / "grid" / "cell_dong_map.parquet").sort_values("share").groupby("cell_idx").last()
    dong = dmap["dong_name"].reindex(cells).fillna("").values; gu = dmap["H_DNG_CD"].astype(str).str[:5].map(GU).reindex(cells).fillna("").values
    tt_, ii_ = np.where(labs >= 0); lb = labs[tt_, ii_]
    df = pd.DataFrame({"comp": lb, "t": tt_, "i": ii_, "z": z[tt_, ii_], "excess": excess[tt_, ii_], "X": X[tt_, ii_], "mu": B[tt_, ii_]})
    g = df.groupby("comp")
    pk = df.loc[df.groupby("comp")["z"].apply(lambda s: s.abs().idxmax())]
    ev = pd.DataFrame({"n_cellhours": g.size(), "n_cells": g["i"].nunique(), "t0": g["t"].min(), "t1": g["t"].max(),
                       "sum_excess": g["excess"].sum(), "sum_abs_excess": g["excess"].apply(lambda s: s.abs().sum()), "mean_z": g["z"].mean()})
    ev["peak_z"] = pk.set_index("comp")["z"]; ev["peak_cell_pos"] = pk.set_index("comp")["i"]; ev["peak_t"] = pk.set_index("comp")["t"]
    ev["peak_X"] = pk.set_index("comp")["X"]; ev["peak_mu"] = pk.set_index("comp")["mu"]
    ev["direction"] = np.where(ev.peak_z > 0, "increase", "decrease"); ev["duration_h"] = (dtime.values[ev.t1] - dtime.values[ev.t0]).astype("timedelta64[h]").astype(int) + 1
    ev["date"] = dtime.dt.strftime("%Y-%m-%d").values[ev.t0]; ev["start"] = dtime.dt.strftime("%H:%M").values[ev.t0]; ev["end"] = dtime.dt.strftime("%H:%M").values[ev.t1]
    ev["end_date"] = dtime.dt.strftime("%Y-%m-%d").values[ev.t1]; ev["peak_time"] = dtime.dt.strftime("%Y-%m-%d %H:%M").values[ev.peak_t]
    ev["peak_cell"] = cells[ev.peak_cell_pos.values]; ev["dong"] = dong[ev.peak_cell_pos.values]; ev["gu"] = gu[ev.peak_cell_pos.values]
    ev["holiday"] = pd.Series(tiv.ymd.values[ev.t0.values]).isin(HOLIDAY_SET).values; ev["daytype_start"] = dtype_[ev.t0.values]
    ev["day_flag_share"] = day_share.reindex(tiv.ymd.values[ev.t0.values]).values; ev["citywide_day"] = ev.day_flag_share > 0.08
    ev = ev[ev.n_cellhours >= args.min_cellhours].copy()
    ev["rank_score"] = ev.sum_abs_excess * np.sqrt(ev.peak_z.abs())
    ev = ev.sort_values("sum_abs_excess", ascending=False).reset_index().rename(columns={"comp": "event_id"})
    ev["dongs"] = ""
    topids = ev.event_id.head(args.top).values; sub = df[df.comp.isin(topids)]
    ev.loc[ev.event_id.isin(topids), "dongs"] = ev.loc[ev.event_id.isin(topids), "event_id"].map(sub.groupby("comp")["i"].apply(lambda s: ", ".join(pd.Series(dong[s.values]).value_counts().index[:3])))
    cols = ["event_id", "date", "start", "end_date", "end", "duration_h", "gu", "dong", "dongs", "direction", "peak_z", "mean_z", "n_cells", "n_cellhours", "sum_excess", "sum_abs_excess", "peak_X", "peak_mu", "peak_time", "peak_cell", "holiday", "daytype_start", "day_flag_share", "citywide_day", "rank_score"]
    ev[cols].to_csv(OUT / f"events_all_{ytag}.csv", index=False, encoding="utf-8-sig")
    top = ev[cols].head(args.top); top.to_csv(OUT / f"events_top{args.top}_{ytag}.csv", index=False, encoding="utf-8-sig")
    lines += ["## F6: event catalog", "", f"flagged cell-hours {int(flag.sum())} ({flag.mean():.3%}); connected events (>= {args.min_cellhours} cell-hours): {len(ev)}; "
              f"single cell-hour flags dropped: {int((labs >= 0).sum() - ev.n_cellhours.sum())}", "",
              f"| statistic | increase | decrease |", "|---|---|---|",
              f"| events | {(ev.direction == 'increase').sum()} | {(ev.direction == 'decrease').sum()} |",
              f"| median duration (h) | {ev[ev.direction == 'increase'].duration_h.median():.0f} | {ev[ev.direction == 'decrease'].duration_h.median():.0f} |",
              f"| median cells | {ev[ev.direction == 'increase'].n_cells.median():.0f} | {ev[ev.direction == 'decrease'].n_cells.median():.0f} |",
              f"| events with >= 10 cells | {((ev.direction == 'increase') & (ev.n_cells >= 10)).sum()} | {((ev.direction == 'decrease') & (ev.n_cells >= 10)).sum()} |",
              f"| events starting on holidays / weekends | {(ev.holiday | (ev.daytype_start > 0)).sum()} | |", "",
              f"### top {min(50, args.top)} events by total |excess| people (sum over cell-hours of X - mu)", "",
              "| # | date | time | gu / dong | direction | peak z | cells | cell-h | duration | excess (k people) | peak X / mu | holiday |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, e in enumerate(top.head(50).itertuples(), 1):
        lines.append(f"| {r} | {e.date} | {e.start}-{e.end}{'' if e.end_date == e.date else ' (+' + e.end_date[5:] + ')'} | {e.gu} {e.dong} | {e.direction} | {e.peak_z:.1f} | {e.n_cells} | {e.n_cellhours} | {e.duration_h} h | {e.sum_excess/1e3:.1f} | {e.peak_X:.0f} / {e.peak_mu:.0f} | {'Y' if e.holiday else ''} |")
    lines.append("")
    loc = ev[~ev.citywide_day][cols].head(args.top); loc.to_csv(OUT / f"events_top{args.top}_localized_{ytag}.csv", index=False, encoding="utf-8-sig")
    lines += [f"### top {min(50, args.top)} localized events (days where < 8% of cells are flagged; excludes holiday-like city-wide days)", "",
              "| # | date | time | gu / dong | direction | peak z | cells | cell-h | duration | excess (k people) | peak X / mu | day share |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, e in enumerate(loc.head(50).itertuples(), 1):
        lines.append(f"| {r} | {e.date} | {e.start}-{e.end} | {e.gu} {e.dong} | {e.direction} | {e.peak_z:.1f} | {e.n_cells} | {e.n_cellhours} | {e.duration_h} h | {e.sum_excess/1e3:.1f} | {e.peak_X:.0f} / {e.peak_mu:.0f} | {e.day_flag_share:.3f} |")
    lines += ["", "### city-wide days (>= 8% of cells flagged, daily mean)", "", "| day | share | holiday |", "|---|---|---|"]
    for d, v in day_share[day_share > 0.08].sort_values(ascending=False).items():
        lines.append(f"| {d} | {v:.3f} | {'Y' if d in HOLIDAY_SET else ''} |")
    lines.append("")
    # events by month and by gu
    ev["month"] = ev.date.str[:7]
    lines += ["### events per month (increase / decrease, >= 10 cells)", "", ev[ev.n_cells >= 10].groupby(["month", "direction"]).size().unstack(fill_value=0).T.to_markdown(), "",
              "### top-200 events by gu", "", top.gu.value_counts().head(12).to_frame("events").T.to_markdown(), ""]
    # ---- F5: GRU residual vs anomalies ----
    pfs = {"2024": ML_RES / f"pred_{args.pred}_val.npz", "2023": ML_RES / f"pred_{args.pred}_train.npz", "2025": ML_RES / f"pred_{args.pred}_test.npz"}
    Xh = np.full((T, N), np.nan, np.float32)
    for y in years:
        pf = pfs.get(y)
        if pf and pf.exists():
            zf = np.load(pf); pt = pd.to_datetime(pd.read_csv(str(pf).replace(".npz", "_time.csv")).datetime)
            pos = pd.Series(np.arange(len(pt)), index=pt).reindex(dtime).values; okp = np.isfinite(pos)
            Xh[okp] = zf["Xhat"][pos[okp].astype(int)]
    okh = ~np.isnan(Xh[:, 0]) & (Xh[:, 0] != 0)
    if okh.any():
        R = X[okh] - Xh[okh]; za = np.abs(z[okh]); Ra = np.abs(R); fl = flag[okh]; flr = flag_raw[okh]
        rng = np.random.default_rng(0); sub = rng.choice(R.size, min(5_000_000, R.size), replace=False)
        from scipy.stats import spearmanr
        corr_p = float(np.corrcoef(za.ravel()[sub], Ra.ravel()[sub])[0, 1]); corr_s = float(spearmanr(za.ravel()[sub], Ra.ravel()[sub]).correlation)
        topR = Ra >= np.quantile(Ra, .99); topz = za >= np.quantile(za, .99)
        pr = lambda a, b: (float((a & b).sum() / a.sum()), float((a & b).sum() / b.sum()), float((a & b).sum() / (a | b).sum()))
        p1, r1, j1 = pr(topR, fl); p2, r2, j2 = pr(topR, topz)
        lines += [f"## F5: GRU residual R = X - Xhat ({args.pred}, {int(okh.sum())} predicted hours) vs anomaly score", "",
                  f"- Corr(|R|, |z|): Pearson {corr_p:.3f}, Spearman {corr_s:.3f} (5M-entry subsample)",
                  f"- top-1% |R| vs event cell-hours ({fl.mean():.2%} of entries): precision {p1:.3f}, recall {r1:.3f}, Jaccard {j1:.3f}; vs all |z| > {args.tau:g} ({flr.mean():.2%}): precision {float((topR & flr).sum() / topR.sum()):.3f}",
                  f"- top-1% |R| vs top-1% |z|: precision {p2:.3f}, recall {r2:.3f}, Jaccard {j2:.3f} (random: 0.01 / 0.01 / 0.005)",
                  f"- share of ||R||^2 inside event entries: {float((R[fl]**2).sum() / (R**2).sum()):.3f}; inside top-1% |z|: {float((R[topz]**2).sum() / (R**2).sum()):.3f}",
                  f"- signed: Corr(R, X - mu) on event entries {float(np.corrcoef(R[fl], excess[okh][fl])[0, 1]):.3f}; mean R / mean excess on surge entries: {float(R[z[okh] > args.tau].mean()):.1f} / {float(excess[okh][z[okh] > args.tau].mean()):.1f} people", ""]
    # ---- figures ----
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    ax = axs[0, 0]; ax.hist(np.clip(z.ravel()[::37], -15, 15), bins=200, log=True, color="tab:blue"); ax.axvline(args.tau, color="r"); ax.axvline(-args.tau, color="r"); ax.set_title("distribution of z (clipped to ±15)")
    ax = axs[0, 1]; mc = pd.DataFrame({"pos": (z > args.tau).sum(1), "neg": (z < -args.tau).sum(1)}, index=dtime).resample("D").sum(); ax.plot(mc.index, mc.pos, lw=.7, label="surge cell-hours / day"); ax.plot(mc.index, -mc.neg, lw=.7, label="drop cell-hours / day (neg)"); ax.legend(); ax.grid(alpha=.3); ax.set_title(f"flagged cell-hours per day (|z| > {args.tau:g})")
    for j, (nm, m) in enumerate((("surge z > tau", z > args.tau), ("drop z < -tau", z < -args.tau))):
        ax = axs[1, j]; v = np.full(len(grid), np.nan, np.float32); v[cells] = m.sum(0); im = ax.imshow(to_lattice(np.log1p(v), grid), origin="lower", cmap="magma"); ax.axis("off"); ax.set_title(f"log(1 + flagged hours per cell): {nm}"); plt.colorbar(im, ax=ax, fraction=.04)
    fig.tight_layout(); fig.savefig(OUT / f"fig_z_overview_{ytag}.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 5)); t20 = top.head(60)
    ax.scatter(pd.to_datetime(t20.date), [int(s[:2]) for s in t20.start], s=np.sqrt(t20.sum_abs_excess) / 3, c=np.where(t20.direction == "increase", "tab:red", "tab:blue"), alpha=.6)
    for e in t20.head(25).itertuples():
        ax.annotate(f"{e.dong}", (pd.Timestamp(e.date), int(e.start[:2])), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_ylabel("start hour"); ax.set_title("top-60 events (size = total |excess|; red increase, blue decrease)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / f"fig_top_events_{ytag}.png", dpi=110); plt.close(fig)
    (OUT / f"z_stats_{ytag}.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
