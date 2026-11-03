"""Summarise the QC tables and produce sanity figures.

Outputs (processed/qc/):
  summary.md                       coverage, suppression regimes, missing days, totals by month
  fig_daily_totals.png             daily mean SPOP / pop10 / foreign totals over the whole period
  fig_groupsum_ratio.png           groupsum/SPOP and 0-9 share per day (shows the suppression regimes)
  fig_map_pop10_2023-06-14.png     night vs afternoon map for a representative weekday
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from loader import load_grid, load_grid_meta, load_local, to_lattice

REP_DAY = "20230614"


def main():
    # months can be rebuilt individually and appended, so keep the last record per day
    qc = pd.read_csv(C.OUT_QC / "local_daily_qc.csv", dtype={"ymd": str}).drop_duplicates("ymd", keep="last").sort_values("ymd")
    qc["date"] = pd.to_datetime(qc.ymd, format="%Y%m%d")
    qc["ym"] = qc.ymd.str[:6]
    fq_path = C.OUT_QC / "foreign_daily_qc.csv"
    fq = (pd.read_csv(fq_path, dtype={"ymd": str}).drop_duplicates(["kind", "ymd"], keep="last").sort_values(["kind", "ymd"])
          if fq_path.exists() else None)
    meta = load_grid_meta()

    # Anomalous-day rules (domestic). These days stay in the tensors but are listed for exclusion.
    ok = qc[qc.available]
    med_total = ok.spop_total_daily_mean.median()
    med_cells = ok.n_cells.median()
    rules = {
        "incomplete_hours": ok.n_hours < 24,
        "total_x2_or_more": ok.spop_total_daily_mean > 1.5 * med_total,
        "total_low": ok.spop_total_daily_mean < 0.85 * med_total,
        "cells_missing": ok.n_cells < 0.9 * med_cells,          # ~1,750 peripheral cells dropped and mass redistributed
    }
    reasons = pd.DataFrame(rules).apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    anomalous = ok.loc[reasons != "", ["ymd", "n_cells", "n_hours", "spop_total_daily_mean"]].assign(reason=reasons[reasons != ""])
    anomalous.to_csv(C.OUT_QC / "anomalous_days.csv", index=False, encoding="utf-8-sig")

    # Foreigners: level shifts between months are methodological (2024-06, 2025-03, 2025-09), so compare
    # each day with its own month's median instead of the global one.
    if fq is not None:
        fa0 = fq[fq.available].copy()
        fa0["ym"] = fa0.ymd.str[:6]
        mmed = fa0.groupby(["kind", "ym"]).total_daily_mean.transform("median")
        fr = pd.DataFrame({"total_x2_or_more": fa0.total_daily_mean > 1.5 * mmed,
                           "total_low": fa0.total_daily_mean < 0.5 * mmed})
        freason = fr.apply(lambda r: ";".join(k for k, v in r.items() if v), axis=1)
        fanom = fa0.loc[freason != "", ["kind", "ymd", "total_daily_mean"]].assign(reason=freason[freason != ""])
        fanom.to_csv(C.OUT_QC / "anomalous_days_foreign.csv", index=False, encoding="utf-8-sig")
    else:
        fanom = None

    lines = ["# Living-population pipeline QC summary", ""]
    lines += ["## Standard grid", "",
              f"- cells: {meta['n_cells']}, lattice {meta['lattice_shape_rows_cols']} (rows x cols), EPSG:{meta['crs_epsg']}",
              f"- rook edges {meta['rook_edges']}, queen edges {meta['queen_edges']}",
              f"- cells with dong mapping: {meta['cells_with_dong_mapping']}, split across several dongs: {meta['cells_split_across_multiple_dongs']}",
              f"- populated CELL_IDs absent from the shapefile (dropped): {meta['cells_with_population_but_not_in_grid']}", ""]

    avail = qc[qc.available]
    missing = qc[~qc.available].ymd.tolist()
    lines += ["## Domestic (내국인) coverage", "",
              f"- months: {qc.ym.min()} .. {qc.ym.max()}, days expected {len(qc)}, present {len(avail)}",
              f"- missing days: {missing if missing else 'none'}",
              f"- malformed rows skipped: {int(avail.bad_rows.sum())} (days affected: {avail.loc[avail.bad_rows > 0, 'ymd'].tolist()})",
              f"- cells per day: {int(avail.n_cells.min())} .. {int(avail.n_cells.max())}",
              f"- days with fewer than 24 hours: {avail.loc[avail.n_hours < 24, 'ymd'].tolist() or 'none'}",
              f"- source fallback to SE_ copy: {avail.loc[avail.source.astype(str).str.startswith('SE_'), 'ymd'].tolist() or 'none'}", "",
              "### Anomalous days (kept in tensors, listed in qc/anomalous_days.csv; loader can drop them)", "",
              "| ymd | cells | hours | daily mean SPOP | reason |", "|---|---|---|---|---|"]
    for _, r in anomalous.iterrows():
        lines.append(f"| {r.ymd} | {int(r.n_cells)} | {int(r.n_hours)} | {r.spop_total_daily_mean:,.0f} | {r.reason} |")
    lines.append("")

    lines += ["## Suppression regimes (domestic)", "", "| regime | months | days | '*' entries/day | groupsum/SPOP | 0-9 share |", "|---|---|---|---|---|---|"]
    for r, g in avail.groupby("regime"):
        lines.append(f"| {r} | {g.ym.min()}..{g.ym.max()} | {len(g)} | {g.n_star_entries.mean():,.0f} | {g.groupsum_over_spop.mean():.4f} | {g.share_0_9_daily_mean.mean():.4f} |")
    lines += ["", "A: full precision. B: values<=3 shown as '*' (see masked_ files). C: values<=3 shown as 0 (undetectable).", ""]

    lines += ["## Monthly totals (domestic, daily mean over the month)", "", "| month | days | SPOP | pop10 (ages 10+) | 0-9 share | night 03h | afternoon 14h |", "|---|---|---|---|---|---|---|"]
    for ym, g in avail.groupby("ym"):
        lines.append(f"| {ym} | {len(g)} | {g.spop_total_daily_mean.mean():,.0f} | {g.pop10_total_daily_mean.mean():,.0f} | {g.share_0_9_daily_mean.mean():.4f} | {g.spop_total_tt03.mean():,.0f} | {g.spop_total_tt14.mean():,.0f} |")
    lines.append("")

    if fq is not None:
        fq["ym"] = fq.ymd.str[:6]
        fa = fq[fq.available]
        lines += ["## Foreigners coverage", ""]
        for kind, g in fa.groupby("kind"):
            miss = fq[(fq.kind == kind) & (~fq.available)].ymd.tolist()
            lines += [f"- {kind}: {g.ym.min()}..{g.ym.max()}, days {len(g)}, missing {miss or 'none'}, formats {sorted(g.format.dropna().unique().tolist())}, "
                      f"malformed rows {int(g.bad_rows.sum())}, mean daily total {g.total_daily_mean.mean():,.0f}, "
                      f"F2 '*' totals/day {g.loc[g.format == 'F2', 'n_star_total'].mean():,.0f}, natsum/total (F2) {g.loc[g.format == 'F2', 'natsum_over_total'].mean():.3f}"]
        if fanom is not None and len(fanom):
            lines += ["", "Anomalous foreigner days (vs own month median; qc/anomalous_days_foreign.csv): " +
                      ", ".join(f"{r.kind} {r.ymd} ({r.reason})" for r in fanom.itertuples())]
        lines += ["", "Level shifts in long-term totals at 2024-06 (+15%), 2025-03 (-30%) and 2025-09 (+20%) coincide with "
                  "source format/method changes and should not be read as real population change.",
                  "", "| month | long total | long CHN share | temp total | temp CHN share |", "|---|---|---|---|---|"]
        for ym, g in fa.groupby("ym"):
            l, t = g[g.kind == "long"], g[g.kind == "temp"]
            lines.append(f"| {ym} | {l.total_daily_mean.mean():,.0f} | {l.chn_share.mean():.3f} | {t.total_daily_mean.mean():,.0f} | {t.chn_share.mean():.3f} |")
        lines.append("")

    (C.OUT_QC / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Figures
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(avail.date, avail.spop_total_daily_mean / 1e6, lw=0.8, label="SPOP (all ages)")
    ax.plot(avail.date, avail.pop10_total_daily_mean / 1e6, lw=0.8, label="pop10 (ages 10+)")
    if fq is not None:
        for kind, lab in (("long", "long-term foreigners"), ("temp", "short-term foreigners")):
            g = fa[fa.kind == kind].copy(); g["date"] = pd.to_datetime(g.ymd, format="%Y%m%d")
            ax.plot(g.date, g.total_daily_mean / 1e6, lw=0.8, label=lab)
    for x, lab in (("2024-07-01", "B: '*' masking"), ("2025-01-01", "C: 0 masking")):
        ax.axvline(pd.Timestamp(x), color="k", ls="--", lw=0.6); ax.text(pd.Timestamp(x), ax.get_ylim()[1] * 0.97, lab, fontsize=7, va="top")
    ax.set_ylabel("daily mean population (millions)"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.OUT_QC / "fig_daily_totals.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axs[0].plot(avail.date, avail.groupsum_over_spop, lw=0.8); axs[0].set_ylabel("groupsum / SPOP"); axs[0].grid(alpha=0.3)
    axs[1].plot(avail.date, avail.share_0_9_daily_mean, lw=0.8); axs[1].set_ylabel("0-9 share"); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.OUT_QC / "fig_groupsum_ratio.png", dpi=130); plt.close(fig)

    ym = REP_DAY[:6]
    X, ti = load_local([ym], cols=["SPOP", "M00", "F00"])
    pop10 = X[:, :, 0] - X[:, :, 1] - X[:, :, 2]
    grid = load_grid(with_geometry=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    sel = {f"{REP_DAY} 03h": 3, f"{REP_DAY} 14h": 14}
    imgs = {}
    for ax, (lab, tt) in zip(axs[:2], sel.items()):
        t = ti.index[(ti.ymd == REP_DAY) & (ti.tt == tt)][0]
        imgs[lab] = to_lattice(pop10[t], grid)
        im = ax.imshow(np.log10(imgs[lab] + 1), origin="lower", cmap="magma"); ax.set_title(f"log10(pop10+1) {lab}"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.04)
    night, day = imgs[f"{REP_DAY} 03h"], imgs[f"{REP_DAY} 14h"]
    ratio = np.log2((day + 1) / (night + 1))
    im = axs[2].imshow(ratio, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4); axs[2].set_title("log2(day/night)"); axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.04)
    fig.tight_layout(); fig.savefig(C.OUT_QC / f"fig_map_pop10_{REP_DAY[:4]}-{REP_DAY[4:6]}-{REP_DAY[6:]}.png", dpi=130); plt.close(fig)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
