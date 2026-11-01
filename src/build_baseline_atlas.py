"""Baseline vulnerability atlas (proposal sections 6-7, population-only part).

Per 250m cell, from observations alone (no model, no facilities yet):
  1. Normal pattern  m_i(daytype, hour) and scale s_i(daytype, hour)   -- median / MAD over regime A (2023-01..2024-06)
  2. Anomaly         Z_i(t) = (X_i(t) - m) / s  over the whole period; exceedance rates, extreme quantiles, top events
  3. Exposure        night (03h) / afternoon (14h) / max-hour population, day-night ratio, peak hour, hourly CV
  4. Composition     elderly 65+, 70+, youth 10-19, female shares at night and day (regime A only, clean demographics)
                     long-/short-term foreigner shares (F1 format months, no masking)
  5. Seismic proxy   Vs30 site class and an indicative amplification factor; exposure = population x factor

Outputs (processed/atlas/):
  cell_profiles_pop10.npz        median[N,3,24], scale[N,3,24], daytypes
  atlas_cells.parquet / .gpkg    one row per cell with all indicators
  z_top_events.csv               cell-hours with the largest positive Z (event candidates for the disaster DB)
  city_daily_excess.csv          city-wide positive excess sum per day (event days)
  city_hourly_profile.csv        city totals by daytype x hour
  fig_atlas_*.png, atlas_summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from loader import (load_pop10, load_local, load_foreign, load_grid, months_between, to_lattice, anomalous_days)

OUT = C.OUT / "atlas"
REGIME_A = months_between("202301", "202406")
ALL_MONTHS = months_between("202301", "202601")
DAYTYPES = ["weekday", "saturday", "sunday_holiday"]
NIGHT_TT, DAY_TT = 3, 14
Z_THRESHOLDS = (2.0, 3.0, 5.0)

HOLIDAYS = {  # Korean public holidays incl. substitutes and one-off holidays
    2023: ["0101", "0121", "0122", "0123", "0124", "0301", "0505", "0527", "0529", "0606", "0815",
           "0928", "0929", "0930", "1002", "1003", "1009", "1225"],
    2024: ["0101", "0209", "0210", "0211", "0212", "0301", "0410", "0506", "0515", "0606", "0815",
           "0916", "0917", "0918", "1001", "1003", "1009", "1225"],
    2025: ["0101", "0127", "0128", "0129", "0130", "0301", "0303", "0505", "0506", "0603", "0606", "0815",
           "1003", "1005", "1006", "1007", "1008", "1009", "1225"],
    2026: ["0101", "0216", "0217", "0218", "0301", "0302"],
}
HOLIDAY_SET = {f"{y}{md}" for y, l in HOLIDAYS.items() for md in l}


def daytype_of(ti: pd.DataFrame) -> np.ndarray:
    dow = ti.datetime.dt.dayofweek.values
    dt = np.where(dow == 5, 1, 0)
    dt = np.where((dow == 6) | ti.ymd.isin(HOLIDAY_SET).values, 2, dt)
    return dt


def site_class(vs30):
    """NEHRP-style site class from Vs30 (m/s) and an indicative short-period amplification factor."""
    cls = np.full(len(vs30), "NA", dtype=object)
    fac = np.full(len(vs30), np.nan)
    bins = [(760, np.inf, "B", 1.0), (360, 760, "C", 1.2), (180, 360, "D", 1.6), (0, 180, "E", 2.5)]
    for lo, hi, c, f in bins:
        m = (vs30 >= lo) & (vs30 < hi)
        cls[m], fac[m] = c, f
    return cls, fac


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=True)
    N = len(grid)
    anom = anomalous_days()

    # ---------- 1. normal pattern from regime A ----------
    XA, tiA = load_pop10(REGIME_A, drop_missing_days=True, drop_anomalous=True)
    dtA = daytype_of(tiA)
    med = np.zeros((N, 3, 24), np.float32)
    scale = np.zeros((N, 3, 24), np.float32)
    ndays_dt = np.zeros((3, 24), int)
    for d in range(3):
        for h in range(24):
            sel = (dtA == d) & (tiA.tt.values == h)
            block = XA[sel]                      # [days, N]
            ndays_dt[d, h] = len(block)
            m = np.median(block, axis=0)
            mad = np.median(np.abs(block - m), axis=0) * 1.4826
            med[:, d, h] = m
            scale[:, d, h] = np.maximum(mad, np.sqrt(np.maximum(m, 0)) + 2.0)   # floor: Poisson-like noise + 2 persons
    np.savez_compressed(OUT / "cell_profiles_pop10.npz", median=med, scale=scale, daytypes=np.array(DAYTYPES),
                        n_days=ndays_dt, source_months=np.array(REGIME_A))
    del XA

    # ---------- 2. anomaly Z over the whole period ----------
    X, ti = load_pop10(ALL_MONTHS, drop_missing_days=True, drop_anomalous=True)
    dt = daytype_of(ti)
    tt = ti.tt.values
    T = len(ti)
    exceed = {z: np.zeros(N, np.int32) for z in Z_THRESHOLDS}
    zmax = np.full(N, -np.inf, np.float32)
    zmax_t = np.zeros(N, np.int64)
    excess_city = np.zeros(T, np.float64)          # sum over cells of max(X - m, 0)      (raw)
    excess_adj = np.zeros(T, np.float64)           # same after scaling each hour's city total to the baseline total
    level = np.zeros(T, np.float64)                # city total / baseline total per hour (drift of the whole series)
    top_rows = []
    chunk = 24 * 31
    for s0 in range(0, T, chunk):
        s1 = min(T, s0 + chunk)
        m = med[:, dt[s0:s1], tt[s0:s1]].T          # [t, N]
        sc = scale[:, dt[s0:s1], tt[s0:s1]].T
        Xc = X[s0:s1]
        # level adjustment: the city total drifts by a few % (method changes, 2025 level shift); Z is meant to
        # measure spatial redistribution, so each hour is rescaled to the baseline city total first
        r = m.sum(axis=1) / np.maximum(Xc.sum(axis=1), 1.0)
        level[s0:s1] = 1.0 / r
        Xa = Xc * r[:, None]
        Z = (Xa - m) / sc
        for z in Z_THRESHOLDS:
            exceed[z] += (Z > z).sum(axis=0)
        cm = Z.max(axis=0); upd = cm > zmax
        zmax[upd] = cm[upd]; zmax_t[upd] = s0 + Z.argmax(axis=0)[upd]
        excess_city[s0:s1] = np.maximum(Xc - m, 0).sum(axis=1)
        excess_adj[s0:s1] = np.maximum(Xa - m, 0).sum(axis=1)
        # candidate top events in this chunk: population >= 200 and a non-trivial baseline (cells that were
        # empty in regime A are new developments or grid changes, not events)
        Zc = np.where(m >= 20, Z, -np.inf)
        flat = np.argsort(Zc, axis=None)[-300:]
        tI, cI = np.unravel_index(flat, Z.shape)
        for a, b in zip(tI, cI):
            if Xc[a, b] >= 200 and np.isfinite(Zc[a, b]):
                top_rows.append((s0 + a, b, float(Z[a, b]), float(Xc[a, b]), float(m[a, b])))
    top = pd.DataFrame(top_rows, columns=["t", "cell_idx", "Z", "pop10", "baseline"]).sort_values("Z", ascending=False)
    top = top.drop_duplicates("cell_idx").head(300)
    top["datetime"] = ti.datetime.values[top.t.values]
    top["ymd"] = ti.ymd.values[top.t.values]
    top["dong"] = grid.dong_major_name.values[top.cell_idx.values]
    top["CELL_ID"] = grid.CELL_ID.values[top.cell_idx.values]
    top[["datetime", "ymd", "dong", "CELL_ID", "cell_idx", "pop10", "baseline", "Z"]].to_csv(
        OUT / "z_top_events.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

    daily = pd.DataFrame({"ymd": ti.ymd.values, "excess_raw": excess_city, "excess_adj": excess_adj, "level": level}).groupby("ymd").agg(
        excess_raw=("excess_raw", "sum"), excess_adj=("excess_adj", "sum"), level=("level", "mean"))
    dts = ti.groupby("ymd").datetime.first()
    daily["daytype"] = [DAYTYPES[d] for d in daytype_of(pd.DataFrame({"datetime": dts.values, "ymd": dts.index}))]
    daily.sort_values("excess_adj", ascending=False).to_csv(OUT / "city_daily_excess.csv", encoding="utf-8-sig", float_format="%.3f")

    # ---------- 3. exposure indicators (whole period, anomalies dropped) ----------
    night = X[tt == NIGHT_TT].mean(axis=0)
    day = X[tt == DAY_TT].mean(axis=0)
    hourly_mean = np.stack([X[tt == h].mean(axis=0) for h in range(24)], axis=1)   # [N, 24]
    wk = dt == 0
    hourly_wk = np.stack([X[wk & (tt == h)].mean(axis=0) for h in range(24)], axis=1)
    max_hour_pop = hourly_wk.max(axis=1)
    peak_hour = hourly_wk.argmax(axis=1)
    cv_hourly = hourly_wk.std(axis=1) / np.maximum(hourly_wk.mean(axis=1), 1e-6)
    q99 = np.quantile(X, 0.99, axis=0)
    q999 = np.quantile(X, 0.999, axis=0)
    city = pd.DataFrame({DAYTYPES[d]: [X[(dt == d) & (tt == h)].sum(axis=1).mean() for h in range(24)] for d in range(3)})
    city.index.name = "hour"
    city.to_csv(OUT / "city_hourly_profile.csv", float_format="%.0f")
    del X

    # ---------- 4. composition from regime A (clean demographics), weekdays only ----------
    cols = C.LOCAL_VALUE_COLS
    ci = {c: cols.index(c) for c in cols}
    sums = {k: np.zeros((N, len(cols)), np.float64) for k in ("night", "day")}
    cnt = {k: 0 for k in sums}
    for ym in REGIME_A:
        L, tl = load_local([ym], drop_missing_days=True)
        d_l = daytype_of(tl); t_l = tl.tt.values
        for k, h in (("night", NIGHT_TT), ("day", DAY_TT)):
            sel = (d_l == 0) & (t_l == h)
            sums[k] += L[sel].sum(axis=0); cnt[k] += int(sel.sum())
    comp = {}
    for k in sums:
        mean = sums[k] / max(cnt[k], 1)
        tot10 = mean[:, ci["SPOP"]] - mean[:, ci["M00"]] - mean[:, ci["F00"]]
        den = np.maximum(tot10, 1e-6)
        g = lambda names: sum(mean[:, ci[n]] for n in names)
        comp[f"share_65plus_{k}"] = g(["M65", "M70", "F65", "F70"]) / den
        comp[f"share_70plus_{k}"] = g(["M70", "F70"]) / den
        comp[f"share_10_19_{k}"] = g(["M10", "M15", "F10", "F15"]) / den
        comp[f"share_female_{k}"] = g([f"F{a}" for a in C.AGE_BANDS if a != "00"]) / den
        comp[f"pop10_{k}_regimeA"] = tot10

    f1_months = REGIME_A
    fl, tfl = load_foreign("long", f1_months, drop_missing_days=True)
    ft, tft = load_foreign("temp", f1_months, drop_missing_days=True)
    for name, F, tf in (("long", fl, tfl), ("temp", ft, tft)):
        dd = daytype_of(tf); hh = tf.tt.values
        comp[f"foreign_{name}_night"] = F[(dd == 0) & (hh == NIGHT_TT)].mean(axis=0)
        comp[f"foreign_{name}_day"] = F[(dd == 0) & (hh == DAY_TT)].mean(axis=0)
    del fl, ft
    for k in ("night", "day"):
        comp[f"share_foreign_{k}"] = (comp[f"foreign_long_{k}"] + comp[f"foreign_temp_{k}"]) / np.maximum(
            comp[f"pop10_{k}_regimeA"] + comp[f"foreign_long_{k}"] + comp[f"foreign_temp_{k}"], 1e-6)

    # ---------- 5. seismic proxy ----------
    vs30 = grid.Vs30.values.astype(float)
    cls, fac = site_class(np.nan_to_num(vs30, nan=-1))
    cls[np.isnan(vs30)] = "NA"; fac[np.isnan(vs30)] = np.nan

    # ---------- assemble ----------
    atlas = grid[["cell_idx", "CELL_ID", "row", "col", "CELL_X", "CELL_Y", "dong_major", "dong_major_name", "Vs30", "geometry"]].copy()
    atlas["pop10_night"] = night
    atlas["pop10_day"] = day
    atlas["pop10_maxhour_weekday"] = max_hour_pop
    atlas["peak_hour_weekday"] = peak_hour
    atlas["day_night_ratio"] = (day + 1) / (night + 1)
    atlas["cv_hourly_weekday"] = cv_hourly
    atlas["pop10_q99"] = q99
    atlas["pop10_q999"] = q999
    for z in Z_THRESHOLDS:
        atlas[f"exceed_rate_Z{z:g}"] = exceed[z] / T
    atlas["Z_max"] = zmax
    atlas["Z_max_datetime"] = ti.datetime.values[zmax_t]
    for k, v in comp.items():
        atlas[k] = v
    atlas["site_class"] = cls
    atlas["amp_factor_indicative"] = fac
    atlas["seismic_exposure_night"] = night * fac
    atlas["seismic_exposure_day"] = day * fac
    atlas["seismic_exposure_65plus_night"] = night * fac * comp["share_65plus_night"]
    populated = (night + day) > 0
    atlas["populated"] = populated
    atlas.drop(columns="geometry", errors="ignore").to_parquet(OUT / "atlas_cells.parquet", index=False)
    atlas.reset_index(drop=True).to_file(OUT / "atlas_cells.gpkg", layer="atlas", driver="GPKG", index=False)

    # ---------- figures ----------
    g0 = grid
    def panel(ax, vals, title, cmap="magma", log=False, vmin=None, vmax=None, mask_unpop=True):
        v = np.array(vals, dtype=float)
        if mask_unpop:
            v = np.where(populated, v, np.nan)
        if log:
            v = np.log10(v + 1)
        im = ax.imshow(to_lattice(v, g0), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.035)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panel(axs[0, 0], night, "log10 pop10 night 03h (mean)", log=True)
    panel(axs[0, 1], day, "log10 pop10 afternoon 14h (mean)", log=True)
    panel(axs[0, 2], np.log2((day + 1) / (night + 1)), "log2 day/night ratio", cmap="RdBu_r", vmin=-4, vmax=4)
    panel(axs[1, 0], peak_hour, "weekday peak hour", cmap="twilight", vmin=0, vmax=23)
    panel(axs[1, 1], cv_hourly, "hourly CV (weekday)", cmap="viridis", vmin=0, vmax=1.5)
    panel(axs[1, 2], q999, "log10 pop10 99.9% quantile", log=True)
    fig.suptitle("Baseline exposure atlas (pop10 = ages 10+, 2023-01..2026-01 excl. anomalous days)")
    fig.tight_layout(); fig.savefig(OUT / "fig_atlas_exposure.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panel(axs[0, 0], comp["share_65plus_night"], "share 65+ night (regime A)", cmap="YlOrRd", vmin=0, vmax=0.4)
    panel(axs[0, 1], comp["share_65plus_day"], "share 65+ afternoon (regime A)", cmap="YlOrRd", vmin=0, vmax=0.4)
    panel(axs[0, 2], comp["share_10_19_day"], "share 10-19 afternoon (regime A)", cmap="YlGnBu", vmin=0, vmax=0.4)
    panel(axs[1, 0], comp["share_foreign_night"], "foreigner share night", cmap="PuRd", vmin=0, vmax=0.3)
    panel(axs[1, 1], comp["share_foreign_day"], "foreigner share afternoon", cmap="PuRd", vmin=0, vmax=0.3)
    panel(axs[1, 2], comp["share_female_night"], "female share night", cmap="RdBu", vmin=0.35, vmax=0.65)
    fig.suptitle("Composition atlas (weekday means, 2023-01..2024-06)")
    fig.tight_layout(); fig.savefig(OUT / "fig_atlas_composition.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panel(axs[0, 0], atlas["exceed_rate_Z3"], "exceedance rate Z>3", cmap="inferno", vmin=0, vmax=0.05)
    panel(axs[0, 1], np.minimum(zmax, 60), "max Z observed (capped 60)", cmap="inferno")
    panel(axs[0, 2], scale[:, 0, DAY_TT] / np.maximum(med[:, 0, DAY_TT], 1), "relative scale s/m, weekday 14h", cmap="viridis", vmin=0, vmax=0.5)
    panel(axs[1, 0], vs30, "Vs30 (m/s)", cmap="terrain", mask_unpop=False)
    panel(axs[1, 1], atlas["seismic_exposure_day"], "log10 seismic exposure (14h x amp)", log=True)
    panel(axs[1, 2], atlas["seismic_exposure_65plus_night"], "log10 seismic exposure 65+ (03h x amp)", log=True)
    fig.suptitle("Anomaly and seismic-proxy atlas")
    fig.tight_layout(); fig.savefig(OUT / "fig_atlas_anomaly_seismic.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for d in DAYTYPES:
        ax.plot(city.index, city[d] / 1e6, marker="o", ms=3, label=d)
    ax.set_xlabel("hour"); ax.set_ylabel("city pop10 (millions)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_city_hourly_profile.png", dpi=120); plt.close(fig)

    dd = daily.reset_index(); dd["date"] = pd.to_datetime(dd.ymd, format="%Y%m%d")
    fig, axs = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axs[0].plot(dd.date, dd.level, lw=0.7); axs[0].set_ylabel("city total / baseline total"); axs[0].grid(alpha=0.3)
    axs[1].plot(dd.date, dd.excess_adj / 1e6, lw=0.7)
    for r in dd.sort_values("excess_adj", ascending=False).head(10).itertuples():
        axs[1].annotate(r.ymd, (r.date, r.excess_adj / 1e6), fontsize=7, rotation=60)
    axs[1].set_ylabel("level-adjusted positive excess\n(M person-hours/day)"); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_city_daily_excess.png", dpi=120); plt.close(fig)

    # ---------- summary ----------
    p = atlas[populated]
    lines = ["# Baseline vulnerability atlas — summary", "",
             f"- normal pattern: median/MAD per cell x daytype x hour from {REGIME_A[0]}..{REGIME_A[-1]} (days per slot: weekday {ndays_dt[0,0]}, saturday {ndays_dt[1,0]}, sunday/holiday {ndays_dt[2,0]})",
             f"- Z computed over {ti.ymd.nunique()} days ({ALL_MONTHS[0]}..{ALL_MONTHS[-1]}), anomalous days dropped: {sorted(anom)}",
             f"- populated cells: {int(populated.sum())} of {N}", "",
             "## City-wide", "",
             f"- weekday pop10 night 03h {city.weekday[3]/1e6:.2f}M, afternoon 14h {city.weekday[14]/1e6:.2f}M (ratio {city.weekday[14]/city.weekday[3]:.3f})",
             f"- sunday/holiday 14h {city.sunday_holiday[14]/1e6:.2f}M", "",
             "## Cell-level distribution (populated cells)", "", "| indicator | p10 | median | p90 | max |", "|---|---|---|---|---|"]
    for c in ["pop10_night", "pop10_day", "day_night_ratio", "cv_hourly_weekday", "share_65plus_night", "share_65plus_day",
              "share_10_19_day", "share_foreign_night", "share_foreign_day", "exceed_rate_Z3", "Z_max", "seismic_exposure_day"]:
        q = p[c].quantile([0.1, 0.5, 0.9]).values
        lines.append(f"| {c} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {p[c].max():.3f} |")
    lines += ["", "## Site class (Vs30)", "", p.site_class.value_counts().to_frame("cells").to_markdown(), "",
              "## Top event days (city-wide positive excess after removing the day's total-level drift)", "",
              daily.sort_values("excess_adj", ascending=False).head(15).to_markdown(floatfmt=".3g"), "",
              "## Top cell-hour anomalies (Z, pop10 >= 200, regime-A baseline >= 20)", "",
              top[["datetime", "dong", "CELL_ID", "pop10", "baseline", "Z"]].head(25).to_markdown(index=False, floatfmt=".1f"), ""]
    (OUT / "atlas_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
