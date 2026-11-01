"""Task 1.4-1.9: composition, temporal profiles, low-rank (SVD) structure and year-to-year stability.

Total population = pop10 (ages 10+), years 2023-2025 (anomalous / missing hours excluded), cells populated in 2023.
Outputs (processed/eda/):
  profiles_<year|all>.npz      hourly[N,24], weekday[N,24], weekend[N,24], weekly[N,168], monthly[N,12,24],
                               seasonal[N,4,24] (+ *_norm = divided by the profile sum), cells, year
  composition_2023.npz         daytype x hour x cell x (14 age x 2 sex) shares (regime A only)
  svd_<year|all>.npz           singular values, energy curves, top-30 temporal (U) and spatial (V) modes, raw and cell-centred
  svd_year_stability.csv       principal angles / subspace overlaps between years
  fig_*.png, task1_summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import config as C                                                      # noqa: E402
from loader import load_pop10, load_grid, months_between, anomalous_days, to_lattice   # noqa: E402
from canonical import load_canonical, calendar_features, EDA, AGE_LABELS, SEX_LABELS   # noqa: E402

YEARS = [2023, 2024, 2025]
SEASONS = ["spring", "summer", "autumn", "winter"]


def load_year(y):
    X, ti = load_pop10(months_between(f"{y}01", f"{y}12"))
    X = X.astype(np.float32).copy()
    bad = (~ti.available.values) | ti.ymd.isin(anomalous_days()).values
    X[bad] = np.nan
    return X, ti, calendar_features(ti)


def profiles(X, cal):
    """X [T, N] with NaN rows. Returns dict of profile arrays (nan-aware means)."""
    ok = ~np.isnan(X[:, 0])
    Xo, c = X[ok], cal[ok].reset_index(drop=True)
    def mean_by(mask):
        return np.nanmean(Xo[mask], axis=0) if mask.any() else np.full(X.shape[1], np.nan, np.float32)
    P = {}
    P["hourly"] = np.stack([mean_by((c.hour == h).values) for h in range(24)], 1)
    wd = (c.daytype == 0).values; we = (c.daytype != 0).values
    P["weekday"] = np.stack([mean_by(wd & (c.hour == h).values) for h in range(24)], 1)
    P["weekend"] = np.stack([mean_by(we & (c.hour == h).values) for h in range(24)], 1)
    nh = (~c.is_holiday).values
    P["weekly"] = np.stack([mean_by(nh & (c.dow == d).values & (c.hour == h).values) for d in range(7) for h in range(24)], 1)
    P["monthly"] = np.stack([np.stack([mean_by((c.month == m).values & (c.hour == h).values) for h in range(24)], 1) for m in range(1, 13)], 1)
    P["seasonal"] = np.stack([np.stack([mean_by((c.season == s).values & (c.hour == h).values) for h in range(24)], 1) for s in SEASONS], 1)
    for k in list(P):
        v = P[k]
        tot = np.nansum(v, axis=-1, keepdims=True) if v.ndim == 2 else np.nansum(v, axis=(-2, -1), keepdims=True)
        P[k + "_norm"] = v / np.maximum(tot, 1e-6)
    return P


def svd_analysis(M, name, grid, cells, ti_ok, r_keep=30):
    """M [T, N] no NaN. Raw and per-cell-centred SVD on GPU; returns dict and saves figures."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for variant in ("raw", "centred"):
        A = M - M.mean(axis=0, keepdims=True) if variant == "centred" else M
        At = torch.tensor(A, device=dev)
        U, S, Vh = torch.linalg.svd(At, full_matrices=False)
        S = S.cpu().numpy(); U = U[:, :r_keep].cpu().numpy(); V = Vh[:r_keep].T.cpu().numpy()
        del At
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        out[variant] = {"S": S, "energy": energy, "U": U, "V": V}
        # sign convention: make the spatial mode mostly positive
        for k in range(r_keep):
            if V[:, k].sum() < 0:
                V[:, k] *= -1; U[:, k] *= -1
    np.savez_compressed(EDA / f"svd_{name}.npz", cells=cells,
                        **{f"{v}_{k}": out[v][k] for v in out for k in ("S", "energy", "U", "V")})
    # figures: decay + temporal modes + spatial modes (centred variant, the informative one)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for v in out:
        axs[0].semilogy(np.arange(1, 61), out[v]["S"][:60], marker=".", label=v)
        axs[1].plot(np.arange(1, 61), out[v]["energy"][:60], marker=".", label=v)
    axs[0].set_title(f"singular values ({name})"); axs[1].set_title("cumulative energy"); axs[1].set_ylim(0, 1.01)
    for a in axs: a.grid(alpha=.3); a.legend(); a.set_xlabel("rank")
    fig.tight_layout(); fig.savefig(EDA / f"fig_svd_decay_{name}.png", dpi=120); plt.close(fig)

    U, V = out["centred"]["U"], out["centred"]["V"]
    fig, axs = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    t = pd.to_datetime(ti_ok.datetime.values)
    for k in range(6):
        axs[k].plot(t, U[:, k], lw=0.5); axs[k].set_ylabel(f"u{k+1}"); axs[k].grid(alpha=.3)
    axs[0].set_title(f"temporal modes (centred, {name})"); fig.tight_layout(); fig.savefig(EDA / f"fig_svd_temporal_{name}.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    two_weeks = slice(0, 24 * 14)
    for k in range(4):
        axs[0].plot(t[two_weeks], U[two_weeks, k], lw=0.9, label=f"u{k+1}")
    axs[0].legend(ncol=4); axs[0].set_title(f"first two weeks of temporal modes (centred, {name})"); axs[0].grid(alpha=.3)
    axs[1].plot(t[two_weeks], M[two_weeks].sum(1) / 1e6, color="k", lw=0.9); axs[1].set_ylabel("city pop10 (M)"); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(EDA / f"fig_svd_temporal_zoom_{name}.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for k, ax in enumerate(axs.ravel()):
        vec = np.full(len(grid), np.nan, np.float32); vec[cells] = V[:, k]
        lim = np.nanquantile(np.abs(vec), 0.99)
        im = ax.imshow(to_lattice(vec, grid), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"spatial mode v{k+1} (σ²-share {(out['centred']['S'][k]**2/np.sum(out['centred']['S']**2)):.3f})"); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=.035)
    fig.suptitle(f"spatial modes (centred, {name})"); fig.tight_layout(); fig.savefig(EDA / f"fig_svd_spatial_{name}.png", dpi=110); plt.close(fig)
    return out


def principal_angles(V1, V2, r):
    s = np.linalg.svd(V1[:, :r].T @ V2[:, :r], compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1, 1)))
    return ang, float(np.mean(s ** 2))          # mean squared cosine = subspace overlap in [0,1]


def main():
    EDA.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=False)
    data = {y: load_year(y) for y in YEARS}
    X23 = data[2023][0]
    cells = np.where(np.nanmax(X23, axis=0) > 0)[0]
    lines = ["# Task 1 summary (profiles, composition, SVD)", "", f"- cells populated in 2023: {len(cells)}; years {YEARS}; total = pop10 (ages 10+)", ""]

    # ---------- profiles ----------
    P_all = {}
    for y in YEARS:
        X, ti, cal = data[y]
        P = profiles(X[:, cells], cal)
        np.savez_compressed(EDA / f"profiles_{y}.npz", cells=cells, year=y, **P)
        P_all[y] = P
    pooled = {k: np.nanmean(np.stack([P_all[y][k] for y in YEARS]), axis=0) for k in P_all[2023]}
    np.savez_compressed(EDA / "profiles_all.npz", cells=cells, **pooled)
    # city-level figures
    city = {y: {k: np.nansum(P_all[y][k], axis=0) for k in ("hourly", "weekday", "weekend", "weekly", "monthly", "seasonal")} for y in YEARS}
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for y in YEARS:
        axs[0, 0].plot(city[y]["weekday"] / 1e6, marker=".", label=f"{y} weekday"); axs[0, 0].plot(city[y]["weekend"] / 1e6, ls="--", label=f"{y} weekend")
        axs[0, 1].plot(city[y]["weekly"] / 1e6, lw=1, label=str(y))
        axs[1, 0].plot(range(1, 13), city[y]["monthly"].mean(axis=1) / 1e6, marker="o", label=str(y))
        axs[1, 1].plot(city[y]["seasonal"].T / 1e6)
    axs[0, 0].set_title("city pop10 by hour"); axs[0, 1].set_title("city weekly profile (Mon 0h .. Sun 23h)")
    axs[1, 0].set_title("monthly mean"); axs[1, 1].set_title("seasonal hourly profiles (spring, summer, autumn, winter)")
    for a in axs.ravel(): a.grid(alpha=.3)
    axs[0, 0].legend(fontsize=7); axs[0, 1].legend(); axs[1, 0].legend()
    fig.tight_layout(); fig.savefig(EDA / "fig_profiles_city.png", dpi=120); plt.close(fig)
    # example cells: strongest daytime, strongest night-time, most balanced (2023 weekday profile)
    wk = P_all[2023]["weekday"]; ratio = (wk[:, 14] + 1) / (wk[:, 3] + 1)
    big = wk.mean(1) > 500
    ex = {"office-like (max day/night)": int(np.argmax(np.where(big, ratio, 0))),
          "residential-like (min day/night)": int(np.argmin(np.where(big, ratio, 9e9))),
          "largest mean": int(np.argmax(wk.mean(1)))}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    for ax, (lab, i) in zip(axs, ex.items()):
        ax.plot(P_all[2023]["weekly_norm"][i] * 168, lw=1)
        ax.set_title(f"{lab}: {grid.dong_major_name.values[cells[i]]} (mean {wk[i].mean():,.0f})"); ax.set_ylabel("normalised (mean=1)"); ax.grid(alpha=.3)
        for d in range(1, 7): ax.axvline(24 * d, color="k", lw=.3)
    fig.tight_layout(); fig.savefig(EDA / "fig_profiles_examples.png", dpi=120); plt.close(fig)
    lines += ["## City-level profile (2023 weekday, millions)", "",
              f"- min {city[2023]['weekday'].min()/1e6:.2f} at {int(np.argmin(city[2023]['weekday']))}h, max {city[2023]['weekday'].max()/1e6:.2f} at {int(np.argmax(city[2023]['weekday']))}h; weekend max {city[2023]['weekend'].max()/1e6:.2f}",
              f"- monthly means 2023 (M): {np.round(city[2023]['monthly'].mean(axis=1)/1e6, 2).tolist()}", ""]

    # ---------- composition (regime A, 2023) ----------
    Xc, tic, _ = load_canonical(months_between("202301", "202312"))
    Xc = Xc[:, cells]
    calc = calendar_features(tic)
    ok = ~np.isnan(Xc[:, 0, 0, 0])
    comp = np.zeros((3, 24, len(cells), 14, 2), np.float32)
    for d in range(3):
        for h in range(24):
            m = ok & (calc.daytype == d).values & (calc.hour == h).values
            mean = np.nanmean(Xc[m], axis=0)                           # [N, 14, 2]
            comp[d, h] = mean / np.maximum(mean.sum(axis=(1, 2), keepdims=True), 1e-6)
    np.savez_compressed(EDA / "composition_2023.npz", cells=cells, shares=comp, age_labels=np.array(AGE_LABELS), sex_labels=np.array(SEX_LABELS),
                        note="P(daytype, hour, cell, age, sex): mean 2023 population share; 0-9 included here for completeness but is a derived column")
    city_comp = np.nanmean(comp[0, 14], axis=0)
    lines += ["## Composition (2023 weekday 14h, city mean shares by age band, M/F)", "",
              pd.DataFrame(city_comp, index=AGE_LABELS, columns=SEX_LABELS).round(4).to_markdown(), ""]

    # ---------- SVD per year and pooled ----------
    svd_out = {}
    for y in YEARS:
        X, ti, cal = data[y]
        ok = ~np.isnan(X[:, 0])
        svd_out[y] = svd_analysis(X[ok][:, cells], str(y), grid, cells, ti[ok].reset_index(drop=True))
    Xall = np.concatenate([data[y][0] for y in YEARS]); tiall = pd.concat([data[y][1] for y in YEARS], ignore_index=True)
    ok = ~np.isnan(Xall[:, 0])
    svd_out["all"] = svd_analysis(Xall[ok][:, cells], "all", grid, cells, tiall[ok].reset_index(drop=True))
    lines += ["## Low-rank structure (energy captured by rank r)", "", "| data | variant | r=1 | r=2 | r=5 | r=10 | r=20 | r=50 |", "|---|---|---|---|---|---|---|---|"]
    for name, o in svd_out.items():
        for v in ("raw", "centred"):
            e = o[v]["energy"]
            lines.append(f"| {name} | {v} | " + " | ".join(f"{e[r-1]:.4f}" for r in (1, 2, 5, 10, 20, 50)) + " |")
    # year-to-year stability of spatial subspaces (centred)
    rows = []
    for r in (1, 3, 5, 10, 20):
        for a, b in ((2023, 2024), (2024, 2025), (2023, 2025)):
            ang, ov = principal_angles(svd_out[a]["centred"]["V"], svd_out[b]["centred"]["V"], r)
            rows.append({"r": r, "years": f"{a}-{b}", "overlap": ov, "max_angle_deg": ang.max(), "mean_angle_deg": ang.mean()})
    stab = pd.DataFrame(rows); stab.to_csv(EDA / "svd_year_stability.csv", index=False)
    lines += ["", "## Year-to-year stability of the leading spatial subspace (centred SVD; overlap = mean squared cosine of principal angles)", "",
              stab.pivot(index="r", columns="years", values="overlap").round(3).to_markdown(), "",
              "Leading spatial modes 1-3, correlation between years:", ""]
    for k in range(3):
        c = np.corrcoef(np.stack([svd_out[y]["centred"]["V"][:, k] for y in YEARS]))
        lines.append(f"- v{k+1}: corr(2023,2024)={c[0,1]:.3f}, corr(2024,2025)={c[1,2]:.3f}, corr(2023,2025)={c[0,2]:.3f}")
    (EDA / "task1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
