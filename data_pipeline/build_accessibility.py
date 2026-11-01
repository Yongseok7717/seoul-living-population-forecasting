"""Accessibility index A_{i,g} and population pressure R_{i,g} (proposal section 6), population-only baseline.

A_{i,g} = sum_f  w_f,g  C_f  K(d_if),  K(d) = exp(-(d/r_f)^2 / 2) truncated at 2 r_f  (Gaussian, r_f = service radius)
C_f = 1 unless a capacity proxy exists (shelter area, clinic beds). Capacity-free values are *accessibility potentials*.
R_{i,g}(t) = X_{i,g}(t) / (A_{i,g} + eps)

Groups and the facility sets that serve them:
  all10   : hospital+clinic (health), shelters (evacuation)
  age65   : hospital+clinic, senior_center (welfare)
  age1019 : schools
Population fields come from the atlas (night 03h / afternoon 14h weekday means; 65+ and 10-19 shares from regime A).

Outputs (processed/accessibility/): accessibility_cells.parquet, fig_accessibility.png, fig_pressure.png, summary.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import config as C
from loader import load_grid, to_lattice

OUT = C.OUT / "accessibility"
RADIUS = {"shelter_eq_outdoor": 500, "shelter_eq_indoor": 500, "shelter_civil_defense": 500,
          "hospital": 2000, "clinic": 750, "emergency": 3000, "school_elem": 1000, "school_mid": 1500, "school_high": 2000,
          "senior_center": 500, "senior_welfare_center": 1000}
CAPACITY = {"shelter_eq_outdoor": "area_m2", "shelter_eq_indoor": "area_m2", "clinic": "beds", "emergency": "er_beds"}
SERVICE_SETS = {
    "health_all10": ["hospital", "clinic", "emergency"],
    "shelter_all10": ["shelter_eq_outdoor", "shelter_eq_indoor", "shelter_civil_defense"],
    "health_age65": ["hospital", "clinic", "emergency"],
    "welfare_age65": ["senior_center", "senior_welfare_center"],
    "school_age1019": ["school_elem", "school_mid", "school_high"],
}


def kernel_sum(tree_xy, pts, cap, r):
    """sum_f cap_f * exp(-(d/r)^2/2) for facilities within 2r of each cell."""
    tree = cKDTree(pts)
    out = np.zeros(len(tree_xy))
    for i, hits in enumerate(tree.query_ball_point(tree_xy, 2 * r)):
        if hits:
            d = np.linalg.norm(pts[hits] - tree_xy[i], axis=1)
            out[i] = np.sum(cap[hits] * np.exp(-0.5 * (d / r) ** 2))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=False)
    xy = grid[["CELL_X", "CELL_Y"]].values.astype(float)
    fac = pd.read_parquet(C.OUT / "facilities" / "facilities.parquet")
    fac = fac[fac.has_coords]
    atlas = pd.read_parquet(C.OUT / "atlas" / "atlas_cells.parquet").set_index("cell_idx")

    acc = pd.DataFrame({"cell_idx": grid.cell_idx.values})
    for t, r in RADIUS.items():
        d = fac[fac.ftype == t]
        pts = d[["x", "y"]].values.astype(float)
        cap = np.ones(len(d))
        if t in CAPACITY and CAPACITY[t] in d.columns:
            c = pd.to_numeric(d[CAPACITY[t]], errors="coerce")
            cap = c.fillna(c.median() if c.notna().any() else 1.0).clip(lower=1).values
            cap = cap / np.median(cap)                        # relative capacity, median facility = 1
        acc[f"A_{t}"] = kernel_sum(xy, pts, cap, r)
    for name, types in SERVICE_SETS.items():
        acc[f"A_{name}"] = sum(acc[f"A_{t}"] for t in types)

    # population fields (weekday means) for the groups
    pop = pd.DataFrame(index=grid.cell_idx.values)
    pop["all10_night"] = atlas.pop10_night; pop["all10_day"] = atlas.pop10_day
    pop["age65_night"] = atlas.pop10_night_regimeA * atlas.share_65plus_night
    pop["age65_day"] = atlas.pop10_day_regimeA * atlas.share_65plus_day
    pop["age1019_day"] = atlas.pop10_day_regimeA * atlas.share_10_19_day
    eps = 0.05
    acc = acc.set_index("cell_idx")
    R = pd.DataFrame(index=acc.index)
    R["R_health_all10_night"] = pop.all10_night / (acc.A_health_all10 + eps)
    R["R_health_all10_day"] = pop.all10_day / (acc.A_health_all10 + eps)
    R["R_shelter_all10_night"] = pop.all10_night / (acc.A_shelter_all10 + eps)
    R["R_shelter_all10_day"] = pop.all10_day / (acc.A_shelter_all10 + eps)
    R["R_health_age65_night"] = pop.age65_night / (acc.A_health_age65 + eps)
    R["R_health_age65_day"] = pop.age65_day / (acc.A_health_age65 + eps)
    R["R_welfare_age65_day"] = pop.age65_day / (acc.A_welfare_age65 + eps)
    R["R_school_age1019_day"] = pop.age1019_day / (acc.A_school_age1019 + eps)
    out = acc.join(pop).join(R).reset_index().rename(columns={"index": "cell_idx"})
    out["populated"] = atlas.populated.reindex(out.cell_idx).values
    out.to_parquet(OUT / "accessibility_cells.parquet", index=False)

    populated = out.populated.values.astype(bool)
    def panel(ax, v, title, cmap="viridis", log=True, q=(0.02, 0.98)):
        v = np.where(populated, v, np.nan).astype(float)
        if log:
            v = np.log10(v + 1e-3)
        lo, hi = np.nanquantile(v, q)
        im = ax.imshow(to_lattice(v, grid), origin="lower", cmap=cmap, vmin=lo, vmax=hi); ax.set_title(title, fontsize=10); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.035)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panel(axs[0, 0], out.A_health_all10, "log10 A 보건(병원 2km + 의원 750m, 병상 가중)")
    panel(axs[0, 1], out.A_shelter_all10, "log10 A 대피소(지진옥외·실내·민방위, 500m)")
    panel(axs[0, 2], out.A_welfare_age65, "log10 A 경로당(500m, 일부 자치구만)")
    panel(axs[1, 0], out.A_school_age1019, "log10 A 학교(초1km·중1.5km·고2km)")
    panel(axs[1, 1], out.A_hospital, "log10 A 병원만(2km)")
    panel(axs[1, 2], out.A_shelter_civil_defense, "log10 A 민방위대피시설(500m)")
    fig.suptitle("접근성 지수 A (Gaussian kernel, capacity 없으면 potential)"); fig.tight_layout(); fig.savefig(OUT / "fig_accessibility.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    panel(axs[0, 0], out.R_shelter_all10_night, "log10 R 대피소 압력, 야간 03h (전체 10+)", cmap="magma")
    panel(axs[0, 1], out.R_shelter_all10_day, "log10 R 대피소 압력, 주간 14h", cmap="magma")
    panel(axs[0, 2], np.log2((out.R_shelter_all10_day + 1) / (out.R_shelter_all10_night + 1)), "log2 주간/야간 대피소 압력비", cmap="RdBu_r", log=False)
    panel(axs[1, 0], out.R_health_all10_day, "log10 R 보건 압력, 주간 (전체 10+)", cmap="magma")
    panel(axs[1, 1], out.R_health_age65_night, "log10 R 보건 압력, 야간 (65+)", cmap="magma")
    panel(axs[1, 2], out.R_school_age1019_day, "log10 R 학교 압력, 주간 (10-19)", cmap="magma")
    fig.suptitle("인구 압력 R = X / (A + eps): 값이 클수록 시설 대비 인구가 많음"); fig.tight_layout(); fig.savefig(OUT / "fig_pressure.png", dpi=120); plt.close(fig)

    p = out[populated]
    lines = ["# Accessibility / pressure summary", "",
             f"- radii (m): {RADIUS}", f"- capacity proxies: {CAPACITY} (relative to median facility); others = 1 (potential)", "",
             "| indicator | cells with A=0 | p10 | median | p90 | max |", "|---|---|---|---|---|---|"]
    for c in [c for c in out.columns if c.startswith("A_") and c[2:] in SERVICE_SETS] + [c for c in out.columns if c.startswith("R_")]:
        q = p[c].quantile([.1, .5, .9]).values
        lines.append(f"| {c} | {int((p[c.replace('R_', 'A_').rsplit('_', 1)[0] if c.startswith('R_') else c] == 0).sum())} | {q[0]:.3g} | {q[1]:.3g} | {q[2]:.3g} | {p[c].max():.3g} |")
    lines += ["", "## Top 15 cells by daytime shelter pressure", "",
              p.sort_values("R_shelter_all10_day", ascending=False).head(15)[["cell_idx", "all10_day", "A_shelter_all10", "R_shelter_all10_day"]]
              .assign(dong=lambda d: grid.dong_major_name.reindex(d.cell_idx).values).to_markdown(index=False, floatfmt=".1f"), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
