"""Task 2: functional urban clustering from living-population dynamics (total = pop10, 2023 unless stated).

2.2  baseline: k-means / GMM / spectral on the normalised weekly profile x_i in R^168 (holidays excluded)
2.3  reduced: k-means on the SVD spatial embedding z_i = V_r(i,:) (centred SVD of 2023) and on PCA of the profiles
2.4-2.5 cluster profiles, magnitude, maps
2.6  weekday vs weekend clusterings and the transition matrix
2.7  seasonal / monthly clusterings (ARI, NMI against the annual clustering)
2.8  demographic composition per cluster (regime A composition, 2023)
2.9  temporal + demographic clustering
2.10 robustness over K, normalisation, r, algorithm, year (ARI/NMI, silhouette, Davies-Bouldin, spatial coherence)
2.11 outputs: labels parquet, cluster profiles, maps, transition matrices, embeddings

Outputs: processed/eda/cluster_* , fig_cluster_*.png, task2_summary.md
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
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import config as C                                                # noqa: E402
from loader import load_grid, load_adjacency, to_lattice          # noqa: E402
from canonical import EDA, AGE_LABELS, SEX_LABELS             # noqa: E402

K_MAIN = 6
K_RANGE = [4, 5, 6, 7, 8, 10]
SEED = 0
SEASONS = ["spring", "summer", "autumn", "winter"]
MIN_MEAN = 20      # cells with mean weekday population below this are too noisy for shape clustering


def zscore_rows(P):
    return (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)


def kmeans(X, k, seed=SEED):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(X)


def spatial_coherence(labels, A):
    """share of rook-neighbour pairs with the same label."""
    A = A.tocoo()
    m = A.row < A.col
    return float((labels[A.row[m]] == labels[A.col[m]]).mean())


def order_labels(labels, weekly_norm):
    """relabel clusters by daytime share (weekday 14h share, ascending) so numbering is comparable across runs."""
    k = labels.max() + 1
    day = np.array([weekly_norm[labels == c][:, 14 + 24 * np.arange(5)].mean() for c in range(k)])
    order = np.argsort(day)
    remap = np.empty(k, int); remap[order] = np.arange(k)
    return remap[labels]


def describe_clusters(labels, prof, weekly, weekly_norm, grid, cells, name, lines):
    k = labels.max() + 1
    rows = []
    for c in range(k):
        m = labels == c
        wn = weekly_norm[m].mean(0) * 168
        wd = wn[:24 * 5].reshape(5, 24).mean(0); we = wn[24 * 5:].reshape(2, 24).mean(0)
        rows.append({"cluster": c, "n_cells": int(m.sum()), "mean_pop": float(weekly[m].mean()),
                     "wd_14h": float(wd[14]), "wd_03h": float(wd[3]), "wd_21h": float(wd[21]), "we_14h": float(we[14]), "we_21h": float(we[21]),
                     "day/night": float((wd[14] + 1e-6) / (wd[3] + 1e-6)), "weekend/weekday_14h": float((we[14] + 1e-6) / (wd[14] + 1e-6)),
                     "top_dongs": ", ".join(pd.Series(grid.dong_major_name.values[cells[m]]).value_counts().head(5).index)})
    t = pd.DataFrame(rows)
    lines += [f"### {name}", "", t.to_markdown(index=False, floatfmt=".2f"), ""]
    return t


def plot_clusters(labels, weekly_norm, grid, cells, title, fname, k=None):
    k = k or labels.max() + 1
    cmap = plt.get_cmap("tab10", k)
    fig = plt.figure(figsize=(17, 7))
    ax = fig.add_subplot(1, 2, 1)
    vec = np.full(len(grid), np.nan, np.float32); vec[cells] = labels
    im = ax.imshow(to_lattice(vec, grid), origin="lower", cmap=cmap, vmin=-.5, vmax=k - .5); ax.axis("off"); ax.set_title(title)
    plt.colorbar(im, ax=ax, ticks=range(k), fraction=.035)
    ax2 = fig.add_subplot(1, 2, 2)
    for c in range(k):
        ax2.plot(weekly_norm[labels == c].mean(0) * 168, color=cmap(c), lw=1.5, label=f"c{c} (n={int((labels==c).sum())})")
    for d in range(1, 7): ax2.axvline(24 * d, color="k", lw=.3)
    ax2.set_xticks(np.arange(0, 168, 24)); ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax2.set_ylabel("normalised weekly profile (mean = 1)"); ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(EDA / fname, dpi=120); plt.close(fig)


def main():
    grid = load_grid(with_geometry=False)
    P23 = np.load(EDA / "profiles_2023.npz")
    cells_all = P23["cells"]
    weekly = P23["weekly"]; weekly_norm = P23["weekly_norm"]
    keep = np.isfinite(weekly).all(1) & (weekly.mean(1) >= MIN_MEAN)
    cells = cells_all[keep]; weekly = weekly[keep]; weekly_norm = weekly_norm[keep]
    N = len(cells)
    A_rook = load_adjacency("rook")[cells][:, cells]
    lines = ["# Task 2 summary: functional clustering", "",
             f"- cells used: {N} of {len(cells_all)} (mean weekday population >= {MIN_MEAN}); features from 2023 unless stated", ""]

    # ---------- 2.2 baseline clusterings on the 168-h normalised profile ----------
    Xn = weekly_norm * 168                                           # mean 1 per cell
    Xz = zscore_rows(weekly)
    labs = {}
    labs["kmeans_norm"] = order_labels(kmeans(Xn, K_MAIN), weekly_norm)
    labs["kmeans_z"] = order_labels(kmeans(Xz, K_MAIN), weekly_norm)
    labs["gmm_norm"] = order_labels(GaussianMixture(K_MAIN, covariance_type="diag", n_init=3, random_state=SEED).fit_predict(Xn), weekly_norm)
    # ---------- 2.3 reduced representations ----------
    svd = np.load(EDA / "svd_2023.npz")
    V = svd["centred_V"]                                             # [N_all, 30] spatial modes of the centred SVD
    Z_svd = V[keep][:, :10] * svd["centred_S"][:10]                  # scaled embedding z_i
    pca = PCA(10, random_state=SEED).fit(Xn); Z_pca = pca.transform(Xn)
    labs["kmeans_svd10"] = order_labels(kmeans(Z_svd, K_MAIN), weekly_norm)
    labs["kmeans_pca10"] = order_labels(kmeans(Z_pca, K_MAIN), weekly_norm)
    labs["spectral_pca10"] = order_labels(SpectralClustering(K_MAIN, affinity="nearest_neighbors", n_neighbors=30, random_state=SEED, assign_labels="kmeans").fit_predict(Z_pca), weekly_norm)
    lines += ["## Clustering variants (K=6) vs the baseline k-means on the normalised weekly profile", "",
              "| variant | ARI vs baseline | NMI | silhouette (profile space) | Davies-Bouldin | spatial coherence |", "|---|---|---|---|---|---|"]
    base = labs["kmeans_norm"]
    for name, l in labs.items():
        lines.append(f"| {name} | {adjusted_rand_score(base, l):.3f} | {normalized_mutual_info_score(base, l):.3f} | "
                     f"{silhouette_score(Xn, l, sample_size=4000, random_state=SEED):.3f} | {davies_bouldin_score(Xn, l):.3f} | {spatial_coherence(l, A_rook):.3f} |")
    lines.append("")
    # ---------- 2.4 / 2.5 description and maps ----------
    main_lab = labs["kmeans_norm"]
    desc = describe_clusters(main_lab, None, weekly, weekly_norm, grid, cells, "Baseline k-means (K=6) on normalised weekly profile, 2023", lines)
    plot_clusters(main_lab, weekly_norm, grid, cells, "functional clusters (k-means K=6, 2023 weekly profile)", "fig_cluster_main.png")
    plot_clusters(labs["kmeans_svd10"], weekly_norm, grid, cells, "clusters on SVD embedding (r=10)", "fig_cluster_svd10.png")
    # ---------- 2.6 weekday vs weekend ----------
    wd = P23["weekday_norm"][keep] * 24; we = P23["weekend_norm"][keep] * 24
    l_wd = order_labels(kmeans(wd, K_MAIN), weekly_norm); l_we = order_labels(kmeans(we, K_MAIN), weekly_norm)
    T = pd.crosstab(l_wd, l_we, normalize="index")
    changed = float((l_wd != l_we).mean())
    lines += ["## Weekday vs weekend clustering (K=6 each, labels ordered by daytime share)", "",
              f"- cells whose cluster changes between weekday and weekend clusterings: {changed:.1%} (label ids are ordered by daytime share, so this is indicative only)",
              "", "Transition matrix P(weekend cluster | weekday cluster):", "", T.round(2).to_markdown(), ""]
    plot_clusters(l_wd, weekly_norm, grid, cells, "weekday-profile clusters", "fig_cluster_weekday.png")
    plot_clusters(l_we, weekly_norm, grid, cells, "weekend-profile clusters", "fig_cluster_weekend.png")
    # ---------- 2.7 seasonal / monthly ----------
    seasonal = P23["seasonal_norm"][keep] * 24; monthly = P23["monthly_norm"][keep] * 24
    rows = []
    for s_i, s in enumerate(SEASONS):
        l = kmeans(seasonal[:, s_i], K_MAIN)
        rows.append({"period": s, "ARI_vs_annual_hourly": adjusted_rand_score(labs["kmeans_norm"], l), "NMI": normalized_mutual_info_score(labs["kmeans_norm"], l)})
    hourly_lab = kmeans(P23["hourly_norm"][keep] * 24, K_MAIN)
    for m in range(12):
        l = kmeans(monthly[:, m], K_MAIN)
        rows.append({"period": f"month {m+1}", "ARI_vs_annual_hourly": adjusted_rand_score(hourly_lab, l), "NMI": normalized_mutual_info_score(hourly_lab, l)})
    st = pd.DataFrame(rows)
    lines += ["## Seasonal and monthly stability (K=6 on 24-h profiles; ARI/NMI against the annual 24-h clustering)", "", st.round(3).to_markdown(index=False), ""]
    # ---------- 2.8 demographic composition per cluster (2023 regime A) ----------
    comp = np.load(EDA / "composition_2023.npz")["shares"]           # [3, 24, N_all, 14, 2]
    wd14 = comp[0, 14][keep]; wd03 = comp[0, 3][keep]                # weekday 14h and 03h shares
    def agg(sh):                                                     # -> 10-19, 20-34, 35-49, 50-64, 65+, female share (ages 10+)
        s = sh[:, 1:, :]; tot = s.sum((1, 2)) + 1e-9
        return pd.DataFrame({"10-19": s[:, 0:2].sum((1, 2)) / tot, "20-34": s[:, 2:5].sum((1, 2)) / tot, "35-49": s[:, 5:8].sum((1, 2)) / tot,
                             "50-64": s[:, 8:11].sum((1, 2)) / tot, "65+": s[:, 11:].sum((1, 2)) / tot, "female": s[:, :, 1].sum(1) / tot})
    d14, d03 = agg(wd14), agg(wd03)
    comp_tab = pd.concat([d14.groupby(main_lab).mean().add_prefix("14h_"), d03.groupby(main_lab).mean().add_prefix("03h_")], axis=1)
    lines += ["## Demographic composition by cluster (2023 weekday, shares of ages 10+)", "", comp_tab.round(3).to_markdown(), ""]
    # ---------- 2.9 temporal + demographic clustering ----------
    Zd = np.concatenate([d14.values, d03.values], 1); Zd = (Zd - Zd.mean(0)) / (Zd.std(0) + 1e-9)
    Zt = (Z_pca - Z_pca.mean(0)) / (Z_pca.std(0) + 1e-9)
    l_td = order_labels(kmeans(np.concatenate([Zt, Zd], 1), K_MAIN), weekly_norm)
    l_d = kmeans(Zd, K_MAIN)
    lines += ["## Temporal + demographic clustering (PCA-10 of profile + 12 composition shares, K=6)", "",
              f"- ARI(temporal+demographic vs temporal-only) = {adjusted_rand_score(labs['kmeans_pca10'], l_td):.3f}, NMI = {normalized_mutual_info_score(labs['kmeans_pca10'], l_td):.3f}",
              f"- ARI(demographic-only vs temporal-only) = {adjusted_rand_score(labs['kmeans_pca10'], l_d):.3f}: composition carries information largely independent of the temporal shape", ""]
    describe_clusters(l_td, None, weekly, weekly_norm, grid, cells, "Temporal + demographic clusters", lines)
    comp_td = d14.groupby(l_td).mean()
    lines += ["Composition (weekday 14h) of the temporal+demographic clusters:", "", comp_td.round(3).to_markdown(), ""]
    plot_clusters(l_td, weekly_norm, grid, cells, "temporal + demographic clusters", "fig_cluster_temporal_demographic.png")
    # ---------- 2.10 robustness ----------
    rob = []
    for k in K_RANGE:
        l = kmeans(Xn, k)
        rob.append({"setting": f"K={k}", "silhouette": silhouette_score(Xn, l, sample_size=4000, random_state=SEED), "davies_bouldin": davies_bouldin_score(Xn, l),
                    "spatial_coherence": spatial_coherence(l, A_rook), "ARI_vs_K6": adjusted_rand_score(base, l)})
    for r in (3, 5, 10, 20):
        l = order_labels(kmeans(V[keep][:, :r] * svd["centred_S"][:r], K_MAIN), weekly_norm)
        rob.append({"setting": f"SVD r={r}", "silhouette": silhouette_score(Xn, l, sample_size=4000, random_state=SEED), "davies_bouldin": davies_bouldin_score(Xn, l),
                    "spatial_coherence": spatial_coherence(l, A_rook), "ARI_vs_K6": adjusted_rand_score(base, l)})
    for seed in (1, 2):
        l = order_labels(kmeans(Xn, K_MAIN, seed), weekly_norm)
        rob.append({"setting": f"seed {seed}", "silhouette": np.nan, "davies_bouldin": np.nan, "spatial_coherence": spatial_coherence(l, A_rook), "ARI_vs_K6": adjusted_rand_score(base, l)})
    for y in (2024, 2025):
        Py = np.load(EDA / f"profiles_{y}.npz"); wy = Py["weekly_norm"][keep] * 168
        okk = np.isfinite(wy).all(1)
        l = order_labels(kmeans(np.nan_to_num(wy), K_MAIN), weekly_norm)
        rob.append({"setting": f"year {y} profile", "silhouette": np.nan, "davies_bouldin": np.nan, "spatial_coherence": spatial_coherence(l, A_rook), "ARI_vs_K6": adjusted_rand_score(base[okk], l[okk])})
        labs[f"kmeans_norm_{y}"] = l
    rb = pd.DataFrame(rob)
    lines += ["## Robustness (baseline features unless stated; ARI against the K=6 2023 baseline)", "", rb.round(3).to_markdown(index=False), ""]
    # ---------- 2.11 outputs ----------
    out = pd.DataFrame({"cell_idx": cells, "CELL_ID": grid.CELL_ID.values[cells], "dong": grid.dong_major_name.values[cells], "mean_weekday_pop": weekly.mean(1)})
    for name, l in labs.items():
        out[f"c_{name}"] = l
    out["c_weekday"] = l_wd; out["c_weekend"] = l_we; out["c_temporal_demographic"] = l_td
    for j in range(10):
        out[f"z_svd{j+1}"] = Z_svd[:, j]; out[f"z_pca{j+1}"] = Z_pca[:, j]
    out.to_parquet(EDA / "cluster_labels.parquet", index=False)
    np.savez_compressed(EDA / "cluster_profiles.npz", cells=cells, labels=main_lab,
                        profiles=np.stack([weekly_norm[main_lab == c].mean(0) for c in range(K_MAIN)]),
                        profiles_abs=np.stack([weekly[main_lab == c].mean(0) for c in range(K_MAIN)]))
    T.to_csv(EDA / "cluster_transition_weekday_weekend.csv", encoding="utf-8-sig")
    (EDA / "task2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
