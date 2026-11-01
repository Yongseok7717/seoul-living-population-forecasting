"""Task 3: relational structures over the 250 m grid (nodes = all 10,125 cells; cell_idx order of processed/grid).

Outputs (processed/relations/):
  spatial_<name>.npz            W (CSR, symmetric, Gaussian weights sigma=500 m), L_comb = D - W, L_norm = I - D^-1/2 W D^-1/2
                                names: rook, queen, radius500, radius750, knn8
  func_<name>.npz               incidence H (CSR N x M), weights w (M), L_norm; names: hard_k6, hard_k4, soft_k6,
                                weekday_k6, weekend_k6, season_<s>_k6, group_<g>_k6 (65+, 10-19, 20-34, 50-64, female, male)
  func_similarity_knn10.npz     functional kNN graph (cosine on PCA-10 embedding) with geographic distances of the edges
  fac_<type>.npz                per facility type: hard incidence (d <= r), kernel incidence exp(-d^2/2r^2) truncated at 2r,
                                hyperedge weights (relative capacity if available), L_norm (hard)
  fac_combined.npz              union of all facility hypergraphs (block-concatenated incidence), L_norm
  fac_group_<g>.npz             demographic-specific operator L_{H,g} = sum_m omega_{g,m} L_{H,m} (weights in demographic_weights.json)
  relations_manifest.json       what exists, sizes, parameters, ablation configurations
  comparison.md, fig_*.png      connectivity statistics and spectral comparison (spectral part in spectral_analysis.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import config as C                                                   # noqa: E402
from loader import load_grid, load_adjacency                         # noqa: E402

REL = C.OUT / "relations"
EDA = C.OUT / "eda"
SIGMA_M = 500.0
FACILITY_RADIUS = {"hospital": 2000, "clinic": 750, "emergency": 3000, "shelter_eq_outdoor": 500, "shelter_eq_indoor": 500,
                   "shelter_civil_defense": 500, "school_elem": 1000, "school_mid": 1500, "school_high": 2000,
                   "senior_center": 500, "senior_welfare_center": 1000, "nursing_home": 1000}
CAPACITY = {"shelter_eq_outdoor": "area_m2", "shelter_eq_indoor": "area_m2", "clinic": "beds", "emergency": "er_beds", "nursing_home": "capacity"}
# transparent demographic weights omega_{g,m} (Task 3.11); 0-9 excluded by project decision, so no kindergarten/childcare
DEMO_WEIGHTS = {
    "age10_19": {"school_elem": 0.5, "school_mid": 1.0, "school_high": 1.0, "hospital": 0.3, "clinic": 0.3, "emergency": 0.3,
                 "shelter_eq_outdoor": 1.0, "shelter_eq_indoor": 1.0, "shelter_civil_defense": 1.0},
    "age20_64": {"hospital": 1.0, "clinic": 1.0, "emergency": 1.0, "shelter_eq_outdoor": 1.0, "shelter_eq_indoor": 1.0, "shelter_civil_defense": 1.0},
    "age65p": {"hospital": 1.0, "clinic": 1.0, "emergency": 1.0, "senior_center": 1.0, "senior_welfare_center": 1.0, "nursing_home": 0.5,
               "shelter_eq_outdoor": 1.0, "shelter_eq_indoor": 1.0, "shelter_civil_defense": 1.0},
    "foreign": {"hospital": 1.0, "clinic": 0.5, "emergency": 1.0, "shelter_eq_outdoor": 1.0, "shelter_eq_indoor": 1.0, "shelter_civil_defense": 1.0},
}


def laplacians(W):
    d = np.asarray(W.sum(1)).ravel()
    L_comb = (sp.diags(d) - W).tocsr()
    dis = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L_norm = (sp.diags((d > 0).astype(np.float32)) - sp.diags(dis) @ W @ sp.diags(dis)).tocsr()
    return L_comb.astype(np.float32), L_norm.astype(np.float32)


def hyper_laplacian(H, w=None):
    """L = I - Dv^-1/2 H W De^-1 H^T Dv^-1/2 ; nodes without hyperedges get a zero row."""
    H = sp.csr_matrix(H, dtype=np.float32)
    w = np.ones(H.shape[1], np.float32) if w is None else np.asarray(w, np.float32)
    de = np.asarray(H.sum(0)).ravel(); dv = np.asarray((H @ sp.diags(w)).sum(1)).ravel()
    dvis = np.where(dv > 0, 1 / np.sqrt(np.maximum(dv, 1e-12)), 0.0); dei = np.where(de > 0, 1 / np.maximum(de, 1e-12), 0.0)
    P = sp.diags(dvis) @ H @ sp.diags(w * dei) @ H.T @ sp.diags(dvis)
    return (sp.diags((dv > 0).astype(np.float32)) - P).tocsr().astype(np.float32)


def save_npz(path, **arrays):
    out = {}
    for k, v in arrays.items():
        if sp.issparse(v):
            v = v.tocsr(); out[f"{k}__data"] = v.data; out[f"{k}__indices"] = v.indices; out[f"{k}__indptr"] = v.indptr; out[f"{k}__shape"] = np.array(v.shape)
        else:
            out[k] = v
    np.savez_compressed(path, **out)


def hyperedge_stats(H, xy):
    """size, radius of gyration (m) and share of member pairs farther than 5 km, per hyperedge (sampled)."""
    H = sp.csc_matrix(H)
    sizes = np.diff(H.indptr)
    rg, far = [], []
    for e in range(H.shape[1]):
        idx = H.indices[H.indptr[e]:H.indptr[e + 1]]
        if len(idx) < 2:
            rg.append(0.0); far.append(0.0); continue
        p = xy[idx]; c = p.mean(0); rg.append(float(np.sqrt(((p - c) ** 2).sum(1).mean())))
        if len(idx) > 400:
            idx = np.random.default_rng(0).choice(idx, 400, replace=False); p = xy[idx]
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)); far.append(float((d[np.triu_indices(len(p), 1)] > 5000).mean()))
    return pd.DataFrame({"size": sizes, "radius_gyration_m": rg, "share_pairs_gt_5km": far})


def main():
    REL.mkdir(parents=True, exist_ok=True)
    grid = load_grid(with_geometry=False)
    N = len(grid); xy = grid[["CELL_X", "CELL_Y"]].values.astype(float)
    manifest = {"n_nodes": int(N), "node_order": "processed/grid/grid_index.parquet cell_idx", "sigma_m": SIGMA_M, "spatial": {}, "functional": {}, "facility": {}}
    lines = ["# Task 3 relational structures: connectivity comparison", "", f"- nodes: all {N} grid cells", ""]

    # ---------- 3.2 spatial graphs ----------
    tree = cKDTree(xy)
    def gauss(d):
        return np.exp(-d ** 2 / (2 * SIGMA_M ** 2)).astype(np.float32)
    spatial = {}
    for name in ("rook", "queen"):
        A = load_adjacency(name).tocoo()
        d = np.sqrt(((xy[A.row] - xy[A.col]) ** 2).sum(1)); spatial[name] = sp.csr_matrix((gauss(d), (A.row, A.col)), shape=(N, N))
    for r in (500, 750):
        pairs = tree.query_pairs(r, output_type="ndarray")
        d = np.sqrt(((xy[pairs[:, 0]] - xy[pairs[:, 1]]) ** 2).sum(1)); w = gauss(d)
        W = sp.csr_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(N, N)); spatial[f"radius{r}"] = W
    dk, ik = tree.query(xy, k=9)
    rows = np.repeat(np.arange(N), 8); cols = ik[:, 1:].ravel(); w = gauss(dk[:, 1:].ravel())
    W = sp.csr_matrix((w, (rows, cols)), shape=(N, N)); spatial["knn8"] = W.maximum(W.T)
    lines += ["## Spatial graphs (Gaussian weights, sigma 500 m)", "", "| graph | edges | mean degree | isolated nodes |", "|---|---|---|---|"]
    for name, W in spatial.items():
        Lc, Ln = laplacians(W); deg = np.diff(W.indptr)
        save_npz(REL / f"spatial_{name}.npz", W=W, L_comb=Lc, L_norm=Ln)
        manifest["spatial"][name] = {"edges": int(W.nnz // 2), "mean_degree": float(deg.mean())}
        lines.append(f"| {name} | {W.nnz//2:,} | {deg.mean():.2f} | {int((deg==0).sum())} |")
    lines.append("")

    # ---------- 3.4 functional similarity ----------
    lab = pd.read_parquet(EDA / "cluster_labels.parquet")
    cells = lab.cell_idx.values; Nf = len(cells)
    Z = lab[[f"z_pca{j}" for j in range(1, 11)]].values
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Zt = torch.tensor(Zn, device=dev, dtype=torch.float32)
    S = Zt @ Zt.T; S.fill_diagonal_(-2)
    topv, topi = torch.topk(S, 10, dim=1)
    topi = topi.cpu().numpy(); topv = topv.cpu().numpy()
    rows = np.repeat(np.arange(Nf), 10); cols = topi.ravel()
    gd = np.sqrt(((xy[cells[rows]] - xy[cells[cols]]) ** 2).sum(1))
    Wf = sp.csr_matrix((topv.ravel().astype(np.float32), (cells[rows], cells[cols])), shape=(N, N)); Wf = Wf.maximum(Wf.T)
    save_npz(REL / "func_similarity_knn10.npz", W=Wf, L_norm=laplacians(Wf)[1], edge_geo_distance_m=gd)
    q = np.quantile(gd, [0.1, 0.25, 0.5, 0.75, 0.9])
    lines += ["## Functional similarity (cosine on PCA-10 of the normalised weekly profile, 10 nearest functional neighbours)", "",
              f"- geographic distance of functional neighbours (m): p10 {q[0]:.0f}, p25 {q[1]:.0f}, median {q[2]:.0f}, p75 {q[3]:.0f}, p90 {q[4]:.0f}; share > 5 km: {(gd>5000).mean():.1%}",
              f"- share of functional neighbours that are also rook neighbours: {float(np.mean([spatial['rook'][cells[r], cells[c]] > 0 for r, c in zip(rows[:20000], cols[:20000])])):.1%} (sample)", ""]

    # ---------- 3.5-3.7 functional hypergraphs ----------
    func = {}
    def hard(labels, name):
        k = int(labels.max() + 1); H = sp.csr_matrix((np.ones(Nf, np.float32), (cells, labels)), shape=(N, k)); func[name] = (H, None)
    hard(lab.c_kmeans_norm.values, "hard_k6")
    km4 = KMeans(4, n_init=10, random_state=0).fit(Z); hard(km4.labels_, "hard_k4")
    km6 = KMeans(6, n_init=10, random_state=0).fit(Z)
    d2 = km6.transform(Z) ** 2; tau = np.median(d2.min(1))
    logits = -d2 / tau; logits -= logits.max(1, keepdims=True)          # stable softmax (far-away cells underflowed to 0/0 before)
    Pm = np.exp(logits); Pm /= Pm.sum(1, keepdims=True)
    Hs = sp.csr_matrix((Pm.ravel().astype(np.float32), (np.repeat(cells, 6), np.tile(np.arange(6), Nf))), shape=(N, 6)); func["soft_k6"] = (Hs, None)
    hard(lab.c_weekday.values, "weekday_k6"); hard(lab.c_weekend.values, "weekend_k6")
    P23 = np.load(EDA / "profiles_2023.npz"); keep = pd.Series(np.arange(len(P23["cells"])), index=P23["cells"]).reindex(cells).values
    for si, s in enumerate(["spring", "summer", "autumn", "winter"]):
        prof = np.nan_to_num(P23["seasonal_norm"][keep][:, si] * 24)
        hard(KMeans(6, n_init=10, random_state=0).fit_predict(prof), f"season_{s}_k6")
    if (EDA / "group_cluster_labels.parquet").exists():          # demographic-group hypergraphs: not used in the paper, built if available
        glab = pd.read_parquet(EDA / "group_cluster_labels.parquet").set_index("cell_idx").reindex(cells)
        for g in ("65+", "10-19", "20-34", "50-64", "female", "male"):
            hard(glab[f"c_{g}"].values.astype(int), f"group_{g}_k6")
    lines += ["## Functional hypergraphs (hyperedge = cluster; soft = k-means membership softmax)", "", "| hypergraph | hyperedges | mean size | mean radius of gyration (km) | share of member pairs > 5 km |", "|---|---|---|---|---|"]
    for name, (H, w) in func.items():
        L = hyper_laplacian(H, w); save_npz(REL / f"func_{name}.npz", H=H, L_norm=L, w=np.ones(H.shape[1], np.float32) if w is None else w)
        st = hyperedge_stats((H > 0.2).astype(np.float32) if name.startswith("soft") else H, xy)
        manifest["functional"][name] = {"hyperedges": int(H.shape[1]), "mean_size": float(st["size"].mean())}
        lines.append(f"| {name} | {H.shape[1]} | {st['size'].mean():.0f} | {st.radius_gyration_m.mean()/1000:.1f} | {st.share_pairs_gt_5km.mean():.2f} |")
    # stability across time (Jaccard of best-matched hyperedges) and across groups
    def best_jaccard(H1, H2):
        A = (H1 > 0).astype(np.float32).T @ (H2 > 0).astype(np.float32)             # intersections [k1, k2]
        s1 = np.asarray((H1 > 0).sum(0)).ravel(); s2 = np.asarray((H2 > 0).sum(0)).ravel()
        J = A.toarray() / (s1[:, None] + s2[None, :] - A.toarray() + 1e-9)
        return float(np.mean(J.max(1)))
    base = func["hard_k6"][0]
    lines += ["", "Stability of functional hyperedges (mean best-match Jaccard against the annual K=6 hypergraph):", ""]
    for name in ("weekday_k6", "weekend_k6", "season_spring_k6", "season_summer_k6", "season_autumn_k6", "season_winter_k6", "group_65+_k6", "group_10-19_k6", "group_20-34_k6", "group_male_k6", "group_female_k6"):
        if name in func:
            lines.append(f"- {name}: {best_jaccard(base, func[name][0]):.3f}")
    lines.append("")

    # ---------- 3.8-3.11 facility hypergraphs ----------
    fac = pd.read_parquet(C.OUT / "facilities" / "facilities.parquet")
    fac = fac[fac.has_coords]
    fac_L = {}
    lines += ["## Facility hypergraphs (hard: d <= r; kernel: exp(-d^2/2r^2) truncated at 2r)", "", "| type | facilities | r (m) | mean hyperedge size | nodes covered | nodes with >= 5 hyperedges | capacity weight |", "|---|---|---|---|---|---|---|"]
    blocks, wblocks, types_present = [], [], []
    for t, r in FACILITY_RADIUS.items():
        d = fac[fac.ftype == t]
        if len(d) == 0:
            continue
        pts = d[["x", "y"]].values.astype(float)
        cap = np.ones(len(d), np.float32)
        if t in CAPACITY and CAPACITY[t] in d:
            c = pd.to_numeric(d[CAPACITY[t]], errors="coerce"); cap = (c.fillna(c.median() if c.notna().any() else 1).clip(lower=1) / max(c.median(), 1)).values.astype(np.float32)
        rows, cols, vals, rows2, cols2, vals2 = [], [], [], [], [], []
        for e, hits in enumerate(tree.query_ball_point(pts, 2 * r)):
            if not hits:
                continue
            dd = np.sqrt(((xy[hits] - pts[e]) ** 2).sum(1))
            k = np.exp(-dd ** 2 / (2 * r ** 2))
            rows2 += hits; cols2 += [e] * len(hits); vals2 += list(k)
            m = dd <= r; rows += list(np.array(hits)[m]); cols += [e] * int(m.sum()); vals += [1.0] * int(m.sum())
        Hh = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(N, len(d)))
        Hk = sp.csr_matrix((np.array(vals2, np.float32), (rows2, cols2)), shape=(N, len(d)))
        L = hyper_laplacian(Hh, cap); fac_L[t] = L
        save_npz(REL / f"fac_{t}.npz", H_hard=Hh, H_kernel=Hk, w=cap, L_norm=L, fid=d.fid.values)
        deg = np.diff(Hh.indptr) if False else np.asarray((Hh > 0).sum(1)).ravel()
        manifest["facility"][t] = {"facilities": int(len(d)), "radius_m": r, "nodes_covered": int((deg > 0).sum())}
        lines.append(f"| {t} | {len(d)} | {r} | {np.asarray(Hh.sum(0)).mean():.1f} | {int((deg>0).sum())} | {int((deg>=5).sum())} | {'yes' if t in CAPACITY else 'no'} |")
        blocks.append(Hh); wblocks.append(cap); types_present.append(t)
    Hall = sp.hstack(blocks).tocsr(); wall = np.concatenate(wblocks)
    L_all = hyper_laplacian(Hall, wall); save_npz(REL / "fac_combined.npz", H_hard=Hall, w=wall, L_norm=L_all, types=np.array(types_present), block_sizes=np.array([b.shape[1] for b in blocks]))
    lines.append("")
    # demographic-specific operators
    for g, wts in DEMO_WEIGHTS.items():
        Lg = sum(w * fac_L[m] for m, w in wts.items() if m in fac_L)
        save_npz(REL / f"fac_group_{g}.npz", L=Lg.tocsr(), weights=json.dumps(wts))
    (REL / "demographic_weights.json").write_text(json.dumps(DEMO_WEIGHTS, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- 3.13 overlap between functional clusters and facility coverage ----------
    from sklearn.metrics import normalized_mutual_info_score
    lab_full = np.full(N, -1); lab_full[cells] = lab.c_kmeans_norm.values
    lines += ["## Overlap of functional clusters with facility coverage (NMI between cluster label and coverage tercile on clustered cells)", ""]
    for t in types_present:
        Hh = blocks[types_present.index(t)]; deg = np.asarray((Hh > 0).sum(1)).ravel()[cells]
        ter = pd.qcut(deg + 1e-6 * np.random.default_rng(0).random(len(deg)), 3, labels=False, duplicates="drop")
        lines.append(f"- {t}: NMI = {normalized_mutual_info_score(lab.c_kmeans_norm.values, ter):.3f}; mean coverage per cluster: " +
                     ", ".join(f"c{c}={deg[lab.c_kmeans_norm.values==c].mean():.1f}" for c in range(6)))
    lines.append("")

    # ---------- 3.16 ablation configurations ----------
    manifest["ablation_configs"] = {
        "none": [], "G_spatial": ["spatial_rook"], "H_func": ["func_hard_k6"], "H_fac": ["fac_combined"],
        "G_spatial+H_func": ["spatial_rook", "func_hard_k6"], "G_spatial+H_fac": ["spatial_rook", "fac_combined"],
        "G_spatial+H_func+H_fac": ["spatial_rook", "func_hard_k6", "fac_combined"],
        "variants": {"spatial": list(spatial), "functional": list(func), "facility": types_present + ["combined"] + [f"group_{g}" for g in DEMO_WEIGHTS]},
    }
    (REL / "relations_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (REL / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
