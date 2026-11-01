"""Task 14-D/E: mobility-aware relations (training data only, no temporal leakage).

    python graphs/build_mobility.py [--lags 6] [--topk 10]

Writes processed/relations/
  mob_lag.npz     W[i,j] = max_tau corr(z_i(t), z_j(t+tau)), tau=1..L, top-k per row (i leads j), train split only  (inferred temporal dependency)
  mob_ext.npz     W[i,j] = max_tau P(E_j(t+tau)=1 | E_i(t)=1), q95 events on the train split, top-k per row
  trans.npz       W = K C K^T: cell-station kernel K(d)=exp(-(d/500)^2) (d<=1000 m), station connectivity C (same line: 2 nearest stations within 3 km; transfers: same name within 500 m), top-k per row
  trans_hyper.npz H: hyperedges = station catchments (cells within 600 m) and line service areas (cells within 600 m of any station of the line); w = 1
  mobility_summary.md
All matrices are on the full 10,125-cell index (zeros outside the 7,807-cell analysis subset), same npz layout as build_relations.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # shared config and loaders
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import config as C
from residual_data import ResidualData, REL   # noqa: E402

TRANS = C.OPEN_DIR / "교통"


def save_sparse(path, **mats):
    out = {}
    for key, M in mats.items():
        if sp.issparse(M):
            M = M.tocsr(); out.update({f"{key}__data": M.data, f"{key}__indices": M.indices, f"{key}__indptr": M.indptr, f"{key}__shape": np.array(M.shape)})
        else:
            out[key] = M
    np.savez_compressed(path, **out)


def topk_rows(A, k, thr):
    """keep the k largest entries per row (above thr, no diagonal) of a dense [n,n] array -> csr"""
    A = A.copy(); np.fill_diagonal(A, -np.inf)
    idx = np.argpartition(-A, k, axis=1)[:, :k]; rows = np.repeat(np.arange(A.shape[0]), k); vals = A[rows, idx.ravel()]
    keep = vals > thr
    return sp.csr_matrix((vals[keep], (rows[keep], idx.ravel()[keep])), shape=A.shape)


def lag_corr(Z, ok, lags, dev):
    """Z [T,N] standardized residuals (nan rows masked by ok); returns max over tau of corr(z_i(t), z_j(t+tau)) as [N,N]"""
    import torch
    Zt = torch.tensor(np.nan_to_num(Z), device=dev, dtype=torch.float32); best = None
    for tau in range(1, lags + 1):
        m = ok[:-tau] & ok[tau:]; A = Zt[:-tau][m]; B = Zt[tau:][m]
        A = (A - A.mean(0)) / (A.std(0) + 1e-6); B = (B - B.mean(0)) / (B.std(0) + 1e-6)
        Cm = (A.T @ B) / len(A)
        best = Cm if best is None else torch.maximum(best, Cm)
        print("lag", tau, "rows", int(m.sum()), "mean corr", float(Cm.mean()), flush=True)
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lags", type=int, default=6); ap.add_argument("--topk", type=int, default=10); ap.add_argument("--k", type=int, default=24); ap.add_argument("--local_km", type=float, default=3.0)
    args = ap.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = ResidualData(k=args.k, horizon=24); cells = d.cells; N, NA = d.N, len(d.grid); lines = []
    xy = d.grid[["CELL_X", "CELL_Y"]].values.astype(float); cxy = xy[cells]

    # ---------- inferred lag-dependency graph (train residuals) ----------
    Z = np.clip(d.RP["train"] / d.r_std, -20, 20); ok = ~np.isnan(d.RP["train"][:, 0])
    Cmax = lag_corr(Z, ok, args.lags, dev)
    W = topk_rows(Cmax, args.topk, 0.1)
    dist = np.hypot(cxy[W.nonzero()[0], 0] - cxy[W.nonzero()[1], 0], cxy[W.nonzero()[0], 1] - cxy[W.nonzero()[1], 1])
    lines += ["## mob_lag (inferred lag dependency, train 2023)", f"- edges {W.nnz}, mean corr {W.data.mean():.3f}, median edge distance {np.median(dist):.0f} m, share > 5 km {(dist > 5000).mean():.2f}, share adjacent (<=354 m) {(dist <= 354).mean():.2f}"]
    Wfull = sp.csr_matrix((W.data, (cells[W.nonzero()[0]], cells[W.nonzero()[1]])), shape=(NA, NA)); save_sparse(REL / "mob_lag.npz", W=Wfull)
    print(lines[-1], flush=True)
    # distance-limited variant: candidate pairs within --local_km (removes city-wide synchrony, keeps plausible movement dependencies)
    from scipy.spatial import cKDTree as _T
    near = _T(cxy).query_pairs(r=args.local_km * 1000.0, output_type="ndarray")
    M = np.full_like(Cmax, -np.inf); M[near[:, 0], near[:, 1]] = Cmax[near[:, 0], near[:, 1]]; M[near[:, 1], near[:, 0]] = Cmax[near[:, 1], near[:, 0]]
    WL = topk_rows(M, args.topk, 0.1); r, c = WL.nonzero(); dist = np.hypot(cxy[r, 0] - cxy[c, 0], cxy[r, 1] - cxy[c, 1])
    lines += [f"## mob_lag_local (same, candidates within {args.local_km} km)", f"- edges {WL.nnz}, mean corr {WL.data.mean():.3f}, median edge distance {np.median(dist):.0f} m, share adjacent (<=354 m) {(dist <= 354).mean():.2f}"]
    save_sparse(REL / "mob_lag_local.npz", W=sp.csr_matrix((WL.data, (cells[r], cells[c])), shape=(NA, NA))); print(lines[-1], flush=True)

    # ---------- extreme-transition graph ----------
    thr = float(np.nanquantile(np.abs(d.RP["train"][ok]), 0.95)); E = (np.abs(np.nan_to_num(d.RP["train"])) > thr) & ok[:, None]
    Et = torch.tensor(E.astype(np.float32), device=dev); cnt = Et.sum(0); best = None
    for tau in range(1, args.lags + 1):
        P = (Et[:-tau].T @ Et[tau:]) / torch.clamp(cnt[:, None], min=1)
        best = P if best is None else torch.maximum(best, P)
    Pm = best.cpu().numpy(); Pm[cnt.cpu().numpy() < 20] = 0
    base = E.mean(); WE = topk_rows(Pm, args.topk, 3 * base)         # lift >= 3 over the base event rate
    r, c = WE.nonzero(); dist = np.hypot(cxy[r, 0] - cxy[c, 0], cxy[r, 1] - cxy[c, 1])
    lines += ["## mob_ext (extreme transition P(E_j(t+tau) | E_i(t)), q95 train)", f"- threshold {thr:.0f} people, base rate {base:.4f}, edges {WE.nnz}, mean P {WE.data.mean():.3f}, median distance {np.median(dist):.0f} m, share > 5 km {(dist > 5000).mean():.2f}"]
    save_sparse(REL / "mob_ext.npz", W=sp.csr_matrix((WE.data, (cells[r], cells[c])), shape=(NA, NA))); print(lines[-1], flush=True)

    # ---------- transport graph and hypergraph (subway station master) ----------
    st = pd.read_csv(TRANS / "OA-21232_서울시역사마스터정보.csv", encoding="cp949")
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True); sx, sy = tr.transform(st["경도"].values, st["위도"].values); st["x"], st["y"] = sx, sy
    inside = (sx > xy[:, 0].min() - 1500) & (sx < xy[:, 0].max() + 1500) & (sy > xy[:, 1].min() - 1500) & (sy < xy[:, 1].max() + 1500)
    st = st[inside].reset_index(drop=True); S = len(st); sxy = st[["x", "y"]].values
    # cell-station kernel
    from scipy.spatial import cKDTree
    tree = cKDTree(sxy); pairs = tree.query_ball_point(cxy, r=1000.0)
    rows, cols, vals = [], [], []
    for i, js in enumerate(pairs):
        for j in js:
            dd = np.hypot(cxy[i, 0] - sxy[j, 0], cxy[i, 1] - sxy[j, 1]); rows.append(i); cols.append(j); vals.append(np.exp(-(dd / 500.0) ** 2))
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, S))
    # station connectivity: same line -> 2 nearest within 3 km; transfers: same name within 500 m
    Cr, Cc = [], []
    for line, g in st.groupby("호선"):
        idx = g.index.values; P = sxy[idx]
        if len(idx) < 2:
            continue
        t2 = cKDTree(P); dd, nn = t2.query(P, k=min(3, len(idx)))
        for a in range(len(idx)):
            for b, dist_ab in zip(nn[a][1:], dd[a][1:]):
                if dist_ab <= 3000:
                    Cr += [idx[a], idx[b]]; Cc += [idx[b], idx[a]]
    for name, g in st.groupby("역사명"):
        idx = g.index.values
        for a in idx:
            for b in idx:
                if a != b and np.hypot(sxy[a, 0] - sxy[b, 0], sxy[a, 1] - sxy[b, 1]) <= 500:
                    Cr.append(a); Cc.append(b)
    Cm = sp.csr_matrix((np.ones(len(Cr)), (Cr, Cc)), shape=(S, S)); Cm.data[:] = 1; Cm = Cm + sp.eye(S)
    At = (K @ Cm @ K.T).tocsr(); At.setdiag(0); At.eliminate_zeros()
    Ad = At.toarray(); WT = topk_rows(Ad, args.topk, 1e-3); r, c = WT.nonzero(); dist = np.hypot(cxy[r, 0] - cxy[c, 0], cxy[r, 1] - cxy[c, 1])
    lines += ["## trans (K C K^T, subway stations)", f"- stations in box {S} ({st['호선'].nunique()} lines), cells with a station within 1 km {int((K.getnnz(1) > 0).sum())}/{N}, station edges {Cm.nnz - S}, cell edges {WT.nnz}, median edge distance {np.median(dist):.0f} m, share > 2 km {(dist > 2000).mean():.2f}"]
    save_sparse(REL / "trans.npz", W=sp.csr_matrix((WT.data, (cells[r], cells[c])), shape=(NA, NA))); print(lines[-1], flush=True)
    # hypergraph: station catchments (600 m) + line service areas
    pairs6 = tree.query_ball_point(cxy, r=600.0); hr, hc = [], []
    for i, js in enumerate(pairs6):
        for j in js:
            hr.append(i); hc.append(j)
    Hs = sp.csr_matrix((np.ones(len(hr)), (hr, hc)), shape=(N, S))
    line_ids = {l: i for i, l in enumerate(sorted(st["호선"].unique()))}; Lm = sp.csr_matrix((np.ones(S), (np.arange(S), st["호선"].map(line_ids).values)), shape=(S, len(line_ids)))
    Hl = (Hs @ Lm).tocsr(); Hl.data[:] = 1
    keep_s = np.asarray(Hs.sum(0)).ravel() >= 2; Hs = Hs[:, keep_s]
    H = sp.hstack([Hs, Hl]).tocsr(); w = np.concatenate([np.ones(Hs.shape[1]), np.ones(Hl.shape[1])]).astype(np.float32)
    Hfull = sp.csr_matrix((H.data, (cells[H.nonzero()[0]], H.nonzero()[1])), shape=(NA, H.shape[1]))
    lines += ["## trans_hyper (station catchments 600 m + line service areas)", f"- hyperedges {Hs.shape[1]} catchments + {Hl.shape[1]} lines, mean catchment size {Hs.sum(0).mean():.1f} cells, cells covered {int((H.getnnz(1) > 0).sum())}/{N}"]
    save_sparse(REL / "trans_hyper.npz", H=Hfull, w=w); print(lines[-1], flush=True)
    (REL / "mobility_summary.md").write_text("\n".join(["# Mobility-aware relations (Task 14-D/E)", ""] + lines), encoding="utf-8"); print("saved", REL / "mobility_summary.md")


if __name__ == "__main__":
    main()
