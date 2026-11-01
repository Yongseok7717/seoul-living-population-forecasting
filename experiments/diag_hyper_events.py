"""Two checks that close the logic of the propagation argument.

(1) Propagation diagnostic (dR2, smooth share) for the FACILITY hypergraph (fac_combined: cells sharing a facility catchment, 12 types)
    and the TRANSIT hypergraph (station catchments + line areas), i.e. the operators used in the earlier hypergraph experiments,
    on all hours and restricted to weather-alert hours / q95 event cell-hours (2025 test, coefficients fitted on 2023).
(2) Event-rate test at facility cells: share of q95 event cell-hours (positive residual = crowding) at cells with an emergency room /
    hospital / shelter during alert windows versus the same cells in placebo windows (+-1..4 weeks), by alert group; the rally/crowd
    alerts serve as positive control.  This tests concentration at the cell level rather than the district-average level.
Outputs results/diag_propagation/{hyper.csv, facility_event_rate.csv}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "diag_propagation"; sys.path.insert(0, str(SRC)); sys.path.insert(0, str(Path(__file__).parent))
from train_moe import MoEData                 # noqa: E402
from residual_data import load_sparse, REL    # noqa: E402
from diag_propagation import design, r2       # noqa: E402
import config as C                            # noqa: E402
from loader import months_between             # noqa: E402

GROUPS = {"rain/flood": {"호우", "홍수", "산사태"}, "heat": {"폭염", "기타:폭염"}, "snow/cold": {"대설", "한파", "기타:결빙·한파"}, "fire": {"화재", "산불"}, "wind/dust": {"강풍", "미세먼지"}, "rally/crowd (control)": {"기타:집회·인파"}}
GU = {"11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구", "11410": "서대문구",
      "11440": "마포구", "11470": "양천구", "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구", "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구"}


def hyper_P(fname, key, cells):
    z = np.load(REL / fname, allow_pickle=True); Hm = load_sparse(REL / fname, key)[cells].tocsr(); w = z["w"] if "w" in z.files else np.ones(Hm.shape[1])
    de = np.asarray(Hm.sum(0)).ravel(); W = (Hm @ sp.diags(w / np.maximum(de, 1)) @ Hm.T).tocsr(); W.setdiag(0); W.eliminate_zeros(); W.data = np.abs(W.data)
    dsum = np.asarray(W.sum(1)).ravel(); return (sp.diags(1 / np.maximum(dsum, 1e-9)) @ W).tocsr().astype(np.float32)


def main():
    md = MoEData(72, 24, "q95"); d = md.d; N = d.N
    Z = {s: np.clip(d.RP[s] / d.r_std, -20, 20).astype(np.float32) for s in ("train", "test")}; E = {s: md.E[s].astype(bool) for s in ("train", "test")}   # RP = residual in people
    months = {"train": months_between("202303", "202312"), "test": months_between("202501", "202512")}; W = {}
    for s, ms in months.items():
        parts = []
        for m in ms:
            f = C.OUT / "priors" / f"prior_dmsg_{m}.npy"
            parts.append(np.load(f)[:, d.cells, 1] > 0 if f.exists() else np.zeros((len(pd.read_parquet(C.OUT_LOCAL / f"time_index_{m}.parquet")), N), bool))
        W[s] = np.concatenate(parts)
    # ---------------- (1) hypergraph operators
    rows = []
    for rel, P in (("facility hypergraph (12 types)", hyper_P("fac_combined.npz", "H_hard", d.cells)), ("transit hypergraph", hyper_P("trans_hyper.npz", "H", d.cells))):
        deg = np.asarray((P != 0).sum(1)).ravel(); has = deg > 0; M = {}
        for s in Z:
            z0 = np.nan_to_num(Z[s]); w = (~np.isnan(Z[s])).astype(np.float32); num = (P @ z0.T).T; den = (P @ w.T).T
            m = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32); m[:, ~has] = np.nan; M[s] = m; del z0, w, num, den
        for h in (1, 24):
            o, n, y = design(Z["train"], M["train"], h); r_own, b_own = r2(o, y); r_full, b_full = r2(np.concatenate([o, n], 1), y); del o, n, y
            T = Z["test"].shape[0]; t0 = max(2, 24 - h); ts = np.arange(t0, T - h); lagZ = lambda k_: Z["test"][ts + k_]; lagM = lambda k_: M["test"][ts + k_]; yy = Z["test"][ts + h]
            ok = ~(np.isnan(yy) | np.isnan(lagZ(0)) | np.isnan(lagZ(-1)) | np.isnan(lagZ(-2)) | np.isnan(lagZ(h - 24)) | np.isnan(lagM(0)) | np.isnan(lagM(-1)) | np.isnan(lagM(-2)) | np.isnan(lagM(h - 24)))
            for name, msk in {"all": np.ones_like(ok), "alert (weather) hours": W["test"][ts + h], "q95 event cell-hours": E["test"][ts + h]}.items():
                mm = ok & msk; o3 = np.stack([lagZ(0)[mm], lagZ(-1)[mm], lagZ(-2)[mm], lagZ(h - 24)[mm]], -1); n3 = np.stack([lagM(0)[mm], lagM(-1)[mm], lagM(-2)[mm], lagM(h - 24)[mm]], -1); y3 = yy[mm]
                t_own, _ = r2(o3, y3, b_own); t_full, _ = r2(np.concatenate([o3, n3], 1), y3, b_full); zc, mc = yy[mm], M["test"][ts + h][mm]
                rows.append({"relation": rel, "subset": name, "h": h, "mean_degree": float(deg[has].mean()), "corr_contemporaneous": float(np.corrcoef(zc, mc)[0, 1]), "smooth_energy_share": 1 - float(((zc - mc) ** 2).sum() / (zc ** 2).sum()), "dR2_train": r_full - r_own, "R2_own_test": t_own, "dR2_test": t_full - t_own})
                print(rows[-1], flush=True)
        del M
    pd.DataFrame(rows).to_csv(OUT / "hyper.csv", index=False, encoding="utf-8-sig")
    # ---------------- (2) event-rate test at facility cells (2025 test, positive events only = crowding)
    Rz = Z["test"]; Epos = E["test"] & (Rz > 0); T = Rz.shape[0]
    ti = pd.concat([pd.read_parquet(C.OUT_LOCAL / f"time_index_{m}.parquet") for m in months["test"]]); tt = pd.to_datetime(ti.datetime).values; assert len(tt) == T
    dmap = pd.read_parquet(C.OUT / "grid" / "cell_dong_map.parquet").sort_values("share").groupby("cell_idx").last(); gu_all = dmap["H_DNG_CD"].astype(str).str[:5].map(GU).reindex(range(10125)).fillna("").values; gu_of = gu_all[d.cells]
    fc = pd.read_parquet(C.OUT / "facilities" / "facility_counts_by_cell.parquet").set_index("cell_idx").reindex(d.cells).fillna(0)
    classes = {"all cells": np.ones(N, bool), "emergency room": fc.n_emergency.values > 0, "hospital": fc.n_hospital.values > 0, "earthquake shelter": (fc.n_shelter_eq_indoor.values + fc.n_shelter_eq_outdoor.values) > 0, "civil-defence shelter": fc.n_shelter_civil_defense.values > 0}
    ev = pd.read_csv(C.TABLES / "disaster_events_seoul.csv", encoding="utf-8-sig"); ev["t0"] = pd.to_datetime(ev.first_ts).dt.floor("h"); ev["t1"] = pd.to_datetime(ev.last_ts) + pd.Timedelta(hours=12); out = []
    for e in ev.itertuples():
        grp = next((g for g, s in GROUPS.items() if e.type in s), None)
        if grp is None:
            continue
        h = np.where((tt >= np.datetime64(e.t0)) & (tt <= np.datetime64(e.t1)))[0]
        if len(h) == 0:
            continue
        cells = np.arange(N) if e.gu == "ALL" else np.where(np.isin(gu_of, str(e.gu).split(",")))[0]
        for cname, cm in classes.items():
            cc = cells[cm[cells]]
            if len(cc) == 0:
                continue
            r_ev = float(Epos[np.ix_(h, cc)].mean()); pl = [float(Epos[np.ix_(h + s_ * 24, cc)].mean()) for s_ in (-28, -21, -14, -7, 7, 14, 21, 28) if (h + s_ * 24).min() >= 0 and (h + s_ * 24).max() < T]
            if pl:
                out.append({"group": grp, "date": e.date, "gu": e.gu, "class": cname, "n_cells": len(cc), "rate_event": r_ev, "rate_placebo": float(np.mean(pl)), "ratio": r_ev / max(float(np.mean(pl)), 1e-6)})
    o = pd.DataFrame(out); g = o.groupby(["group", "class"]).agg(n_events=("date", "nunique"), rate_event=("rate_event", "mean"), rate_placebo=("rate_placebo", "mean"), ratio_median=("ratio", "median"), share_ratio_gt1=("ratio", lambda s: float((s > 1).mean()))).reset_index()
    g.to_csv(OUT / "facility_event_rate.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 220); print(g.round(4).to_string()); print("ALLDONE")


if __name__ == "__main__":
    main()
