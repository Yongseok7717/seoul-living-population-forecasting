"""Experiment 6 (T5) - from the 24 h forecast to infrastructure pressure.

Pressure of service set s in cell i at hour t:  R_{i,s}(t) = X_i(t) / (A_{i,s} + 0.05)   (A = accessibility, processed/accessibility)
Forecast pressure uses X_hat from M2 (FiLM-MLP-3L): fixed 2023 model and the quarterly rolling model (exp01 checkpoints), day-ahead
(origin 23:00, targets 00-23 h of the next day, non-overlapping windows), test = the four quarters of 2025.  Reference = baseline only (X_hat = B).

Metrics per target hour, averaged over (a) all hours, (b) event hours (share of q95 cells above its 95th percentile):
  pressure wMAPE;  level hotspot Jaccard (top 5 % cells by R);  surge hotspot Jaccard / precision (top 1 % cells by excess pressure
  dR = (X - B) / (A + eps));  Spearman rank correlation of dR across cells.
Outputs results/exp06_pressure/{summary.csv, summary.md, by_horizon.csv}, results/figures/fig_exp06_pressure.png
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp06_pressure"; LOGS = OUT / "logs"; FIG = ROOT / "results" / "figures"
for d in (OUT, LOGS, FIG):
    d.mkdir(parents=True, exist_ok=True)
ORIG_CKPT = ROOT / "results" / "base" / "ckpt"; ROLL_CKPT = ROOT / "results" / "exp01_rolling" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}
SERVICES = {"shelter": "A_shelter_all10", "health": "A_health_all10"}; EPS = 0.05


def dump(name, splits, ckpt_dir, init):
    f = OUT / f"{name}_pred_test.npz"
    if f.exists():
        return f
    env = dict(os.environ, RD_SPLITS=",".join(splits), LP_RESULTS=str(OUT), LP_CKPT=str(ckpt_dir), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(SRC / "train_cap.py"), "--name", name, "--arch", "mlp:512,256,128", "--mode", "eval", "--init", init, "--eval_stride", "24", "--dump_pred"]
    print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    if rc != 0 or not f.exists():
        print("FAILED", name, rc, flush=True); return None
    return f


def topk_mask(V, frac):
    k = max(1, int(round(V.shape[1] * frac))); thr = np.partition(V, -k, axis=1)[:, -k][:, None]
    return V >= thr


def jaccard(a, b):
    return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)


def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).astype(np.float32); rb = b.argsort(1).argsort(1).astype(np.float32)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def hour_metrics(Xhat, X, B, inv):
    """all inputs [T, N]; inv = 1 / (A + eps) [N].  Returns dict of per-hour arrays."""
    R, Rh = X * inv, Xhat * inv; dR, dRh = (X - B) * inv, (Xhat - B) * inv
    lev, levh = topk_mask(R, 0.05), topk_mask(Rh, 0.05); sur, surh = topk_mask(dR, 0.01), topk_mask(dRh, 0.01)
    return {"abs_err": np.abs(Rh - R).sum(1), "abs_true": R.sum(1), "jac_level": jaccard(lev, levh), "jac_surge": jaccard(sur, surh),
            "prec_surge": (sur & surh).sum(1) / np.maximum(surh.sum(1), 1), "rho_excess": spearman_rows(dR, dRh)}


def main():
    acc = pd.read_parquet(PROJ / "processed" / "accessibility" / "accessibility_cells.parquet").set_index("cell_idx")
    per_hour = []                                                           # rows: model, service, quarter, time, horizon, event share, metrics
    keep = {}                                                               # arrays kept for the figure
    for q, fold in FOLDS.items():
        fixed = ("202303-202312", "202411-202412", fold[2])
        files = {"M2 fixed 2023": dump(f"p_{q}_fixed", fixed, ORIG_CKPT, "t16c_B_A5_lr0.003_wd0"), "M2 rolling": dump(f"p_{q}_roll", fold, ROLL_CKPT, f"roll_{q}_A5")}
        for model, f in files.items():
            if f is None:
                continue
            z = np.load(f, allow_pickle=False); W, H, N = z["X"].shape
            X, Xhat, B, E = (z[k_].reshape(W * H, N) for k_ in ("X", "Xhat", "B", "E")); tm = z["time"].reshape(-1); hz = np.tile(np.arange(1, H + 1), W); cells = z["cells"]
            Xhat = np.maximum(Xhat, 0); share = E.mean(1)
            for s, col in SERVICES.items():
                inv = (1.0 / (acc[col].reindex(cells).fillna(0).values + EPS)).astype(np.float32)
                variants = [(model, Xhat)] + ([("baseline only", B)] if model == "M2 fixed 2023" else [])
                for lab, P in variants:
                    m = hour_metrics(P, X, B, inv)
                    per_hour.append(pd.DataFrame({"model": lab, "service": s, "quarter": q, "time": tm, "horizon": hz, "event_share": share, **m}))
            if model == "M2 rolling":
                keep[q] = (X, Xhat, B, tm, cells, share)
            del X, Xhat, B, E
    df = pd.concat(per_hour, ignore_index=True); ref = df[(df.model == "M2 fixed 2023") & (df.service == "shelter")]; thr = ref.event_share.quantile(0.95); df["event_hour"] = df.time.isin(set(ref.time[ref.event_share >= thr]))   # one common set of event hours (labels of the fixed 2023 split)

    def agg(g):
        return pd.Series({"hours": len(g), "pressure_wMAPE": 100 * g.abs_err.sum() / g.abs_true.sum(), "level_jaccard": g.jac_level.mean(), "surge_jaccard": g.jac_surge.mean(), "surge_precision": g.prec_surge.mean(), "excess_spearman": g.rho_excess.mean()})
    parts = [df.groupby(["service", "model"]).apply(agg).assign(scope="all hours"), df[df.event_hour].groupby(["service", "model"]).apply(agg).assign(scope="event hours")]
    summ = pd.concat(parts).reset_index().sort_values(["service", "scope", "model"]); summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    df["h_bin"] = pd.cut(df.horizon, [0, 6, 12, 18, 24], labels=["1-6", "7-12", "13-18", "19-24"])
    byh = df[df.service == "shelter"].groupby(["model", "h_bin"], observed=True).apply(agg).reset_index(); byh.to_csv(OUT / "by_horizon.csv", index=False, encoding="utf-8-sig")
    byq = df[df.service == "shelter"].groupby(["model", "quarter"]).apply(agg).reset_index(); byq.to_csv(OUT / "by_quarter.csv", index=False, encoding="utf-8-sig")
    md = ["# Experiment 6 - forecast to pressure (T5)", "", f"event hours: share of q95 cells >= {thr:.3f} (top 5 % of hours)", "", summ.round(3).to_markdown(index=False), "", "## shelter pressure by horizon", "", byh.round(3).to_markdown(index=False), "", "## shelter pressure by quarter", "", byq.round(3).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8"); print(summ.round(3).to_string(), flush=True)

    # ---- figure: strongest event hour (rolling model) - actual vs forecast excess shelter pressure + level-hotspot agreement by hour of day
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    sys.path.insert(0, str(SRC)); from loader import load_grid
    g = load_grid(); best = max(keep, key=lambda q_: keep[q_][5].max()); X, Xhat, B, tm, cells, share = keep[best]; i = int(share.argmax())
    inv = 1.0 / (acc["A_shelter_all10"].reindex(cells).fillna(0).values + EPS); dR, dRh = (X[i] - B[i]) * inv, (Xhat[i] - B[i]) * inv
    xy = g.loc[cells]; xc, yc = ["CELL_X"], ["CELL_Y"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5)); vmax = np.quantile(np.abs(dR), 0.995)
    if xc and yc:
        for a, v, ttl in ((ax[0], dR, "actual excess shelter pressure"), (ax[1], dRh, "forecast (M2 rolling, day-ahead)")):
            sc = a.scatter(xy[xc[0]], xy[yc[0]], c=v, s=2, cmap="RdBu_r", vmin=-vmax, vmax=vmax); a.set_aspect("equal"); a.set_axis_off(); a.set_title(f"{ttl}\n{tm[i]}"); fig.colorbar(sc, ax=a, shrink=.7)
    sh = df[df.service == "shelter"].copy(); sh["hod"] = pd.to_datetime(sh.time).dt.hour
    for lab, d in sh.groupby("model"):
        ax[2].plot(d.groupby("hod").jac_surge.mean(), marker="o", ms=3, label=lab)
    ax[2].set_xlabel("hour of day (target)"); ax[2].set_ylabel("surge hotspot Jaccard (top 1 %)"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8); ax[2].set_title("surge hotspot agreement, 2025")
    fig.tight_layout(); fig.savefig(FIG / "fig_exp06_pressure.png", dpi=130); print("figure saved"); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
