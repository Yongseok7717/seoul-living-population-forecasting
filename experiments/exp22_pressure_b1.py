"""Experiment 22 - the pressure index scored on B1, R6 and the persistence reference R1, next to M2 and the baseline (exp06 dumps).

B1 rolling = 0.9 * M2 rolling + 0.1 * R6 (R6 = baseline + zero-shot residual from exp17, 72 h context, rolling folds);
R1 = baseline + residual observed at the 23:00 origin (persisted over the 24 target hours).
Day-ahead origins at 23:00 (exp06 windows); 2025 test quarters; metrics as in exp06.
Outputs results/exp22_pressure_b1/summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "exp22_pressure_b1"; OUT.mkdir(parents=True, exist_ok=True)
E06 = ROOT / "results" / "exp06_pressure"; E17 = ROOT / "results" / "exp17_ctx72"; E18 = ROOT / "results" / "exp18_venue"
sys.path.insert(0, str(ROOT / "experiments"))
import exp06_pressure as e6  # noqa: E402


def main():
    acc = pd.read_parquet(e6.PROJ / "processed" / "accessibility" / "accessibility_cells.parquet").set_index("cell_idx")
    per_hour = []
    for q in e6.FOLDS:
        z = np.load(E06 / f"p_{q}_roll_pred_test.npz"); W, H, N = z["X"].shape
        X, Xhat, B, E = z["X"], z["Xhat"], z["B"], z["E"]; tm = z["time"]; cells = z["cells"]
        origins = pd.to_datetime(tm[:, 0].astype(str)) - pd.Timedelta(hours=1)                                   # 23:00 of the previous day
        # R6 residual forecasts for the fold's 6-hourly origins (exp17), aligned by origin datetime (exp18 stores the origins)
        o18 = pd.to_datetime(np.load(E18 / f"fc_{q}.npz", allow_pickle=True)["origin"]); r6 = np.load(E17 / f"chronos_{q}_roll_residual.npz")["test"].astype(np.float32)
        assert len(o18) == len(r6), (len(o18), len(r6))
        idx = pd.Series(np.arange(len(o18)), index=o18).reindex(origins); ok = idx.notna().values
        R6 = np.full_like(Xhat, np.nan); R6[ok] = B[ok] + r6[idx.values[ok].astype(int)]
        # persistence: residual at the origin hour = last target hour of the previous window (windows are consecutive days)
        R1 = np.full_like(Xhat, np.nan)
        prev_ok = np.zeros(W, bool)
        for w in range(1, W):
            if pd.Timestamp(str(tm[w, 0])) - pd.Timestamp(str(tm[w - 1, 23])) == pd.Timedelta(hours=1):
                R1[w] = B[w] + (X[w - 1, 23] - B[w - 1, 23])[None, :]; prev_ok[w] = True
        B1 = 0.9 * Xhat + 0.1 * R6; Xf = np.load(E06 / f"p_{q}_fixed_pred_test.npz")["Xhat"]; assert Xf.shape == Xhat.shape
        sel = ok & prev_ok                                                                              # windows where every model is defined
        print(q, "windows", W, "scored", int(sel.sum()), flush=True)
        Xs, Bs, Es = (a[sel].reshape(-1, N) for a in (X, B, E)); tms = tm[sel].reshape(-1).astype(str); hz = np.tile(np.arange(1, H + 1), int(sel.sum())); share = Es.mean(1)
        models = {"baseline only": Bs, "R1 persistence": R1[sel].reshape(-1, N), "R6 zero-shot": R6[sel].reshape(-1, N), "M2 fixed": np.maximum(Xf[sel].reshape(-1, N), 0), "M2 rolling": np.maximum(Xhat[sel].reshape(-1, N), 0), "B1 rolling": np.maximum(B1[sel].reshape(-1, N), 0)}
        for s, col in e6.SERVICES.items():
            inv = (1.0 / (acc[col].reindex(cells).fillna(0).values + e6.EPS)).astype(np.float32)
            for lab, P in models.items():
                m = e6.hour_metrics(P.astype(np.float32), Xs, Bs, inv)
                per_hour.append(pd.DataFrame({"model": lab, "service": s, "quarter": q, "time": tms, "horizon": hz, "event_share": share, **m}))
    df = pd.concat(per_hour, ignore_index=True)
    ref = df[(df.model == "baseline only") & (df.service == "shelter")]; thr = ref.event_share.quantile(0.95); df["event_hour"] = df.time.isin(set(ref.time[ref.event_share >= thr]))

    def agg(g):
        return pd.Series({"hours": len(g), "pressure_wMAPE": 100 * g.abs_err.sum() / g.abs_true.sum(), "level_jaccard": g.jac_level.mean(), "surge_jaccard": g.jac_surge.mean(), "surge_precision": g.prec_surge.mean(), "excess_spearman": g.rho_excess.mean()})
    parts = [df.groupby(["service", "model"]).apply(agg).assign(scope="all hours"), df[df.event_hour].groupby(["service", "model"]).apply(agg).assign(scope="event hours")]
    summ = pd.concat(parts).reset_index().sort_values(["service", "scope", "model"]); summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string(), flush=True); print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
