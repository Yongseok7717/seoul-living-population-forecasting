"""Experiment 28 - additions for the ESWA submission.

(a) Rolling B1 restricted to constant weights: the Q3 fold of exp17 selected a gate-proportional weight; re-run the Q3 assembly with
    --blend_set const so that every quarter uses the B1 definition of the paper ((1-w) M2 + w R6, w constant, selected on validation).
(b) Fully linear reference under the M2 protocol: one affine map from the 72-hour history (z, log-baseline), the future log-baseline and the
    future calendar to the 24 normalized residuals, Huber loss, AdamW one-cycle (peak 3e-3, no weight decay), <= 48 epochs, patience 7.
(c) Paired weekly block bootstrap (2,000 resamples of ISO weeks of the forecast origin):
    fixed protocol - M2 (three runs), M3, M3b, the linear reference, R6 and B1 on the 2025 test origins (every 6 h);
    rolling protocol - fixed, 2023-window, recent-window and rolling M2, quarter means with weeks resampled within each quarter.
Forecast dumps are aggregated to per-origin sums and deleted.  Outputs results/exp28_revision/*.csv
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; OUT = ROOT / "results" / "exp28_revision"; LOGS = OUT / "logs"
for d_ in (OUT, LOGS):
    d_.mkdir(parents=True, exist_ok=True)
ORIG_RES = ROOT / "results" / "base" / "results"; ORIG_CKPT = ROOT / "results" / "base" / "ckpt"
E17 = ROOT / "results" / "exp17_ctx72"; E07 = ROOT / "results" / "exp07_patchtst"; E11 = ROOT / "results" / "exp11_recency"; E23 = ROOT / "results" / "exp23_rolling_robustness"
ROLL_CKPT = ROOT / "results" / "exp01_rolling" / "ckpt"
FOLDS = {"Q1": ("202303-202410", "202411-202412", "202501-202503"), "Q2": ("202303-202501", "202502-202503", "202504-202506"),
         "Q3": ("202303-202504", "202505-202506", "202507-202509"), "Q4": ("202303-202507", "202508-202509", "202510-202512")}
REC = {"Q1": "202401-202410", "Q2": "202404-202501", "Q3": "202407-202504", "Q4": "202410-202507"}
OLD = "202303-202312"


def call(name, cmd, splits=None, env_extra=None):
    env = dict(os.environ, LP_RESULTS=str(OUT), LP_CKPT=str(OUT / "ckpt"), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    if splits:
        env["RD_SPLITS"] = ",".join(splits)
    env.update(env_extra or {}); print(time.strftime("%m-%d %H:%M"), "RUN", name, flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as lg:
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=str(SRC), env=env)
    print(time.strftime("%m-%d %H:%M"), "rc", rc, name, flush=True); return rc


# ---------------------------------------------------------------- (a)
def rolling_b1_const():
    name = "exp28_Q3_roll_residual_const"
    if not (OUT / f"{name}.json").exists():
        call(name, [sys.executable, str(SRC / "train_moe_f.py"), "--name", name, "--normal", "ckpt:roll_Q3_A5", "--extreme", f"npz:{E17 / 'chronos_Q3_roll_residual.npz'}",
                    "--gate", "ckpt:roll_Q3_X2_gate", "--blend_set", "const"], FOLDS["Q3"], {"LP_CKPT": str(ROLL_CKPT)})
    rows = []
    for q in FOLDS:
        f = OUT / f"{name}.json" if q == "Q3" else E17 / f"exp17_{q}_roll_residual.json"; r = json.loads(f.read_text(encoding="utf-8")); t = r["models"]["blend_val"]["test"]
        rows.append({"quarter": q, "blend": r["blend_desc"], "wMAPE": 100 * t["X_wmape"], "MAE_E": t["MAE_E"], "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"]})
    df = pd.DataFrame(rows); df.loc["mean"] = ["mean", "", *df[["wMAPE", "MAE_E", "top1", "recall"]].mean().values]
    df.to_csv(OUT / "rolling_b1_const.csv", index=False, encoding="utf-8-sig"); print(df.round(3).to_string(), flush=True)


# ---------------------------------------------------------------- dumps -> per-origin sums
def eval_flags(args):
    fl = ["--arch", args["arch"], "--cond", args.get("cond", "func_x_cal"), "--d_f", str(args.get("d_f", 8)), "--node_chunk", str(args.get("node_chunk", 500))]
    return fl + [f"--{k}" for k in ("step_cal", "no_future_b", "no_persist") if args.get(k)]


def agg_dump(f, tag):
    """per-origin absolute-error and population sums from a train_cap dump; deletes the dump"""
    z = np.load(f); X = z["X"]; Xh = np.maximum(z["Xhat"], 0); B = z["B"]; tm = pd.to_datetime(z["time"][:, 0].astype(str)) - pd.Timedelta(hours=1)
    d = pd.DataFrame({"origin": tm, "ae": np.abs(Xh - X).sum((1, 2)), "x": X.sum((1, 2))}); del z, X, Xh, B
    return d


def dump(tag, init, ckpt_dir, args, splits=None):
    agg = OUT / f"agg_{tag}.csv"
    if agg.exists():
        return pd.read_csv(agg, parse_dates=["origin"])
    f = OUT / f"d_{tag}_pred_test.npz"
    if not f.exists():
        call(f"d_{tag}", [sys.executable, str(SRC / "train_cap.py"), "--name", f"d_{tag}", "--mode", "eval", "--init", init, "--eval_stride", "6", "--dump_pred"] + eval_flags(args),
             splits, {"LP_CKPT": str(ckpt_dir)})
    d = agg_dump(f, tag); d.to_csv(agg, index=False); f.unlink(); return d


def args_of(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["args"]


# ---------------------------------------------------------------- (b) + fixed-protocol bootstrap
def fixed_protocol():
    lin = "exp28_linear"
    if not (OUT / f"{lin}.json").exists():
        call(lin, [sys.executable, str(SRC / "train_cap.py"), "--name", lin, "--arch", "lin:0", "--cond", "none", "--epochs", "48", "--patience", "7", "--lr", "3e-3", "--wd", "0", "--eval_stride", "6"])
    r = json.loads((OUT / f"{lin}.json").read_text(encoding="utf-8")); m = r["models"]["model"]
    print("linear", {s: round(100 * m[s]["X_wmape"], 3) for s in ("val", "test")}, "params", r.get("params"), flush=True)
    A = {}
    for i, run in enumerate(("t16c_B_A5_lr0.003_wd0", "t16c_S_A5_s1", "t16c_S_A5_s2")):
        A[f"M2_{i}"] = dump(f"M2_{i}", run, ORIG_CKPT, args_of(ORIG_RES / f"{run}.json"))
    A["M3"] = dump("M3", "t16c_T2_cal_nofb", ORIG_CKPT, args_of(ORIG_RES / "t16c_T2_cal_nofb.json"))
    A["M3b"] = dump("M3b", "exp07_P12", E07 / "ckpt", args_of(E07 / "exp07_P12.json"))
    A["LIN"] = dump("LIN", lin, OUT / "ckpt", args_of(OUT / f"{lin}.json"))
    # R6 and B1 from the M2 run used for B1 (run 0): need the per-cell forecasts -> one more dump of run 0 kept in memory
    agg_r6 = OUT / "agg_R6.csv"; agg_b1 = OUT / "agg_B1.csv"
    if not (agg_r6.exists() and agg_b1.exists()):
        f = OUT / "d_M2_0b_pred_test.npz"
        if not f.exists():
            call("d_M2_0b", [sys.executable, str(SRC / "train_cap.py"), "--name", "d_M2_0b", "--mode", "eval", "--init", "t16c_B_A5_lr0.003_wd0", "--eval_stride", "6", "--dump_pred"]
                 + eval_flags(args_of(ORIG_RES / "t16c_B_A5_lr0.003_wd0.json")), None, {"LP_CKPT": str(ORIG_CKPT)})
        z = np.load(f); X = z["X"]; Xm = z["Xhat"]; B = z["B"]; tm = pd.to_datetime(z["time"][:, 0].astype(str)) - pd.Timedelta(hours=1)
        R6 = np.load(E17 / "chronos_std_residual.npz")["test"].astype(np.float32); assert R6.shape == X.shape, (R6.shape, X.shape)
        Xr = np.maximum(B + R6, 0); Xb = np.maximum(0.9 * Xm + 0.1 * (B + R6), 0)
        pd.DataFrame({"origin": tm, "ae": np.abs(Xr - X).sum((1, 2)), "x": X.sum((1, 2))}).to_csv(agg_r6, index=False)
        pd.DataFrame({"origin": tm, "ae": np.abs(Xb - X).sum((1, 2)), "x": X.sum((1, 2))}).to_csv(agg_b1, index=False)
        del z, X, Xm, B, R6, Xr, Xb; f.unlink()
    A["R6"] = pd.read_csv(agg_r6, parse_dates=["origin"]); A["B1"] = pd.read_csv(agg_b1, parse_dates=["origin"])
    for k_, d in A.items():
        assert len(d) == len(A["M2_0"]) and (d.origin.values == A["M2_0"].origin.values).all(), k_
    wk = A["M2_0"].origin.dt.isocalendar(); wk = (wk.year * 100 + wk.week).values
    G = {k_: d.groupby(wk)[["ae", "x"]].sum() for k_, d in A.items()}; weeks = G["M2_0"].index.values
    def stat(sel):
        w = {k_: 100 * g.loc[sel, "ae"].sum() / g.loc[sel, "x"].sum() for k_, g in G.items()}; w["M2"] = np.mean([w["M2_0"], w["M2_1"], w["M2_2"]]); return w
    pt = stat(weeks); rng = np.random.default_rng(0); boots = [stat(rng.choice(weeks, len(weeks), replace=True)) for _ in range(2000)]
    comps = [("B1", "M2_0"), ("M3", "M2"), ("M3b", "M2"), ("LIN", "M2"), ("R6", "M2"), ("M2_1", "M2_0"), ("M2_2", "M2_0")]
    rows = [{"model": k_, "wMAPE": pt[k_], "ci_lo": np.percentile([b[k_] for b in boots], 2.5), "ci_hi": np.percentile([b[k_] for b in boots], 97.5)} for k_ in ["M2", "M2_0", "M2_1", "M2_2", "M3", "M3b", "LIN", "R6", "B1"]]
    rows += [{"model": f"{a} - {b}", "wMAPE": pt[a] - pt[b], "ci_lo": np.percentile([x[a] - x[b] for x in boots], 2.5), "ci_hi": np.percentile([x[a] - x[b] for x in boots], 97.5)} for a, b in comps]
    df = pd.DataFrame(rows); df["weeks"] = len(weeks); df.to_csv(OUT / "bootstrap_fixed.csv", index=False, encoding="utf-8-sig"); print(df.round(3).to_string(), flush=True)


# ---------------------------------------------------------------- rolling-protocol bootstrap
def rolling_protocol():
    A = {"fixed": [], "old": [], "recent": [], "rolling": []}
    a5 = args_of(ORIG_RES / "t16c_B_A5_lr0.003_wd0.json")
    for q, (tr, va, te) in FOLDS.items():
        for lab, src in (("fixed", f"r_{q}_fixed"), ("rolling", f"r_{q}_roll")):
            agg = OUT / f"agg_{src}.csv"
            if not agg.exists():
                agg_dump(E23 / f"{src}_pred_test.npz", src).to_csv(agg, index=False)      # exp23 dumps are kept (not deleted here)
            d = pd.read_csv(agg, parse_dates=["origin"]); d["q"] = q; A[lab].append(d)
        for lab, train in (("old", OLD), ("recent", REC[q])):
            d = dump(f"{lab}_{q}", f"{lab}_{q}_A5", E11 / "ckpt", a5, (train, va, te)); d["q"] = q; A[lab].append(d)
    A = {k_: pd.concat(v).reset_index(drop=True) for k_, v in A.items()}
    for k_, d in A.items():
        assert len(d) == len(A["rolling"]) and (d.origin.values == A["rolling"].origin.values).all(), k_
    wk = A["rolling"].origin.dt.isocalendar(); key = A["rolling"].q + "_" + (wk.year * 100 + wk.week).astype(str)
    G = {k_: d.groupby(key.values)[["ae", "x"]].sum() for k_, d in A.items()}; blocks = {q: [b for b in G["rolling"].index if b.startswith(q)] for q in FOLDS}
    def stat(sel):
        return {k_: np.mean([100 * g.loc[sel[q], "ae"].sum() / g.loc[sel[q], "x"].sum() for q in FOLDS]) for k_, g in G.items()}
    pt = stat(blocks); rng = np.random.default_rng(0)
    boots = [stat({q: rng.choice(b, len(b), replace=True) for q, b in blocks.items()}) for _ in range(2000)]
    comps = [("fixed", "rolling", "fixed - rolling"), ("old", "rolling", "2023 window - rolling"), ("old", "recent", "recency (2023 window - recent window)"), ("recent", "rolling", "more data (recent window - rolling)"), ("fixed", "old", "fixed - 2023 window")]
    rows = [{"quantity": k_, "value": pt[k_], "ci_lo": np.percentile([b[k_] for b in boots], 2.5), "ci_hi": np.percentile([b[k_] for b in boots], 97.5)} for k_ in A]
    rows += [{"quantity": lab, "value": pt[a] - pt[b], "ci_lo": np.percentile([x[a] - x[b] for x in boots], 2.5), "ci_hi": np.percentile([x[a] - x[b] for x in boots], 97.5)} for a, b, lab in comps]
    df = pd.DataFrame(rows); df["blocks"] = sum(len(b) for b in blocks.values()); df.to_csv(OUT / "bootstrap_rolling.csv", index=False, encoding="utf-8-sig"); print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a", "b", "c"]
    if "a" in parts:
        rolling_b1_const()
    if "b" in parts:
        fixed_protocol()
    if "c" in parts:
        rolling_protocol()
    print("ALLDONE", flush=True)
