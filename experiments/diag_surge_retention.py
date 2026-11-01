"""Surge continuation diagnostics (Section 5.6): share of top-1 % targets already under way at the origin, error on the rest, and the
retained share of the surge magnitude on the under-way targets, for M2 (seed 0, fixed), R6 and R5 (72 h context).
Under way = |R| at the origin hour >= half the top-1 % threshold. Retention = sum(R_hat * sign(R)) / sum(|R|) over the under-way top-1 % targets (also reported: median of R_hat / R).
Output: results/diag_propagation/surge_retention.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"; RES = ROOT / "results"; sys.path.insert(0, str(SRC))
from train_moe import MoEData

md = MoEData(72, 24, "q95"); d = md.d; ts = d.windows("test", 6); H = 24
X = np.stack([d.X["test"][t + 1:t + 1 + H] for t in ts]); Bt = np.stack([d.B["test"][t + 1:t + 1 + H] for t in ts]); R = X - Bt
_dump = RES / "diag_surge_retention" / "m2std_pred_test.npz"
if not _dump.exists():                                                   # M2 (seed 0, fixed protocol) test forecasts in people
    import os, subprocess
    _dump.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LP_RESULTS=str(_dump.parent), LP_CKPT=str(ROOT / "results" / "base" / "ckpt"), PYTHONIOENCODING="utf-8"); env.pop("RD_SPLITS", None)
    subprocess.check_call([sys.executable, str(SRC / "train_cap.py"), "--name", "m2std", "--arch", "mlp:512,256,128", "--mode", "eval",
                           "--init", "t16c_B_A5_lr0.003_wd0", "--eval_stride", "6", "--dump_pred"], cwd=str(SRC), env=env)
m2 = np.load(_dump); RM = (m2["Xhat"].astype(np.float32) - m2["B"].astype(np.float32)); assert RM.shape == R.shape and np.allclose(m2["X"][:5], X[:5])
R6 = np.load(RES / "exp17_ctx72" / "chronos_std_residual.npz")["test"].astype(np.float32); R5 = np.load(RES / "exp17_ctx72" / "chronos_std_raw.npz")["test"].astype(np.float32)
absR = np.abs(R); thr = np.quantile(absR, 0.99); top = absR >= thr
R0 = np.stack([d.X["test"][t] - d.B["test"][t] for t in ts])[:, None, :]                        # last observed residual at the origin
under = np.zeros_like(top); under[:, :, :] = (np.abs(R0) >= thr / 2)
uw = top & under; nv = top & ~under; print("threshold", thr, "top-1 % targets", top.sum(), "under way share", uw.sum() / top.sum())
rows = []
for name, Rh in (("M2", RM), ("R6", R6), ("R5", R5)):
    ret = float((Rh[uw] * np.sign(R[uw])).sum() / absR[uw].sum()); med = float(np.median(Rh[uw] / R[uw]))
    err_uw = float(np.abs(Rh[uw] - R[uw]).sum() / absR[uw].sum()); err_nv = float(np.abs(Rh[nv] - R[nv]).sum() / absR[nv].sum())
    rows.append({"model": name, "retention_sum": ret, "retention_median_ratio": med, "rel_err_under_way": err_uw, "rel_err_not_visible": err_nv})
df = pd.DataFrame(rows); print(df.round(3).to_string()); (RES / "diag_propagation").mkdir(parents=True, exist_ok=True); df.to_csv(RES / "diag_propagation" / "surge_retention.csv", index=False)
