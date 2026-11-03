"""Figure 5 (F05b_horizon_learned): test wMAPE by horizon, fixed protocol, from the runs that underlie Table 3:
M2 = exp02 func_x_cal seed 0 (the run B1 is built on), M3 = t16c_T2_cal_nofb, X2/X3 = exp13 seed 2 (soft / oracle),
R6 and B1 = exp17_std_residual (72-hour context; extreme_only / blend_val).  Same style as make_paper_figures.f05."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results" / "base" / "results"; PUB = ROOT / "results"; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "savefig.bbox": "tight", "figure.dpi": 110, "legend.frameon": False, "pdf.fonttype": 42})
C_MAIN, C_MOE, C_ORA, C_TFM, C_BLD = "#1f5fa8", "#c4572b", "#2e8b57", "#7a4fa3", "#b8860b"


def load(path, *keys):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in keys:
        r = r[k]
    return r


curves = [
    ("M2 FiLM-MLP-3L", load(PUB / "exp02_conditioning" / "exp02_func_x_cal_s0.json", "models", "model", "test"), C_MAIN, "o"),
    ("M3 Transformer", load(RES / "t16c_T2_cal_nofb.json", "models", "model", "test"), C_TFM, "o"),
    ("X2 MoE, tuned gate", load(PUB / "exp13_moe_seeds" / "exp13_s2_mix.json", "models", "soft", "test"), C_MOE, "o"),
    ("X3 oracle mixture", load(PUB / "exp13_moe_seeds" / "exp13_s2_mix.json", "models", "oracle", "test"), C_ORA, "o"),
    ("R6 Chronos zero-shot, residual", load(PUB / "exp17_ctx72" / "exp17_std_residual.json", "models", "extreme_only", "test"), "#8c8c8c", "P"),
    ("B1 blend M2 + R6", load(PUB / "exp17_ctx72" / "exp17_std_residual.json", "models", "blend_val", "test"), C_BLD, "X"),
]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for lab, tt, c, mk in curves:
    h = tt["X_wmape_h"]; ks = sorted(h, key=int)
    ax.plot([int(q) for q in ks], [100 * h[q] for q in ks], marker=mk, ms=6, lw=1.8, color=c, label=lab)
    print(f"{lab:32s} pooled {100*tt['X_wmape']:.2f}  h1 {100*h['1']:.2f}  h24 {100*h['24']:.2f}")
ax.set_xscale("log"); ax.set_xticks([1, 3, 6, 12, 24]); ax.set_xticklabels([1, 3, 6, 12, 24]); ax.minorticks_off()
ax.set_xlabel("forecast horizon (h)", fontsize=16); ax.set_ylabel("test wMAPE (%)", fontsize=16); ax.tick_params(labelsize=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=15, frameon=False)
fig.savefig(OUT / "F05b_horizon_learned.pdf"); fig.savefig(OUT / "F05b_horizon_learned.png", dpi=400); print("wrote F05b_horizon_learned")
