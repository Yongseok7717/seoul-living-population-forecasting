"""Appendix capacity figure, re-drawn in the same layout as the learning-curve panels (4.6 x 3.6 in, default fonts, two-row legend
below each panel so that the plot areas align).  Outputs results/figures/F06a_capacity.* and F06b_loss_gap.*."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results" / "base" / "results"; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "savefig.bbox": None, "figure.dpi": 110, "legend.frameon": False, "pdf.fonttype": 42})
C_REF, C_MAIN, C_MOE, C_TFM = "#8c8c8c", "#1f5fa8", "#c4572b", "#7a4fa3"
cap = pd.read_csv(RES / "task16_capacity.csv"); tf = pd.read_csv(RES / "task16_transformer.csv"); a = cap[cap.stage == "A"].sort_values("parameters")


def save(fig, name):
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.30); fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=400); plt.close(fig); print("wrote", name)


fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.plot(a.parameters, a.test_wMAPE, marker="o", color=C_MAIN, label="FiLM-MLP, test")
ax.plot(a.parameters, a.val_wMAPE, marker="o", color=C_MAIN, ls="--", alpha=0.6, label="FiLM-MLP, validation")
ax.set_xscale("log"); ax.set_xticks([1e5, 2e5, 4e5]); ax.set_xticklabels(["100k", "200k", "400k"]); ax.minorticks_off()
ax.set_xlabel("parameters"); ax.set_ylabel("wMAPE (%)"); ax.set_ylim(4.65, 5.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "F06a_capacity")

x = np.arange(len(a)); fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.plot(x, a.train_loss_eval, marker="o", color=C_REF, label="training")
ax.plot(x, a.val_loss, marker="o", color=C_MAIN, label="validation")
ax.plot(x, a.test_loss, marker="o", color=C_MOE, label="test")
ax.set_xticks(x); ax.set_xticklabels([f"L{j}" for j in range(len(x))]);   # ladder L0-L5 numbered by parameter count ax.set_xlabel("encoder (capacity ladder)"); ax.set_ylabel("Huber loss"); ax.set_ylim(0.15, 0.45)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save(fig, "F06b_loss_gap")
