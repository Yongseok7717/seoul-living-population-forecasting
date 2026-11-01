"""Appendix panels for exp27 (1000-epoch runs of M2 without early stopping; weight decay 0, 1e-4, 0.1), in the fixed layout of the
other Appendix D panels: F13a_long_loss (training and validation Huber loss) and F13b_long_wmape (validation wMAPE)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; LOGS = ROOT / "results" / "exp27_long_training" / "logs"; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "savefig.bbox": None, "figure.dpi": 110, "legend.frameon": False, "pdf.fonttype": 42})
RUNS = [("wd0", "weight decay 0", "#1f5fa8"), ("wd1e-4", "$10^{-4}$", "#2e8b57"), ("wd0.1", "0.1", "#c4572b")]


def history(key):
    h = []
    for line in (LOGS / f"exp27_m2_1000ep_{key}.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith('{"epoch"'):
            try:
                h.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return h


def smooth(y, w=9):
    y = np.asarray(y, float); k = np.ones(w) / w
    pad = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def save(fig, name):
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.30); fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=400); plt.close(fig); print("wrote", name)


H = {k: history(k) for k, _, _ in RUNS}
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for k, lab, c in RUNS:
    e = [x["epoch"] for x in H[k]]
    ax.plot(e, smooth([x["train_loss"] for x in H[k]]), color=c, lw=1.3, alpha=0.5)
    ax.plot(e, smooth([x["val_loss"] for x in H[k]]), color=c, lw=1.3, label=lab)
ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss"); ax.set_ylim(0.15, 0.45)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save(fig, "F13a_long_loss")

fig, ax = plt.subplots(figsize=(4.6, 3.8))
for k, lab, c in RUNS:
    e = [x["epoch"] for x in H[k]]; v = [100 * x["val_wmape"] for x in H[k]]
    ax.plot(e, smooth(v), color=c, lw=1.3, label=lab)
    b = int(np.argmin(v)); ax.plot([e[b]], [v[b]], "o", color=c, ms=5)
    print(k, "epochs", len(e), "best", round(v[b], 2), "@", e[b], "final", round(v[-1], 2))
ax.set_xlabel("epoch"); ax.set_ylabel("validation wMAPE (%)"); ax.set_ylim(4.6, 5.8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
save(fig, "F13b_long_wmape")
