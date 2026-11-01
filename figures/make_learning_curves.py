"""Learning curves of the three M2 runs (exp02, FiLM cluster x calendar, seeds 0-2) for Appendix B: per-epoch training and
validation Huber loss (F12a) and validation wMAPE (F12b); the annealing epochs after the early stop are drawn dashed.
Outputs results/figures/F12a_curve_loss.* and F12b_curve_wmape.* ."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "savefig.bbox": None, "figure.dpi": 110, "legend.frameon": False, "pdf.fonttype": 42})
CL = ["#1f5fa8", "#c4572b", "#2e8b57"]
runs = [(s, json.load(open(ROOT / "results" / "exp02_conditioning" / f"exp02_func_x_cal_s{s}.json"))) for s in range(3)]


def split(h, key, scale=1.0):
    ad = [(x["epoch"], x[key] * scale) for x in h if x["stage"] == "adamw"]
    an = [(x["epoch"], x[key] * scale) for x in h if x["stage"] == "anneal"]
    if an:  # join the dashed annealing segment to the last AdamW epoch
        an = [ad[-1]] + an
    return ad, an


def save(fig, name):
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.30); fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=400); plt.close(fig); print("wrote", name)


# (a) Huber losses
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for (s, r), c in zip(runs, CL):
    h = r["history"]
    for key, ls, lab in (("train_loss", "-", "training"), ("val_loss", "-", "validation")):
        ad, an = split(h, key)
        ax.plot(*zip(*ad), ls, color=c, lw=1.4, alpha=1.0 if key == "val_loss" else 0.55, label=lab if s == 0 else None)  # AdamW epochs only; one legend entry per curve type
ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss"); ax.set_ylim(0.15, 0.37)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "F12a_curve_loss")

# (b) validation wMAPE
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for (s, r), c in zip(runs, CL):
    h = r["history"]; ad, an = split(h, "val_wmape", 100.0)
    ax.plot(*zip(*ad), "-", color=c, lw=1.4, label=f"run {s+1}")
    if an:
        ax.plot(*zip(*an), "--", color=c, lw=1.4)
    b = r["best_epoch"]; ax.plot([b], [next(x["val_wmape"] * 100 for x in h if x["epoch"] == b)], "o", color=c, ms=5)
ax.set_xlabel("epoch"); ax.set_ylabel("validation wMAPE (%)"); ax.set_ylim(4.6, 5.6)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "F12b_curve_wmape")
print({s: (r["best_epoch"], r["epochs_run"], round(r["history"][-1]["val_wmape"] * 100, 2)) for s, r in runs})
