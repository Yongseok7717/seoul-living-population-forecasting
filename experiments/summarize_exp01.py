"""Summarise experiment 1 (rolling re-training): per-quarter table, mean over quarters, and a figure -> results/exp01_rolling/{summary.csv, summary_mean.csv, fig_exp01_rolling.png}"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "results" / "exp01_rolling"; FIG = Path(__file__).resolve().parents[1] / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
ROWS = [("M2 FiLM-MLP-3L", "fixed 2023", "roll_{q}_A5fixed", ("models", "model")), ("M2 FiLM-MLP-3L", "rolling", "roll_{q}_A5", ("models", "model")),
        ("X2 MoE", "fixed 2023", "roll_{q}_X2fixed", ("models", "soft")), ("X2 MoE", "rolling", "roll_{q}_X2", ("models", "soft"))]


def main():
    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        for model, kind, pat, key in ROWS:
            f = OUT / f"{pat.format(q=q)}.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            for k in key:
                m = m[k]
            t = m["test"]
            rows.append({"fold": q, "model": model, "kind": kind, "test_wMAPE": t["X_wmape"] * 100, "MAE_N": t["MAE_N"], "MAE_E": t["MAE_E"], "top1": t["R_rel_err_top1pct"], "recall": t["extreme_recall_top1"], "F1": t["extreme_f1_top1"], "peak_mag": t["peak_magnitude_rel_err"], "jaccard": t["top5pct_spatial_jaccard"]})
    df = pd.DataFrame(rows); df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    mean = df.groupby(["model", "kind"])[["test_wMAPE", "MAE_N", "MAE_E", "top1", "recall", "F1", "peak_mag", "jaccard"]].mean().round(3); mean.to_csv(OUT / "summary_mean.csv", encoding="utf-8-sig")
    print(df.round(3).to_string()); print(); print(mean.to_string())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (col, lab) in enumerate([("test_wMAPE", "test wMAPE %"), ("recall", "extreme recall"), ("top1", "top-1% residual error")]):
        for (model, kind), d in df.groupby(["model", "kind"]):
            ax[i].plot(d.fold, d[col], marker="o", ls="-" if kind == "rolling" else "--", label=f"{model} ({kind})")
        ax[i].set_title(lab); ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.suptitle("Experiment 1: fixed 2023 model vs quarterly rolling re-training (test quarters of 2025)"); fig.tight_layout()
    fig.savefig(FIG / "fig_exp01_rolling.png", dpi=130); print("figure saved")


if __name__ == "__main__":
    main()
