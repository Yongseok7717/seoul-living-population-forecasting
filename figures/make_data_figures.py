"""Data figures of the paper: daily city total with the disclosure regimes (Fig. 2), weekly profiles of the six functional clusters
(Fig. 3) and the catalog of anomalous episodes (Fig. E.1).  Vector PDF + 400 dpi PNG in results/figures/.
Inputs: processed/qc (data_pipeline/build_local_tensors.py, qc_report.py), processed/eda/cluster_profiles.npz (functional_clusters.py),
processed/eda/events (anomaly_events.py).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import config as C   # noqa: E402
PRO = C.OUT; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "savefig.bbox": "tight", "figure.dpi": 110, "legend.frameon": False, "pdf.fonttype": 42})
C_REF, C_MAIN, C_MOE = "#8c8c8c", "#1f5fa8", "#c4572b"
CL = ["#1f5fa8", "#c4572b", "#2e8b57", "#7a4fa3", "#d4a017", "#4c4c4c"]
WIDE = (8.8, 3.6)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=400); plt.close(fig); print("wrote", name)


def f01():
    qc = pd.read_csv(PRO / "qc" / "local_daily_qc.csv"); qc["d"] = pd.to_datetime(qc.ymd.astype(str)); qc = qc[qc.available.astype(str) == "True"].sort_values("d").drop_duplicates("d", keep="last")
    an = pd.read_csv(PRO / "qc" / "anomalous_days.csv"); an["d"] = pd.to_datetime(an.ymd.astype(str));
    qc = qc[(qc.d >= "2023-01-01") & (qc.d <= "2025-12-31") & ~qc.d.isin(an.d)]                  # days used in the study
    y = qc.set_index("d").pop10_total_daily_mean / 1e6; wk = y.rolling("7D", center=True).mean()
    fig, ax = plt.subplots(figsize=WIDE); ax.plot(y.index, y.values, lw=0.5, color=C_MAIN, alpha=0.35, label="daily mean"); ax.plot(wk.index, wk.values, lw=1.4, color=C_MAIN, label="7-day mean"); lo, hi = 9.6, 11.4
    # fixed protocol: Jan-Feb 2023 baseline initialisation, Mar-Dec 2023 training, 2024 validation, 2025 test
    for (a, b), lab, al in ((("2023-01-01", "2023-03-01"), "init.", 0.12), (("2023-03-01", "2024-01-01"), "training", 0.03),
                            (("2024-01-01", "2025-01-01"), "validation", 0.07), (("2025-01-01", "2026-01-01"), "test", 0.11)):
        a, b = pd.Timestamp(a), pd.Timestamp(b); ax.axvspan(a, b, color="#000000", alpha=al, lw=0)
        ax.text(a + (b - a) / 2, lo + 0.05, lab, ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set_ylim(lo, hi); ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-01"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylabel("population (million)"); ax.set_xlabel("date"); ax.legend(loc="upper left", ncol=2, fontsize=9); save(fig, "F01a_daily_total")


NAMES = {2857: "residential", 1920: "mixed", 1533: "weak daytime", 194: "weekend leisure", 872: "commercial", 431: "office"}


def f02():
    z = np.load(PRO / "eda" / "cluster_profiles.npz", allow_pickle=True)
    labels = z["labels"].astype(int); K = z["profiles"].shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k in range(K):
        ax.plot(np.arange(168) / 24, z["profiles"][k], color=CL[k % 6], lw=1.8, label=NAMES.get(int((labels == k).sum()), "C%d" % k))
    ax.set_xticks(np.arange(7) + 0.5); ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax.set_xlim(0, 7); ax.set_ylabel("normalized weekly profile", fontsize=14); ax.tick_params(labelsize=12); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13, frameon=False); save(fig, "F02b_cluster_profiles")


def f03():
    ev = pd.concat([pd.read_csv(PRO / "eda" / "events" / f, usecols=["date", "direction", "n_cellhours"]) for f in ("events_all_2023_2024.csv", "events_all_2025.csv")])
    ev["m"] = pd.to_datetime(ev.date).dt.to_period("M").dt.to_timestamp(); big = ev[ev.n_cellhours >= 20]
    fig, ax = plt.subplots(figsize=(5.4, 3.4)); piv = big.groupby(["m", "direction"]).size().unstack(fill_value=0); bottom = np.zeros(len(piv))
    for d_, c_ in zip(piv.columns, (C_MOE, C_MAIN, C_REF)):
        ax.bar(piv.index, piv[d_], width=25, bottom=bottom, color=c_, label=d_); bottom += piv[d_].values
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10))); ax.set_ylabel("episodes per month"); ax.set_xlabel("month"); ax.legend(loc="upper left"); save(fig, "F03a_events_monthly")
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.hist(np.log10(ev.n_cellhours.clip(lower=1)), bins=40, color=C_MAIN, alpha=0.85); ax.set_yscale("log"); ax.set_xlabel("episode size (log$_{10}$ cell-hours)"); ax.set_ylabel("number of episodes"); save(fig, "F03b_event_size")


if __name__ == "__main__":
    for fn in (f01, f02, f03):
        fn()
