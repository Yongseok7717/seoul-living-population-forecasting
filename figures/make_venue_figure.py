"""Figure F11 on B1: observed, baseline, B1 (rolling M2 + 0.1 R6), B1 + fitted event profile, B1 + analog-context R6 (weights from exp21)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "results" / "figures"; E21 = ROOT / "results" / "exp21_event_blend"
sys.path.insert(0, str(ROOT / "experiments"))
from exp18_venue import VEN_EN  # noqa: E402
import exp19_event_profile as e19  # noqa: E402
from exp20_chronos_analog import NBLK, H, MODEL  # noqa: E402

EXAMPLES = [("World Cup Stadium", "2025-02-22"), ("World Cup Stadium", "2025-08-31"), ("KSPO Dome", "2025-05-18"), ("KSPO Dome", "2025-05-23")]


def weights(W, combo, key, origin, names):
    r = W[(W.combo == combo) & (W.key == key) & (W.origin == origin)]
    if not len(r):
        r = W[(W.combo == combo) & (W.key == "pooled") & (W.origin == origin)]
    return [float(r[f"w_{nm}"].iloc[0]) for nm in names]


def main():
    import torch
    from chronos import BaseChronosPipeline
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cuda", torch_dtype=torch.bfloat16)
    dt, hidx, X, B, Rser, fc, vc, cpos = e19.load(); ev = e19.events_table(); P = e19.profiles(ev, dt, hidx, Rser, vc, cpos); W = pd.read_csv(E21 / "weights.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6)); axes = axes.ravel(); short = {"World Cup Stadium": "World Cup Stadium", "KSPO Dome": "KSPO Dome"}
    for k, (ax, (venue, date_s)) in enumerate(zip(axes, EXAMPLES)):
        ven_kr = [kk for kk, vv in VEN_EN.items() if vv == venue][0]; date = pd.Timestamp(date_s)
        e = next(ev.loc[[ev[(ev.venue == ven_kr) & (ev.date == date)].index[0]]].itertuples())
        o = date - pd.Timedelta(hours=1); i = int(fc["test"]["origin"][o]); cc = cpos[vc[ven_kr]["500"]].values
        hrs = [o + pd.Timedelta(hours=h + 1) for h in range(H)]; pos = hidx.reindex(hrs).values.astype(int); A = X[pos][:, cc].sum(1) / 1000; Bs = B[pos][:, cc].sum(1) / 1000
        Pm = P.get(e.Index); rel = np.array([int((h - (date + pd.Timedelta(hours=e.start_hour))) / pd.Timedelta(hours=1)) for h in hrs]); Pw = np.zeros((H, len(cc)), np.float32)
        cc1 = cpos[vc[ven_kr]["1000"]].values; pin = pd.Series(np.arange(len(cc1)), index=cc1)[cc].values
        if Pm is not None:
            ok = (rel >= e19.RMIN) & (rel <= e19.RMAX); Pw[ok] = np.nan_to_num(Pm[pin][:, rel[ok] - e19.RMIN].T)
        B1 = 0.9 * fc["test"]["M2r"][i][:, cc] + 0.1 * fc["test"]["R6"][i][:, cc]
        earlier = [q for q in ev[ev.key == e.key].itertuples() if q.date < date][-NBLK:]; ch = []
        for q in earlier:
            ch += list(pd.date_range(q.date - pd.Timedelta(days=1), q.date + pd.Timedelta(hours=23), freq="h"))
        ch += list(pd.date_range(date - pd.Timedelta(days=1), o, freq="h")); cpos_ = hidx.reindex(ch).values.astype(int)
        qf, _ = pipe.predict_quantiles(torch.tensor(np.nan_to_num(Rser[cpos_][:, cc].T), dtype=torch.float32), prediction_length=H, quantile_levels=[0.5]); ZA = qf[..., 0].float().cpu().numpy().T
        a, b = weights(W, "B1+P", e.key, "day-ahead", ["B1", "P"]); a2, c = weights(W, "B1+ZA", e.key, "day-ahead", ["B1", "ZA"])
        hh = np.arange(H)
        ax.plot(hh, A, "k-", lw=3.2, label="observed"); ax.plot(hh, Bs, color="0.55", ls="--", lw=2.4, label="baseline (R0)")
        ax.plot(hh, Bs + B1.sum(1) / 1000, color="tab:green", lw=2.4, label="B1")
        ax.plot(hh, Bs + (a * B1 + b * Pw).sum(1) / 1000, color="tab:orange", lw=2.4, label="EC (B1 + event profile)")
        ax.plot(hh, Bs + (a2 * B1 + c * ZA).sum(1) / 1000, color="tab:red", lw=2.4, ls="-.", label="ZA (B1 + analog context)")
        ax.set_title(f"{short[venue]}, {date.strftime('%d %b')}", fontsize=20); ax.set_xlabel("hour of the event day" if k >= 2 else "", fontsize=19); ax.tick_params(labelsize=17); ax.set_xlim(0, 23); ax.grid(alpha=0.3)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("population (thousands)", fontsize=19)
    hs, ls = axes[0].get_legend_handles_labels(); fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=15, frameon=False, handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / "F11_venue_profile.pdf"); fig.savefig(FIG / "F11_venue_profile.png", dpi=160); print("figure ok")


if __name__ == "__main__":
    main()
