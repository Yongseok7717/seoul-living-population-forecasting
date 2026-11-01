"""Run every experiment of the paper in dependency order, then draw the figures.

    python run_all.py                 # everything (several GPU-days; each step is resumable and skips finished runs)
    python run_all.py --from exp17    # start at the first step whose script name contains 'exp17'
    python run_all.py --list          # print the steps

Requires the processed data (data_pipeline/, see README) under LIVPOP_DATA.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    # base runs: reference models, conditioning study, mixtures and gates, capacity study, assembled mixtures
    "experiments/base_references.py", "experiments/base_conditioning_m1.py", "experiments/base_gate_variants.py",
    "experiments/base_capacity.py", "experiments/base_mixture.py",
    # ablations and mixtures on the final encoder M2
    "experiments/exp02_conditioning.py", "experiments/exp03_relations.py", "experiments/exp07_patchtst.py",
    "experiments/exp04_event_definition.py", "experiments/exp13_moe_seeds.py", "experiments/exp05_dmsg_gate.py",
    # training-data recency, zero-shot forecasts and the combined forecaster B1
    "experiments/exp01_rolling.py", "experiments/summarize_exp01.py", "experiments/exp11_recency.py", "experiments/exp17_ctx72.py",
    "experiments/exp23_rolling_robustness.py",
    # scheduled events at venues and accessibility-adjusted pressure
    "experiments/exp06_pressure.py", "experiments/exp18_venue.py", "experiments/exp19_event_profile.py", "experiments/exp20_chronos_analog.py",
    "experiments/exp21_event_blend.py", "experiments/exp26_rule_validation.py", "experiments/exp22_pressure_b1.py",
    # forecaster-independent diagnostics
    "experiments/diag_propagation.py", "experiments/diag_level_smooth.py", "experiments/diag_hyper_events.py",
    "experiments/diag_propagation_events.py", "experiments/exp25_lag24.py", "experiments/diag_facility_alerts.py",
    "experiments/diag_surge_retention.py",
    # long training without early stopping (Appendix D)
    "experiments/exp27_long_training.py wd0", "experiments/exp27_long_training.py wd1e-4", "experiments/exp27_long_training.py wd0.1",
    # linear reference R7, constant-weight rolling B1 and block-bootstrap intervals (Appendix F)
    "experiments/exp28_revision.py",
    # figures
    "figures/make_data_figures.py", "figures/make_horizon_figure.py", "figures/make_learning_curves.py", "figures/make_capacity_panels.py",
    "figures/make_long_training_panels.py", "figures/make_venue_figure.py",
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="start", default=None); ap.add_argument("--list", action="store_true"); a = ap.parse_args()
    steps = STEPS
    if a.start:
        k = next(i for i, s in enumerate(STEPS) if a.start in s); steps = STEPS[k:]
    for s in steps:
        if a.list:
            print(s); continue
        script, *args = s.split()
        print(time.strftime("%m-%d %H:%M"), "STEP", s, flush=True)
        rc = subprocess.call([sys.executable, str(ROOT / script), *args], cwd=str((ROOT / script).parent), env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if rc != 0:
            sys.exit(f"step failed: {s} (rc {rc})")


if __name__ == "__main__":
    main()
