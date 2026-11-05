# Fine-grained urban population forecasting in Seoul

Code for the paper *Fine-grained urban population forecasting in Seoul under temporal shift and local extremes*
(Yongseok Jang and Young-Suk Lee).

The hourly Seoul Living Population on the 250 m grid (7,807 analysis cells, 2023–2025) is forecast 24 hours ahead in a causal
baseline–residual framework: a causal same-hour median `B` over the previous 40 weekdays, 8 Saturdays or 8 Sundays/holidays (about eight weeks) removes the weekly cycle, every trained model forecasts the
normalized residual `z = clip((X − B)/s_i, −20, 20)` from a 72-hour history, and the forecast is `X̂ = B + s_i ẑ`.
The repository contains the data pipeline, the models, one script per experiment of the paper and the figure scripts.

**Data are not redistributed.** The 250 m grid Seoul Living Population data were provided by the Seoul Metropolitan Government
and cannot be shared publicly because of the provider's security restrictions. The auxiliary data are public (Seoul Open Data
Plaza, data.go.kr, safetydata.go.kr); `data_pipeline/` downloads or reads them and, given the living-population files, builds
every processed input.

## Layout

```
src/              shared configuration, data loaders and all models / trainers
  config.py            paths (LIVPOP_DATA, results/), constants of the data product
  loader.py            loaders for the processed grid and tensors
  build_baseline_atlas.py   calendar helpers and the population atlas used by the accessibility index
  residual_data.py     causal baseline B, residual R, scales s_i, splits, windows   (python src/residual_data.py builds them)
  data.py, models.py   dataset wrapper and relational operators
  train_residual.py    reference models R0-R4 (persistence, linear AR, GRU, GRU with graph/hypergraph propagation)
  train_func.py        conditioning study on the compact encoder (M1)
  train_cap.py         FiLM-MLP M2, Transformers M3/M3b, linear reference R7, masked experts; L-BFGS, long-training and interpolation modes
  train_moe.py         event labels, gate inputs, reference mixtures of experts and gate variants
  train_moe_f.py       mixtures assembled from frozen experts (X1-X3), zero-shot partners and the constant-weight blend B1 (--gate none)
data_pipeline/    raw downloads -> <LIVPOP_DATA>/processed
experiments/      base runs (base_*.py) and one script per experiment (exp*.py, diag_*.py); outputs in results/
figures/          figure scripts and the TikZ framework diagram
run_all.py        runs all experiments and figures in dependency order (resumable)
```

## Setup

Python 3.11, a CUDA GPU for the trainers (an RTX 4090 was used; M2 trains in about 70 minutes, a full reproduction takes several
GPU-days).

```
pip install -r requirements.txt
```

Paths are set by one environment variable: `LIVPOP_DATA` is the folder with the raw downloads, which also receives `processed/`
(default `./data`); see the docstring of `src/config.py` for the expected layout. Model outputs go to `./results`
(base runs in `results/base/{results,ckpt,logs}`, one folder per experiment otherwise, figures in `results/figures`).
API keys are needed only for the download scripts: copy `.env.example` to `.env` and fill in the keys.

## Data pipeline

Run from the repository root, in this order (each script documents its inputs and outputs):

| step | script | builds |
|---|---|---|
| 1 | `data_pipeline/fetch_seoul.py`, `fetch_datago_standard.py`, `fetch_emergency_api.py` | facility, rainfall and transit downloads |
| 2 | `data_pipeline/geocode_addresses.py`, `prepare_geocode_retry.py` | facility coordinates (VWorld geocoder) |
| 3 | `data_pipeline/fetch_disaster_msgs.py`, `build_disaster_events.py` | emergency-alert messages and alert events |
| 4 | `data_pipeline/build_venue_events.py` | schedule of football, baseball, concert, rally and festival events at venues |
| 5 | `data_pipeline/build_grid.py` | standard 250 m grid, adjacency, cell–district map |
| 6 | `data_pipeline/build_local_tensors.py` | monthly hourly tensors of the population aged 10 and over, disclosure regimes |
| 7 | `data_pipeline/qc_report.py` | daily quality control and the excluded anomalous days |
| 8 | `python src/residual_data.py` | causal baseline, residuals and per-cell scales |
| 9 | `python src/build_baseline_atlas.py`, `data_pipeline/build_anomaly_scale.py` | population atlas; robust scales of the episode catalog |
| 10 | `data_pipeline/weekly_profiles.py`, `functional_clusters.py` (both use `canonical.py`) | normalized weekly profiles; six functional clusters from the 2023 profiles |
| 11 | `data_pipeline/build_facilities.py`, `build_accessibility.py` | facilities on the grid, accessibility index |
| 12 | `data_pipeline/build_relations.py`, `build_mobility.py` | rook, functional kNN, lag-correlation and subway graphs; facility and transit hypergraphs |
| 13 | `data_pipeline/build_hazard.py`, `build_priors.py`, `build_disaster_prior.py` | hourly rainfall and the gate prior channels |
| 14 | `data_pipeline/anomaly_events.py`, `match_disaster_msgs.py`, `validate_events_2025.py` | catalog of anomalous episodes and its validation |

## Experiments and the paper

After the data pipeline, `python run_all.py` runs everything below in dependency order; every step skips runs whose results exist,
so it can be interrupted and resumed. Individual scripts can also be run directly from the repository root.

| paper | script(s) |
|---|---|
| Table 3: R0–R4 | `experiments/base_references.py` |
| Table 3: R7 (fully linear model, `--arch lin:0`); Table 5: rolling B1 with constant weights; Appendix F, bootstrap intervals | `experiments/exp28_revision.py` |
| Table 3: M1; Section 5.4, conditioning of the compact encoder | `experiments/base_conditioning_m1.py` |
| Table 3: M2 (three runs), M3; Appendix D: capacity ladder, optimizer checks, interpolation | `experiments/base_capacity.py` |
| Table 3: M3b | `experiments/exp07_patchtst.py` |
| reference mixture; Section 5.5 gate priors; Appendix B, gate variants | `experiments/base_gate_variants.py` |
| extreme expert used by X1–X3 and the rolling runs; Section 5.5, alternative expert training (F0–F3) | `experiments/base_mixture.py` |
| Table 3: X1–X3 (three runs); Section 5.5, refined event labels | `experiments/exp04_event_definition.py`, `exp13_moe_seeds.py` |
| Section 5.5, emergency-alert gate channels | `experiments/exp05_dmsg_gate.py` |
| Table 3: R5, R6, B1; Table 5: R6 and B1 rows | `experiments/exp17_ctx72.py` |
| Table 4 | `experiments/exp02_conditioning.py`, `exp03_relations.py` |
| Table 5: fixed, rolling and recency-control rows | `experiments/exp01_rolling.py`, `summarize_exp01.py`, `exp11_recency.py` |
| Section 5.6, growth cells and bootstrap of the rolling gain | `experiments/exp23_rolling_robustness.py` |
| Section 5.6, continuation of deviations under way | `experiments/diag_surge_retention.py` |
| Table 6, Fig. 6 | `experiments/exp18_venue.py`, `exp19_event_profile.py`, `exp20_chronos_analog.py`, `exp21_event_blend.py`, `exp26_rule_validation.py`; `figures/make_venue_figure.py` |
| Table 7 | `experiments/exp06_pressure.py`, `exp22_pressure_b1.py` |
| Section 5.3, Appendix C, Table C.1 | `experiments/diag_propagation.py`, `diag_level_smooth.py`, `diag_hyper_events.py`, `diag_propagation_events.py`, `exp25_lag24.py` |
| Appendix D, Fig. D.2 (1,000-epoch runs) | `experiments/exp27_long_training.py {wd0,wd1e-4,wd0.1}`; `figures/make_long_training_panels.py` |
| Appendix E, Table E.1 | `data_pipeline/anomaly_events.py`, `match_disaster_msgs.py`, `validate_events_2025.py`; `experiments/diag_facility_alerts.py` |
| Fig. 2, Fig. 3, Fig. E.1 | `figures/make_data_figures.py` |
| Fig. 4 | `figures/framework.tex` (TikZ) |
| Fig. 5 | `figures/make_horizon_figure.py` |
| Fig. D.1, Fig. D.3 | `figures/make_learning_curves.py`, `figures/make_capacity_panels.py` |

Run names inside `results/base/results` follow the original study: `r_*` and `t6_*` reference models, `tf_*` conditioning study,
`t16c_*` capacity study (ladder A0–A5 of the code is L0–L5 of the paper, numbered there by parameter count; `t16c_B_A5_lr0.003_wd0` is M2, `t16c_S_A5_s1/s2` its other two runs, `t16c_T2_cal_nofb` is M3,
`t16c_E_E2_long` the extreme expert), `moe_*` reference mixtures and gate variants, `moef_*` assembled mixtures
(`moef_F4mix` holds the mixture-tuned gate).

Environment variables used by the experiment scripts when they call the trainers: `RD_SPLITS=train,val,test` (month ranges
`YYYYMM-YYYYMM`) selects the training, validation and test months of the rolling protocol; `LP_RESULTS`, `LP_CKPT` and `LP_CACHE`
redirect the result, checkpoint and expert-cache folders of a trainer.

## License

MIT (see `LICENSE`). The license covers the code only; the data remain under the terms of their providers.

## Citation

If you use this code, please cite the paper (reference to be added on publication).
