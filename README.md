# CPP_code

Code behind the benchmark of EEG models for predicting pain intensity and age. Outputs go to `../CPP_results/`.

**Environment:** conda env `otis_benchmarks_new`, with R `BayesFactor` 0.9.12.4.7. LaBraM uses env `labram`; OTiS `training/OTiS/envs/otis.yaml`.

**Paths:** by default the repository sits next to `data/` and the pretrained weights. Otherwise set `TURBO_ROOT`, `CPP_RESULTS`, `CPP_CHECKPOINTS`, `LABRAM_WEIGHTS`, `OTIS_WEIGHTS` (see `paths.sh`), and `CPP_PYTHON` for the Python environment. `WANDB_ENTITY` sets the Weights & Biases account. Cluster settings come from Slurm's own variables, for example `export SBATCH_QOS=<qos>` and `export SBATCH_PARTITION=<partition>`.

| Script | What it does | Trains? |
|---|---|---|
| `1_train_all_models.sh` | All 72 configurations, for pain intensity and age | Yes |
| `2_train_subgroups.sh` | Pain winners on chronic back pain (CBP) and neuropathic pain (NP) | Yes |
| `3_train_loso.sh` | The four winners, leave-one-study-out | Yes |
| `4_train_raw_targets.sh` | The four winners on pain intensity as rated and age in years | Yes |
| `5_results_manuscript.sh` | Summaries, tables and figures of the main analysis | No |
| `6_results_additional.sh` | Summaries and tables of the additional analyses | No |
| `7_export_results_light.sh` | Copies tables, per-model summaries, figures and CV tables to `CPP_results_light/` | No |
| `8_recreate_from_light.sh` | Recreates combined tables and tile figures from `CPP_results_light/`, compares the tables with the stored ones, and recomputes the overall and per-fold metrics | No |
| `9_misc_analyses.py` | Associations of pain intensity and harmonized global signal power with age (per fold, raw pooled and per dataset), written to `analysis_main/tables/age_associations/`. Run it before script 7 so that the tables are exported | No |

`CPP_results_light/` holds unchanged copies of the tables (including `age_associations/`), per-model summaries, figures and CV tables. Script 8 needs only this repository and the conda environment. The per-subject lists in the tables are sorted by subject ID, and the CV tables give each subject's fold.

Finished runs are skipped. `DRY_RUN=1` before any of scripts 1–4 prints the jobs instead of running them. The four winners are defined in `common.sh`: CPN linear and FBR Schaefer-100 2 s ridge (pain intensity), FBS sensor 2 s ridge and HCF sensor 2 s linear (age).

## Results layout

```
CPP_results/analysis_<name>/
├── inputs/      CV and normalisation tables
├── tables/      combined_*_tables[_small].csv; age_associations/ (main only, script 9)
├── figures/     main analysis only
└── <model>/
    ├── summaries/
    └── <config>/
        ├── features/       features or embeddings, or LOCATION.txt naming where they are stored
        └── predictions/    <algo>_<target>/ or full_<target>/
```

## Analyses

| Folder | Analysis |
|---|---|
| `analysis_main` | All models, 10 cross-validation folds |
| `analysis_CBP-subgroup`, `analysis_NP-subgroup` | Same folds and normalisation, restricted to the subgroup |
| `analysis_LOSO` | 8 folds, each holding out one dataset; targets and EEG power normalised within each raw dataset |
| `analysis_rawTargets` | Same as main, but the target is not harmonized; inputs stay harmonized |

Normalisation is defined in `training/shared/make_analysis_tables.py`. Final models are trained on `train` + `val`.

## Models

| Model | Training | Summary |
|---|---|---|
| OTiS_EEG, OTiS_GEN | `training/OTiS/execution_scripts/submit_{pretrain_s1,pretrain_s2,finetune}.sh`; final layer: `train_embedding_models.py` | `summarize_predictions_OTiS.py`, `summarize_predictions_final_layer_OTiS.py` |
| LaBraM | `training/LaBraM/submit_labram.sh`; final layer: `train_labram_embedding_models.py` | `summarize_predictions_LaBraM.py`, `summarize_predictions_final_layer_LaBraM.py` |
| CND, CNS (ConvNets) | `training/MEEGBenchmark/submit_convnets.sh` | `summarize_convnets.py` |
| HCF, FBR, FBS | `train` in `common.sh` -> `training/MEEGBenchmark/train_{hcf,fbr,fbs}.py` | `summarize_feature_models.py` |
| CPN | `train` in `common.sh` -> `training/CPN/train_cpn_models.py` | `summarize_predictions_CPN.py` |

`MEEGBenchmark` derives from Engemann et al. (2022), NeuroImage 262:119521. Dataset names follow Bott et al. (2025), eBioMedicine 120:105955; see `CPP_results/DATASET_ALIASES.md`.

## Not in this repository

- EEG signals, pretrained weights (`OTiS_weights/`, `labram-base.pth`) and `pytorch-image-models` (commit `779b8e214c95c1a672d0fffd0b25538d8bcf0fe7` plus `training/pytorch-image-models_local.diff`).
- Features stored outside `CPP_results`: a `features/` folder holding only `LOCATION.txt` (FBS, HCF, CPN, OTiS, LaBraM) names a folder relative to the project root, which `resolve_features` in `paths.py` follows. These features must be supplied separately.

## License

© 2026 Felix Bott. The code written for this project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0, full text in `LICENSE`).

**Citation:** if you use this code, cite `[CITATION PLACEHOLDER: benchmark manuscript, add DOI before publishing]`. This citation is the form of attribution requested under section 3(a)(1) of the license.

Third-party code is not covered by this license and keeps its own terms:

| Code | Source | Terms |
|---|---|---|
| `training/LaBraM/` | LaBraM | MIT, `training/LaBraM/LICENSE` |
| `training/MEEGBenchmark/` | Engemann et al. | BSD 3-Clause, `training/MEEGBenchmark/LICENSE` |
| `training/OTiS/` | OTiS | See the OTiS repository |
| `training/pytorch-image-models_local.diff` | pytorch-image-models | Apache-2.0 |
