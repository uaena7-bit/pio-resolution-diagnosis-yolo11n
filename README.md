# Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection

This repository provides the reproducibility materials for the CSSP manuscript:

**Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals**

The materials support a protocol-oriented study of dense small-object detection. The focus is not a new detector architecture, but a reproducible workflow for leakage-controlled input-resolution evaluation, scale-density-counting-deployment diagnosis, and task-objective-driven resolution selection.

## Repository contents

This release includes:

- leakage-controlled split manifests and leakage-audit summaries;
- group definitions for scale- and density-stratified diagnosis;
- evaluation scripts and locked source data for main and supplementary figures/tables;
- bootstrap uncertainty outputs and counting-calibration materials;
- source data and scripts for Fig. 2, Fig. 3, and Fig. 4;
- supplementary source data for Tables S4-S17 and metadata/scripts for Fig. 5 and Figs. S1-S3;
- YOLO11n seed-repeatability source data across 800, 960, and 1280 pixels using seeds 42, 123, and 2024;
- YOLOv8n cross-detector robustness-check source data for Table S13;
- locked-weight YOLO11n deployment benchmark source data and raw repeat logs for Table S14;
- multi-objective resolution-selection source data and README for Table S15;
- paper-linked reproduction scripts for Tables S13-S17;
- raw traceability logs for the YOLOv8n timing recheck and the YOLO11n locked-weight deployment benchmark.

## Main source-data directories

- `split_manifest/`: fixed leakage-controlled train/validation/test split manifests.
- `group_definitions/`: scale and density group definitions used for diagnostic analysis.
- `audit_summaries/`: leakage-control and near-duplicate audit summaries.
- `figure_source_data/`: locked source data used for main-text figures.
- `supplementary_source_data/`: source data for supplementary tables, including Tables S4-S17.
- `supplementary_figures/`: supplementary figure files and related source data.
- `bootstrap/`: bootstrap uncertainty and delta-difference outputs.
- `counting_calibration/`: counting-threshold calibration and confidence-sweep outputs.
- `evaluation_scripts/`: reproduction scripts for evaluation, figure generation, and newly added Tables S13-S17.
- `raw_logs/`: traceability logs retained separately from cleaned table source data.

## Key final table source data

- `supplementary_source_data/TableS11_YOLO11n_seed_repeatability_three_resolutions.csv`
- `supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv`
- `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv`
- `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv`

Supporting README files are provided for Tables S13-S17 where additional interpretation notes are required.

## Interpretation notes

- YOLOv8n is used only as a cross-detector robustness check, not as a detector-ranking benchmark.
- Strict global-first diagnostic AP is an auxiliary contribution-oriented diagnostic measure and is not directly comparable with ordinary subgroup AP or standard global AP.
- Table S14 peak memory refers to validation/inference CUDA memory, not training peak VRAM.
- In Table S14, total latency is the median end-to-end time per image across formal repeats and may not equal the sum of independently computed stage medians.
- Table S15 identifies 960 pixels as a balanced knee-point candidate only under the specified multi-objective weighting scheme. It does not claim that 960 pixels is universally optimal.
- The 1280-pixel setting remains the accuracy-oriented setting, whereas 800 pixels remains the lowest-cost baseline.

## Data availability

The original PIO images are not redistributed in this repository. Users should obtain the raw dataset from the original dataset source. The materials here provide split manifests, audit summaries, source data, scripts, and traceability logs needed to inspect or reproduce the reported analyses after authorized access to the raw images.

## Citation

If you use these materials, please cite the manuscript and this repository release:

Song, Y. *Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals*. GitHub repository release v1.0.0. Accessed 28 May 2026.

## CSSP reproducibility materials update for deployment, sensitivity, and external stress testing

This repository includes the source data and scripts used for the revised deployment, multi-objective sensitivity, and external cross-domain stress-test materials:

- `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv`
- `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv`
- `supplementary_source_data/TableS16_multi_objective_weight_sensitivity.csv`
- `supplementary_source_data/TableS17_VisDrone_YOLO11n_cross_domain_resolution_demo.csv`
- `raw_logs/RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv`
- `audit_summaries/VisDrone_split_leakage_audit_summary.csv`
- `audit_summaries/VisDrone_cross_split_near_duplicate_candidates.csv`
- `evaluation_scripts/reproduce_TablesS14_S16_deployment_and_weight_sensitivity.py`
- `evaluation_scripts/prepare_TableS17_VisDrone2019_DET_for_YOLO.py`
- `evaluation_scripts/reproduce_TableS17_VisDrone_YOLO11n_3res_seed42.py`
- `evaluation_scripts/audit_TableS17_VisDrone2019_DET_split_leakage.py`

The VisDrone2019-DET experiment is provided as an external cross-domain protocol demonstration, not as a VisDrone state-of-the-art benchmark. The original PIO and VisDrone images are not redistributed.


## Current manuscript-to-repository map

This repository corresponds to release `v1.0.0` for the CSSP manuscript and the release points to commit `c2944e2`.

| Manuscript / Supplementary item | Repository location |
|---|---|
| Fixed PIO split manifest and split statistics | `split_manifest/` |
| PIO leakage audit and near-duplicate audit summaries | `audit_summaries/` |
| Scale and density group definitions | `group_definitions/` |
| Figure source data for Figs. 2–4 | `figure_source_data/` |
| Table S11 YOLO11n seed repeatability | `supplementary_source_data/TableS11_YOLO11n_seed_repeatability_three_resolutions.csv` |
| Tables S12a/S12b bootstrap delta and delta-difference analysis | `supplementary_source_data/TableS12_bootstrap_delta_difference.csv` |
| Table S13 YOLOv8n cross-detector check | `supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv` |
| Table S14 30-repeat YOLO11n deployment benchmark | `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv` and `raw_logs/RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv` |
| Table S15 multi-objective resolution selection | `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv` |
| Table S16 multi-objective weight sensitivity | `supplementary_source_data/TableS16_multi_objective_weight_sensitivity.csv` |
| Table S17 VisDrone2019-DET external cross-domain stress test | `supplementary_source_data/TableS17_VisDrone_YOLO11n_cross_domain_resolution_demo.csv` |
| VisDrone official-split auxiliary duplicate audit | `audit_summaries/VisDrone_split_leakage_audit_summary.csv` and related VisDrone audit CSV files |
| Reproduction and audit scripts | `evaluation_scripts/` |

### Online Resource map

| Online Resource | Description in the manuscript | Repository location |
|---|---|---|
| Online Resource 1 | Cross-split leakage audit, near-duplicate audit, component-level split diagnostics, and dataset split statistics | `audit_summaries/`, `split_manifest/`, `group_definitions/` |
| Online Resource 2 | Maximum-detection sensitivity, strict diagnostic AP, and TP/FN/FP summaries | `supplementary_source_data/TableS4*` to `TableS7*`, `supplementary_figures/` |
| Online Resource 3 | Counting calibration and confidence-threshold sensitivity | `counting_calibration/`, `supplementary_source_data/TableS8*` to `TableS10*` |
| Online Resource 4 | Seed repeatability, cross-detector robustness, deployment benchmarking, multi-objective selection, VisDrone stress test, and qualitative examples | `supplementary_source_data/TableS11*`, `TableS13*` to `TableS17*`, `raw_logs/`, `evaluation_scripts/`, `figure_source_data/` |
| Online Resource 5 | Bootstrap uncertainty and delta-difference analysis | `bootstrap/`, `supplementary_source_data/TableS12_bootstrap_delta_difference.csv` |

### Execution notes

The original PIO and VisDrone2019-DET images are not redistributed. Users should obtain the raw datasets from their original sources. Scripts that regenerate qualitative figures or rerun training/evaluation require local access to the corresponding raw images, labels, trained weights, and prediction-output files. The source CSV files in this repository are provided to inspect and reproduce the reported tables and figures where the required local data are available.
