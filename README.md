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
- supplementary source data for Tables S4-S15 and metadata/scripts for Fig. 5 and Figs. S1-S3;
- YOLO11n seed-repeatability source data across 800, 960, and 1280 pixels using seeds 42, 123, and 2024;
- YOLOv8n cross-detector robustness-check source data for Table S13;
- locked-weight YOLO11n deployment benchmark source data and raw repeat logs for Table S14;
- multi-objective resolution-selection source data and README for Table S15;
- paper-linked reproduction scripts for Tables S13-S15;
- raw traceability logs for the YOLOv8n timing recheck and the YOLO11n locked-weight deployment benchmark.

## Main source-data directories

- `split_manifest/`: fixed leakage-controlled train/validation/test split manifests.
- `group_definitions/`: scale and density group definitions used for diagnostic analysis.
- `audit_summaries/`: leakage-control and near-duplicate audit summaries.
- `figure_source_data/`: locked source data used for main-text figures.
- `supplementary_source_data/`: source data for supplementary tables, including Tables S4-S15.
- `supplementary_figures/`: supplementary figure files and related source data.
- `bootstrap/`: bootstrap uncertainty and delta-difference outputs.
- `counting_calibration/`: counting-threshold calibration and confidence-sweep outputs.
- `evaluation_scripts/`: reproduction scripts for evaluation, figure generation, and newly added Tables S13-S15.
- `raw_logs/`: traceability logs retained separately from cleaned table source data.

## Key final table source data

- `supplementary_source_data/TableS11_YOLO11n_seed_repeatability_three_resolutions.csv`
- `supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv`
- `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv`
- `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv`

Supporting README files are provided for Tables S13-S15 where additional interpretation notes are required.

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

Song, Y. *Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals*. GitHub repository release v1.0.0. Accessed 26 May 2026.
