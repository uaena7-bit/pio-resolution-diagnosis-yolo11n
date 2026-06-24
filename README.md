# Leakage-Controlled and Task-Objective-Dependent Input-Resolution Diagnosis for Dense Small-Object Detection

This repository provides the reproducibility materials for the submitted Precision Agriculture manuscript:

**Leakage-Controlled and Task-Objective-Dependent Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Broiler-House Monitoring**

The contribution is a reproducible **decision framework** for leakage-controlled input-resolution evaluation, scale-density-counting-deployment diagnosis, and task-objective-dependent resolution selection. It is not a new detector architecture, and it does not claim that any single input size is universally optimal.

## Version cited by the manuscript

The submitted Precision Agriculture manuscript cites the archived reproducibility release:

- Repository: `uaena7-bit/pio-resolution-diagnosis-yolo11n`
- Release: `v1.1.0 - Precision Agriculture submission materials`
- Manuscript-linked commit: `<TO_BE_FILLED_AFTER_COMMIT>`

For exact reproduction of the submitted manuscript, use the archived release and commit cited above. The `main` branch may contain later documentation-only updates.

## What this repository supports

This release supports inspection or reproduction of the manuscript's reported analyses, including:

- leakage-controlled PIO split manifests and leakage-audit summaries;
- source-group / source-component diagnostics for split transparency;
- scale and density group definitions used for diagnostic AP analyses;
- source data and scripts for main-text figures and selected tables;
- bootstrap uncertainty outputs and delta-difference analyses;
- counting-threshold calibration and confidence-sweep source data;
- YOLO11n seed-repeatability source data across 800, 960, and 1280 px;
- YOLOv8n cross-detector robustness-check source data;
- 30-repeat locked-weight YOLO11n deployment benchmark source data and raw repeat logs;
- multi-objective resolution-selection and weight-sensitivity source data;
- VisDrone2019-DET directional protocol-transfer source data and duplicate-audit summaries.

## Repository contents

| Directory / file | Purpose |
|---|---|
| `split_manifest/` | Fixed leakage-controlled train/validation/test split manifests for the PIO case study. |
| `group_definitions/` | Scale-group and density-group definitions used for diagnostic analyses. |
| `audit_summaries/` | PIO leakage-control summaries, near-duplicate audit outputs, image-hash manifest, and VisDrone auxiliary audit outputs. |
| `figure_source_data/` | Locked source data for main-text figures. |
| `main_figures/` | Final main-text figure files used in the Precision Agriculture submission. |
| `figure_scripts/` | Python scripts used to generate manuscript figures. |
| `supplementary_source_data/` | CSV source data for supplementary tables and manuscript-linked traceability tables. |
| `supplementary_figures/` | Supplementary figure files and related source data. |
| `bootstrap/` | Bootstrap uncertainty outputs and delta-difference analysis files. |
| `counting_calibration/` | Counting-threshold calibration and confidence-sweep outputs. |
| `evaluation_scripts/` | Reproduction scripts for split construction, figure generation, table-source-data preparation, deployment benchmarking, and VisDrone conversion/audit. |
| `raw_logs/` | Traceability logs retained separately from cleaned table source data. |
| `CITATION.cff` | Repository citation metadata. |
| `LICENSE` | License for this repository's own source files, unless otherwise stated. |

## Key source-data files

The final manuscript no longer repeats the seed-repeatability table in the Supplementary Materials. The source data file with the historical `TableS11` name is retained only for traceability to the main manuscript seed-repeatability table.

| Manuscript item | Repository source data |
|---|---|
| Main manuscript seed-repeatability table | `supplementary_source_data/TableS11_YOLO11n_seed_repeatability_three_resolutions.csv` |
| Tables S12a/S12b bootstrap delta and delta-difference analysis | `supplementary_source_data/TableS12_bootstrap_delta_difference.csv` |
| Table S13 YOLOv8n cross-detector check | `supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv` |
| Table S14 30-repeat YOLO11n deployment benchmark | `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv` and `raw_logs/RawLog_TableS14_YOLO11n_deployment_recheck30_raw_repeats.csv` |
| Table S15 multi-objective resolution selection | `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv` |
| Table S16 multi-objective weight sensitivity | `supplementary_source_data/TableS16_multi_objective_weight_sensitivity.csv` |
| Table S17 VisDrone2019-DET directional protocol-transfer demonstration | `supplementary_source_data/TableS17_VisDrone_YOLO11n_cross_domain_resolution_demo.csv` |
| VisDrone official-split auxiliary duplicate audit | `audit_summaries/VisDrone_split_leakage_audit_summary.csv` and related `VisDrone_*` audit CSV files |

## Online Resource map

| Online Resource | Description in the manuscript | Repository location |
|---|---|---|
| Online Resource 1 | Cross-split leakage audit, near-duplicate audit, component-level split diagnostics, and dataset split statistics | `audit_summaries/`, `split_manifest/`, `group_definitions/` |
| Online Resource 2 | Maximum-detection sensitivity, strict diagnostic AP, and TP/FN/FP summaries | `supplementary_source_data/TableS4*` to `TableS7*`, `supplementary_figures/` |
| Online Resource 3 | Counting calibration and confidence-threshold sensitivity | `counting_calibration/`, `supplementary_source_data/TableS8*` to `TableS10*` |
| Online Resource 4 | Seed repeatability source data, cross-detector robustness, deployment benchmarking, multi-objective selection, weight sensitivity, VisDrone protocol-transfer demonstration, and qualitative examples | `supplementary_source_data/TableS11*`, `TableS13*` to `TableS17*`, `raw_logs/`, `evaluation_scripts/`, `figure_source_data/`, `main_figures/`, `figure_scripts/` |
| Online Resource 5 | Bootstrap uncertainty and delta-difference analysis | `bootstrap/`, `supplementary_source_data/TableS12_bootstrap_delta_difference.csv` |

## Interpretation notes

- The tested input resolutions are the discrete candidate set `{800, 960, 1280}`. The repository does not support a universal claim that 960 px is optimal.
- Under the manuscript's task interpretation, 1280 px is the accuracy-oriented setting for box-level detection and localization, 800 px is the lowest-cost baseline and can be preferable for scalar counting under validation-based calibration, and 960 px is selected only as a balanced candidate under the specified multi-objective weighting scheme.
- YOLOv8n is used only as a cross-detector robustness check, not as a detector-ranking benchmark.
- VisDrone2019-DET is used only as an external directional protocol-transfer demonstration, not as a VisDrone state-of-the-art benchmark.
- Strict global-first diagnostic AP is an auxiliary contribution-oriented diagnostic measure. It is not directly comparable with ordinary subgroup AP or standard global AP.
- Deployment timing and CUDA memory values are local-environment measurements. Table S14 peak memory refers to validation/inference CUDA memory, not training peak VRAM.
- In Table S14, total latency is the median end-to-end time per image across formal repeats and may not equal the sum of independently computed stage medians.

## External dependencies and model-license notice

This repository contains source data, audit summaries, and reproduction scripts produced for the manuscript. It does **not** redistribute third-party raw datasets, Ultralytics source code, pretrained detector weights, trained checkpoints, or prediction overlays.

The detector experiments were performed using YOLO-family detectors from Ultralytics, including YOLO11n as the main detector and YOLOv8n as a cross-detector check. Users who rerun training or inference should install Ultralytics from the official source and comply with the applicable Ultralytics license terms:

- Ultralytics YOLO documentation: https://docs.ultralytics.com/models/yolo11
- Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics
- Ultralytics license information: https://ultralytics.com/license

The Ultralytics package metadata identifies the package license as AGPL-3.0. Commercial or closed-source downstream use may require an appropriate Ultralytics license. This repository's MIT license applies only to the files authored and released in this repository, unless a file states otherwise; it does not relicense Ultralytics software/models or third-party datasets.

The experiments used a local Python environment with Python 3.10, PyTorch 2.x, Ultralytics 8.x, CUDA-enabled GPU execution where applicable, and common scientific Python packages such as pandas, numpy, matplotlib, Pillow, OpenCV/scikit-image utilities. Exact local package versions may vary by script; the manuscript reports the locked benchmark environment where needed.

## Raw data availability and redistribution policy

The original PIO and VisDrone2019-DET images are **not redistributed** in this repository. Users should obtain raw data from the original dataset sources and comply with each dataset's terms of use.

- PIO dataset: Boniche et al., *Scientific Data* 13, 801 (2026), https://doi.org/10.1038/s41597-026-07114-5
- VisDrone2019-DET dataset: official VisDrone dataset repository, https://github.com/VisDrone/VisDrone-Dataset

After obtaining the raw images from the official dataset sources, the materials in this repository provide the split manifests, source-group metadata, hash/audit summaries, grouped definitions, source data, and scripts needed to inspect or reproduce the reported analyses.

## Reproducibility workflow

A typical reproduction workflow is:

1. Check out the manuscript-linked release `v1.1.0` and the manuscript-linked commit reported above.
2. Install the Python environment required by the relevant scripts. See `evaluation_scripts/README.md` and each script header for details.
3. Obtain the raw PIO and, where needed, VisDrone2019-DET images from their original sources.
4. Arrange local image and label paths according to the split manifests and script arguments.
5. Use the locked CSV source data to regenerate paper figures/tables that do not require raw images.
6. Use the training/evaluation scripts only when local raw images, labels, model weights, and prediction-output files are available.

Some scripts regenerate plots from locked CSV source data. Other scripts regenerate training, inference, audit, or qualitative visualization outputs and therefore require local raw images and/or local prediction CSV files that are not redistributed here.

## Data and code availability statement

The original PIO dataset and the VisDrone2019-DET dataset should be obtained from their official sources. This repository provides the leakage-controlled PIO split manifest, source-group metadata, leakage-audit summaries, near-duplicate audit outputs, group definitions, bootstrap outputs, counting-calibration outputs, figure source data, supplementary table source data, evaluation scripts, deployment-benchmarking scripts, VisDrone conversion/audit scripts, and traceability logs used to support the manuscript. Raw PIO and VisDrone images, pretrained detector weights, trained checkpoints, and prediction overlays are not redistributed.

## Citation

If you use these materials, please cite the manuscript and this repository release.

Repository release citation:

Song, Y. Leakage-Controlled and Task-Objective-Dependent Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Broiler-House Monitoring. GitHub repository release v1.1.0 - Precision Agriculture submission materials. 2026.

Please also cite the original datasets and third-party software used in any reproduced experiments, including the PIO dataset paper, the VisDrone dataset paper/source, and Ultralytics YOLO where applicable.

## License

The repository's own scripts, source-data files, and documentation are released under the MIT License unless otherwise stated. Third-party datasets, Ultralytics software/models, and any locally downloaded detector weights are governed by their own licenses and terms of use.
