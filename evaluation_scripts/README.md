# Evaluation Scripts README

This directory contains scripts used to support the CSSP manuscript:

Leakage-Controlled and Task-Objective-Dependent Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals

The scripts are provided for reproducibility and traceability. Some scripts regenerate figures or tables directly from locked CSV source data. Other scripts require local access to raw PIO or VisDrone2019-DET images, labels, model weights, or prediction-output files that are not redistributed in this repository.

## Environment

The experiments were conducted in a local Python environment with:

- Python 3.10
- PyTorch 2.x
- Ultralytics 8.x
- CUDA-enabled GPU execution where applicable
- numpy
- pandas
- matplotlib
- Pillow
- OpenCV
- scikit-image
- scipy
- PyYAML
- tqdm

The repository root provides `requirements.txt` with package versions from the local environment.

## Script inventory

The following scripts are provided in this directory:

- `audit_TableS17_VisDrone2019_DET_split_leakage.py`
- `create_pio_grdb_md5_split.py`
- `make_figure1_workflow.py`
- `make_figure2_accuracy_cost_tradeoff.py`
- `make_figure3_ordinary_scale_AP.py`
- `make_figure4_ordinary_density_AP.py`
- `make_figure5_qualitative_resolution_comparison.py`
- `make_figureS1_low_medium_no_gain_examples.py`
- `make_figureS2_strict_diagnostic_AP.py`
- `make_figureS3_validation_confidence_MAE_from_real_csv.py`
- `make_figureS4_continuous_density_diagnostic.py`
- `prepare_TableS17_VisDrone2019_DET_for_YOLO.py`
- `reproduce_TableS13_YOLOv8n_source_data.py`
- `reproduce_TableS13_YOLOv8n_train_eval_seed42.py`
- `reproduce_TableS14_YOLO11n_deployment_benchmark.py`
- `reproduce_TableS15_multi_objective_resolution_selection.py`
- `reproduce_TableS17_VisDrone_YOLO11n_3res_seed42.py`
- `reproduce_TablesS14_S16_deployment_and_weight_sensitivity.py`

## Manuscript and Online Resource mapping

| Manuscript or resource item | Typical script role | Raw images required |
|---|---|---|
| Online Resource 1 | Split construction, leakage audit, source-group audit, near-duplicate audit | Yes for full audit regeneration |
| Online Resource 2 | max_det sensitivity, strict diagnostic AP, TP/FN/FP summaries | Prediction outputs required; raw images may be needed for full regeneration |
| Online Resource 3 | Counting calibration and confidence-threshold sensitivity | Labels and prediction outputs required |
| Online Resource 4 | Seed repeatability, cross-detector check, deployment benchmark, multi-objective scoring, VisDrone demonstration | Mixed |
| Online Resource 5 | Bootstrap uncertainty and delta-difference analysis | AP samples or prediction-derived source data required |
| Main-text Figs. 2-4 | Plotting from locked CSV source data | No |
| Fig. 5 and qualitative supplementary examples | Qualitative visualization from image crops and predictions | Yes |

## CSV-only scripts

Scripts that regenerate plots, tables, score summaries, or bootstrap summaries from locked CSV files usually do not require raw PIO or VisDrone images.

Examples include:

- main-text figure plotting from `figure_source_data/`
- multi-objective scoring from `supplementary_source_data/TableS15*`
- weight-sensitivity analysis from `supplementary_source_data/TableS16*`
- deployment-summary analysis from `supplementary_source_data/TableS14*`
- bootstrap summaries from `bootstrap/` or `supplementary_source_data/TableS12*`

## Raw-data-dependent scripts

Scripts that regenerate training, inference, leakage audits from raw images, near-duplicate checks, VisDrone conversion, or qualitative overlays require local raw images and/or local prediction-output files.

Examples include:

- PIO split reconstruction from raw image metadata
- pHash/dHash and SSIM near-duplicate auditing
- YOLO training or validation
- deployment benchmarking from local model weights
- qualitative visualization using raw images and prediction CSVs
- VisDrone format conversion and auxiliary duplicate audit

## Suggested script header

Each script should begin with a short comment block such as:

Online Resource: 4
Manuscript item: Table S15, Table S16
Description: Multi-objective resolution selection and weight-sensitivity scoring.
Requires raw images: No
Inputs: supplementary_source_data/*.csv
Outputs: table source data or figure files

## Notes

Raw PIO and VisDrone2019-DET images are not redistributed in this repository. Ultralytics YOLO source code, pretrained weights, trained checkpoints, and prediction overlays are also not redistributed. Users should obtain raw datasets and third-party software from their official sources and comply with the relevant license terms.
