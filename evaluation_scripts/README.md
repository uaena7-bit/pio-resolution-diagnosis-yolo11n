# Evaluation scripts

This directory contains paper-linked scripts used to prepare source data, audits, tables, and figures for the CSSP reproducibility release.

## Environment

The experiments were conducted in a local Python environment with:

- Python 3.10
- PyTorch 2.x
- Ultralytics 8.x
- CUDA-enabled GPU inference/training where applicable
- pandas, numpy, matplotlib, Pillow, OpenCV/scikit-image utilities as required by individual scripts

Exact local package versions may differ by script; the manuscript reports the locked benchmark environment where needed.

## Script inventory

| Script | Target |
|---|---|
| `make_Figure1_workflow.py` | Fig. 1 workflow diagram |
| `make_Figure2_accuracy_cost_tradeoff.py` | Fig. 2 accuracy-cost trade-off |
| `make_Figure3_scale_stratified_AP.py` | Fig. 3 scale-stratified ordinary AP |
| `make_Figure4_density_stratified_AP.py` | Fig. 4 density-stratified ordinary AP |
| `make_Figure5_qualitative_resolution_comparison.py` | Fig. 5 qualitative dense-scene comparison |
| `make_FigureS2_strict_diagnostic_AP.py` | Fig. S2 strict global-first diagnostic AP |
| `make_FigureS3_validation_confidence_MAE.py` | Fig. S3 validation confidence-MAE sweep |
| `reproduce_TableS13_YOLOv8n_source_data.py` | Table S13 source-data preparation |
| `reproduce_TableS13_YOLOv8n_train_eval_seed42.py` | Table S13 YOLOv8n train/eval traceability |
| `reproduce_TablesS14_S16_deployment_and_weight_sensitivity.py` | Tables S14-S16 deployment, multi-objective score, and weight-sensitivity source data |
| `prepare_TableS17_VisDrone2019_DET_for_YOLO.py` | VisDrone2019-DET conversion to YOLO format for Table S17 |
| `reproduce_TableS17_VisDrone_YOLO11n_3res_seed42.py` | Table S17 YOLO11n VisDrone 800/960/1280 training and evaluation |
| `audit_TableS17_VisDrone2019_DET_split_leakage.py` | VisDrone official-split auxiliary duplicate audit |

## Execution notes

Scripts that regenerate training/evaluation outputs require local access to raw PIO or VisDrone images, labels, trained weights, and prediction-output files. The original PIO and VisDrone2019-DET images are not redistributed in this repository.
