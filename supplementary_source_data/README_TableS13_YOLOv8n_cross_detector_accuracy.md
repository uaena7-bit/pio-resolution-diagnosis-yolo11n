# Table S13 — YOLOv8n Cross-Detector Robustness Check

This folder provides the final source data for the YOLOv8n cross-detector robustness check.

YOLOv8n was used only to examine whether the leakage-controlled input-resolution diagnosis protocol generalizes beyond the main YOLO11n detector. It is not used as a detector-ranking benchmark, not used for the main seed-repeatability analysis, and not used for the final deployment-resolution conclusion.

## Final Paper Source Data

The final source-data file for the paper is:

```text
supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv
```

This file contains only accuracy and training-time results:

- training time
- validation precision, recall, mAP50, and mAP50-95
- test precision, recall, mAP50, and mAP50-95
- best checkpoint path

Latency columns are intentionally excluded from the final Table S13 source data.

## Raw Latency Recheck File

The 1280-pixel YOLOv8n test-speed recheck file is:

```text
raw_logs/YOLOv8n_1280_test_latency_recheck_raw.csv
```

The original single-run YOLOv8n summary contained a timing anomaly for the 1280-pixel test inference time:

```text
Initial raw recorded value: 116.995 ms/img
Independent re-runs: 5.6 / 5.1 / 5.3 ms/img
Median rechecked inference time: 5.3 ms/img
```

Because only the 1280-pixel test speed was re-checked, the latency values are not uniform in provenance across all YOLOv8n resolutions. Therefore, YOLOv8n latency is not used for the final deployment analysis.

Deployment-oriented conclusions should be based on the repeated locked-weight YOLO11n benchmark.

## Recommended Paper Statement

YOLOv8n was used as a cross-detector robustness check under the same leakage-controlled split and input-resolution protocol. Because the latency values from the initial single-run summary contained a timing anomaly, only accuracy and training-time results were used for the YOLOv8n cross-detector comparison. Deployment-oriented conclusions were based on the repeated locked-weight YOLO11n benchmark.

## Data Dictionary — `TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv`

| Column | Description |
|---|---|
| `experiment_id` | Full experiment identifier |
| `model` | Detector architecture |
| `resolution` | Input image size in pixels |
| `seed` | Random seed |
| `best_pt_path` | Absolute path to the best checkpoint in the local experiment environment |
| `training_time_h` | Total training time in hours, measured by the training script |
| `val_precision` | Validation precision |
| `val_recall` | Validation recall |
| `val_map50` | Validation mAP@0.50 |
| `val_map50_95` | Validation mAP@0.50:0.95 |
| `test_precision` | Test precision |
| `test_recall` | Test recall |
| `test_map50` | Test mAP@0.50 |
| `test_map50_95` | Test mAP@0.50:0.95 |

## Data Dictionary — `YOLOv8n_1280_test_latency_recheck_raw.csv`

| Column | Description |
|---|---|
| `run_id` | Individual re-run identifier |
| `preprocess_ms_per_img` | Preprocessing time per image |
| `inference_ms_per_img` | Inference time per image |
| `postprocess_ms_per_img` | Post-processing time per image |
| `total_ms_per_img` | Sum of preprocessing, inference, and post-processing time per image |
| Remaining columns | Hardware, software, and evaluation context for traceability |

## Interpretation

The cleaned YOLOv8n accuracy summary shows that test mAP50-95 increased from 800 to 960 and further to 1280 pixels. This supports the robustness of the proposed leakage-controlled resolution-diagnosis protocol across another lightweight YOLO-family detector.

However, YOLOv8n is not used to select the final deployment resolution. The final deployment and Pareto-style resolution-selection analysis should use the repeated locked-weight YOLO11n benchmark.
