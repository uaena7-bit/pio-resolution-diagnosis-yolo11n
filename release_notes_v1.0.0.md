# v1.0.0 - CSSP reproducibility materials

This release provides the final pre-submission reproducibility materials for the CSSP manuscript:

**Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals**

## Included materials

- leakage-controlled split manifests for `PIO-GRDB-MD5-7_1_2`;
- MD5, source-group, and near-duplicate leakage audit summaries;
- group definitions for scale- and density-stratified diagnosis;
- locked source data for main and supplementary figures/tables;
- evaluation scripts for generating the released figures where source data are available;
- bootstrap uncertainty and counting-calibration materials;
- supplementary source data for Tables S4-S15 and metadata/scripts for Fig. 5 and Figs. S1-S3.

## Final pre-submission consistency updates

- Clarified the distinction between ordinary subgroup AP and strict global-first diagnostic AP.
- Clarified that strict diagnostic AP values are not directly comparable with ordinary subgroup AP values in Figs. 3 and 4 or with standard global AP.
- Revised Table S6 source data by adding full-test-set global mAP50-95 reference values.
- Standardized Table S12 probability-column descriptions as `P(Delta < 0)` and `P(Delta > 0)` in the supplementary materials.
- Documented the hardware setting as a single NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of dedicated GPU memory.
- Added an optional GitHub-only continuous density diagnostic workflow for future reuse when local per-image prediction outcome caches are available.

No changes were made to the leakage-controlled split, leakage audit results, model training outputs, prediction caches, primary AP results, bootstrap source values, or counting-calibration source values.

The original PIO images are not redistributed. Users should obtain the raw dataset from the original dataset source.

## Final consistency update for CSSP revision

This update synchronizes the repository with the final revised manuscript and supplementary materials.

Added or updated source-data files:

- `supplementary_source_data/TableS11_YOLO11n_seed_repeatability_three_resolutions.csv`
- `supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv`
- `supplementary_source_data/TableS14_YOLO11n_deployment_benchmark_locked_weights.csv`
- `supplementary_source_data/TableS15_multi_objective_resolution_selection.csv`

Added supporting README files:

- `supplementary_source_data/README_TableS13_YOLOv8n_cross_detector_accuracy.md`
- `supplementary_source_data/README_TableS14_YOLO11n_deployment_benchmark_locked_weights.md`
- `supplementary_source_data/README_TableS15_multi_objective_resolution_selection.md`

Added raw traceability logs:

- `raw_logs/RawLog_TableS13_YOLOv8n_train_eval_single_run_seed42.csv`
- `raw_logs/RawLog_TableS13_YOLOv8n_1280_latency_recheck.csv`
- `raw_logs/RawLog_TableS14_YOLO11n_deployment_benchmark_raw_repeats.csv`

Added final paper-linked reproduction scripts:

- `evaluation_scripts/reproduce_TableS13_YOLOv8n_source_data.py`
- `evaluation_scripts/reproduce_TableS13_YOLOv8n_train_eval_seed42.py`
- `evaluation_scripts/reproduce_TableS14_YOLO11n_deployment_benchmark.py`
- `evaluation_scripts/reproduce_TableS15_multi_objective_resolution_selection.py`

The repository now supports the revised manuscript framing: leakage-controlled resolution evaluation, scale-density-counting-deployment diagnosis, and task-objective-driven resolution selection.
