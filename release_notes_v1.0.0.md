# v1.0.0 - CSSP reproducibility materials

This release provides the reproducibility materials for the CSSP manuscript:

**Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals**

This `v1.0.0` release corresponds to commit `c2944e2`.

Included materials:

- leakage-controlled split manifests and leakage-audit summaries;
- group definitions for scale- and density-stratified diagnosis;
- evaluation scripts and fixed source data for main and supplementary figures/tables;
- bootstrap uncertainty outputs and counting-calibration materials;
- source data and scripts for Fig. 2, Fig. 3, and Fig. 4;
- supplementary source data for Tables S4-S17 and metadata/scripts for Fig. 5 and Figs. S1-S3;
- YOLO11n seed-repeatability source data across 800, 960, and 1280 pixels using seeds 42, 123, and 2024;
- YOLOv8n cross-detector robustness-check source data for Table S13;
- 30-repeat locked-weight YOLO11n deployment benchmark source data and raw repeat logs for Table S14;
- updated multi-objective resolution-selection source data and README for Table S15;
- multi-objective weight-sensitivity source data for Table S16;
- VisDrone2019-DET external cross-domain stress-test source data for Table S17;
- auxiliary duplicate-audit summaries for the official VisDrone train/validation/test-dev splits;
- reproduction scripts for Tables S13-S17;
- Figure S2 in a unified line-chart style consistent with main-text Fig. 3;
- notes for strict global-first diagnostic AP, including that these values are not directly comparable with ordinary subgroup AP values in Figs. 3 and Fig. 4 or with standard global AP;
- deployment benchmark notes, including that total latency is the median end-to-end time per image and may not equal the sum of independently computed stage medians;
- documented hardware setting as a single NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of dedicated GPU memory;
- traceability logs for the YOLOv8n timing recheck and the YOLO11n 30-repeat locked-weight deployment benchmark.

No changes were made to the leakage-controlled PIO split, leakage-audit pass/fail results, primary PIO dataset partition, or original PIO image data. The added materials support leakage-controlled input-resolution evaluation, scale-density-counting-deployment diagnosis, multi-objective weight-sensitivity analysis, and external cross-domain protocol stress testing.

The original PIO and VisDrone2019-DET images are not redistributed. Users should obtain the raw datasets from their original dataset sources.
