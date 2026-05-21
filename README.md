# pio-resolution-diagnosis-yolo11n

This repository provides reproducibility materials for the manuscript:

**Leakage-controlled resolution diagnosis for high-density broiler chicken detection using YOLO11n on the PIO dataset**

## Contents

This repository includes:

- leakage-controlled split manifests for PIO-GRDB-MD5-7_1_2;
- MD5, source-group, and near-duplicate leakage audit summaries;
- scale and density group definitions;
- evaluation scripts for standard AP, strict diagnostic AP, bootstrap confidence intervals, counting calibration, latency, and VRAM analysis;
- source data for supplementary tables and figures.

## Dataset

The original PIO dataset is publicly available from Zenodo:

- PIO dataset DOI: 10.5281/zenodo.16686320

This repository does not redistribute the original PIO images. It provides derived split files, audit summaries, scripts, and result tables for reproducibility.

## Main experimental settings

- Dataset split: PIO-GRDB-MD5-7_1_2
- Detector: YOLO11n
- Input resolutions: 800, 960, and 1280 px
- Evaluation: global AP, scale-specific AP, density-specific AP, bootstrap confidence intervals, counting calibration, latency, and VRAM

## Citation

Please cite the associated manuscript, this repository, and the original PIO dataset when using these materials.
