# pio-resolution-diagnosis-yolo11n

Reproducibility materials for the manuscript:

**Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals**

This repository provides the public reproducibility materials corresponding to release **v1.0.0** for the CSSP submission version of the manuscript.

## What is included

- leakage-controlled split manifests for PIO-GRDB-MD5-7_1_2;
- MD5, source-group, and near-duplicate leakage audit summaries;
- group definitions for scale- and density-stratified analysis;
- figure source data for the main ordinary-AP figures:
  - `Fig2_accuracy_cost_tradeoff_source_data.csv`;
  - `Fig3_locked_ordinary_scale_AP_source_data.csv`;
  - `Fig4_locked_ordinary_density_AP_source_data.csv`;
- figure-generation scripts for Fig. 1鈥? and Fig. S3;
- bootstrap and counting-calibration outputs used in the supplementary materials.

## Important distinction: ordinary AP vs strict diagnostic AP

The main-manuscript Fig. 3 and Fig. 4 use **ordinary AP** values from the ordinary scale- and density-stratified evaluation tables.

They should not be confused with strict global-first diagnostic AP, which is used only as a supplementary diagnostic analysis. Strict diagnostic AP is not a replacement for standard global AP or ordinary subgroup AP.

## Reproducing Fig. 2鈥?

Install the minimal plotting dependencies:

```bash
pip install pandas matplotlib
```

From the repository root, run:

```bash
python evaluation_scripts/make_figure2_accuracy_cost_tradeoff.py
python evaluation_scripts/make_figure3_ordinary_scale_AP.py
python evaluation_scripts/make_figure4_ordinary_density_AP.py
```

The generated figures will be written to the `figures/` directory as PNG and PDF files.

## Data availability note

The original PIO images are not redistributed in this repository. Users should obtain the raw PIO dataset from the original dataset source and then use the released split manifests and scripts for reproduction.

## Release note

This release overwrites the earlier v1.0.0 materials with the final CSSP-oriented reproducibility package. The release is intended to match the final submission manuscript and its Data Availability statement.

