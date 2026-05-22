# pio-resolution-diagnosis-yolo11n

Reproducibility materials for the manuscript:

**Leakage-Controlled Input-Resolution Diagnosis for Dense Small-Object Detection in High-Density Visual Signals**

This repository provides the public reproducibility materials corresponding to release **v1.0.0** for the final pre-submission CSSP manuscript.

## What is included

- leakage-controlled split manifests for `PIO-GRDB-MD5-7_1_2`;
- MD5, source-group, and near-duplicate leakage audit summaries;
- group definitions for scale- and density-stratified analysis;
- figure source data for the main ordinary-AP figures:
  - `Fig2_accuracy_cost_tradeoff_source_data.csv`;
  - `Fig3_locked_ordinary_scale_AP_source_data.csv`;
  - `Fig4_locked_ordinary_density_AP_source_data.csv`;
- figure-generation scripts for Fig. 1-4, Fig. S2, and Fig. S3;
- supplementary source data for Tables S4-S12;
- bootstrap and counting-calibration outputs used in the supplementary materials.

## Final v1.0.0 consistency update

The final v1.0.0 update synchronizes the repository with the CSSP pre-submission manuscript and supplementary materials. It includes:

- clarified ordinary subgroup AP versus strict global-first diagnostic AP notes;
- revised Table S6 source data with full-test-set global mAP50-95 reference values;
- standardized Table S12 probability-column names as `P(Delta < 0)` and `P(Delta > 0)` in the supplementary materials;
- documented the hardware setting as a single NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB dedicated GPU memory;
- documented the optional GitHub-only continuous density diagnostic workflow for future reuse.

No changes were made to the leakage-controlled split, leakage audit results, model training outputs, prediction caches, primary AP results, bootstrap source values, or counting-calibration source values.

## Important distinction: ordinary AP vs strict diagnostic AP

The main-manuscript Fig. 3 and Fig. 4 use **ordinary subgroup AP** values from the ordinary scale- and density-stratified evaluation tables.

They should not be confused with **strict global-first diagnostic AP**, which is used only as a supplementary contribution-oriented diagnostic analysis. Strict diagnostic AP preserves the global ranked prediction list and global matching results before attributing matched true positives to predefined groups. Therefore, strict diagnostic AP values are not directly comparable with ordinary subgroup AP values in Figs. 3 and 4 or with standard global AP.

## Reproducing Fig. 2-4

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

## Optional GitHub-only continuous density diagnostic

The reviewer-requested continuous density diagnostic is treated as an optional GitHub-only reproducibility aid rather than a manuscript result. The script

```bash
python evaluation_scripts/make_figureS4_continuous_density_diagnostic.py
```

expects a per-image diagnostic CSV at:

```text
figure_source_data/FigureS4_continuous_density_diagnostic_source_data.csv
```

A schema template is provided as:

```text
figure_source_data/FigureS4_continuous_density_diagnostic_source_data_TEMPLATE.csv
```

This optional figure is intended for future reuse when local per-image prediction outcome caches are available. It is not used to change the locked split, primary AP values, counting-calibration values, or manuscript conclusions.

## Data availability note

The original PIO images are not redistributed in this repository. Users should obtain the raw PIO dataset from the original dataset source and then use the released split manifests and scripts for reproduction.

## Citation

If you use these reproducibility materials, please cite:

Song, Y.: Reproducibility materials for leakage-controlled input-resolution diagnosis of dense small-object detection on PIO. GitHub repository, release v1.0.0. https://github.com/uaena7-bit/pio-resolution-diagnosis-yolo11n (2026). Accessed 22 May 2026.

