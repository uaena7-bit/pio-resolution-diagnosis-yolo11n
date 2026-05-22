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
- supplementary source data for Tables S4-S12 and metadata/scripts for Fig. 5 and Figs. S1-S3.

## Final pre-submission consistency updates

- Clarified the distinction between ordinary subgroup AP and strict global-first diagnostic AP.
- Clarified that strict diagnostic AP values are not directly comparable with ordinary subgroup AP values in Figs. 3 and 4 or with standard global AP.
- Revised Table S6 source data by adding full-test-set global mAP50-95 reference values.
- Standardized Table S12 probability-column descriptions as `P(Delta < 0)` and `P(Delta > 0)` in the supplementary materials.
- Documented the hardware setting as a single NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB of dedicated GPU memory.
- Added an optional GitHub-only continuous density diagnostic workflow for future reuse when local per-image prediction outcome caches are available.

No changes were made to the leakage-controlled split, leakage audit results, model training outputs, prediction caches, primary AP results, bootstrap source values, or counting-calibration source values.

The original PIO images are not redistributed. Users should obtain the raw dataset from the original dataset source.

