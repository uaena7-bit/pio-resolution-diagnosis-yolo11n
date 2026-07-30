# Evaluation scripts

This directory contains evaluation and reproduction scripts supporting the
leakage-controlled and deployment-oriented diagnosis of input-resolution trade-offs.

## Scope

The scripts support the following analyses:

- leakage-controlled PIO split construction and audit;
- duplicate and near-duplicate screening;
- standard detection evaluation;
- maximum-detection sensitivity analysis;
- scale- and density-stratified evaluation;
- paired image-level bootstrap analysis;
- counting-threshold calibration;
- locked-weight deployment benchmarking;
- figure and supplementary-table generation.

## Data requirements

Some scripts operate directly on locked CSV source files distributed in this
repository. Other scripts require local access to the original PIO or PigDetect
images, labels, trained weights, or prediction-output files.

Raw dataset images, trained checkpoints, pretrained third-party weights, and
prediction overlays are not redistributed. Users should obtain the original
datasets and required third-party software from their official sources and
comply with the corresponding licenses and terms of use.

## Reproducibility

The fixed source data used by the manuscript are stored in:

- `figure_source_data/`
- `supplementary_source_data/`
- `audit_summaries/`
- `bootstrap/`
- `counting_calibration/`
- `external_validation/PigDetect/`

The primary PIO experiments and PigDetect external-validation results should be
interpreted according to the protocols documented in the main repository README
and the accompanying manuscript.
