# PigDetect external livestock-domain validation

PigDetect is used as an independent livestock-domain external-validation dataset for the manuscript:

**Leakage-controlled and deployment-aware input-resolution selection for dense broiler detection in precision poultry monitoring**

## Role in the study

PIO is the primary dense poultry-monitoring case study and carries the multi-seed repeatability, counting-calibration, and multi-objective resolution-selection analyses. PigDetect tests whether the resolution-dependent strict-localization trend transfers to a second livestock dataset under a leakage-cleaned split.

## Clean split and audit

- Train: 2,431 images
- Validation: 241 images
- Test: 250 images
- Test annotations: 5,436 instances
- Original train-validation audit: 18 SSIM-confirmed near-duplicate pairs involving 9 validation images
- Post-clean audit: zero filename, filename-stem, image-MD5, label-MD5, and SSIM-confirmed cross-split overlaps

## Controlled experiment

YOLO11n was trained at 800, 960, and 1280 pixels with the same split, pretrained initialization, SGD optimizer, seed 42, batch size 2, augmentation settings, and validation-based checkpoint-selection rule. PigDetect is a fixed-seed external validation; PIO provides the dedicated multi-seed repeatability evidence.

## Included files

- `protocol.json`: locked experiment and evaluation protocol
- `../../audit_summaries/PigDetect_clean_split_audit_summary.csv`: audit summary
- `../../supplementary_source_data/Table_S17a_*.csv` to `Table_S17g_*.csv`: manuscript-linked source tables
- `../../main_figures/Fig6_PigDetect_external_validation.*`: locked qualitative artwork
- `../../figure_source_data/Fig6_PigDetect_qualitative_evidence_metadata.json`: display and diagnostic metadata

Raw PigDetect images, annotations, model weights, and prediction caches are not redistributed. Obtain the dataset from the original source and comply with its license and access conditions.
