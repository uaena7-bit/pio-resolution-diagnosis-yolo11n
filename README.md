# Leakage-controlled and deployment-aware input-resolution selection for dense broiler detection

This repository contains the manuscript-linked reproducibility materials for:

**Leakage-controlled and deployment-aware input-resolution selection for dense broiler detection in precision poultry monitoring**

**Manuscript-linked release:** `v1.2.0 - PigDetect external-validation update`

## Study scope

The study treats input resolution as an application and deployment design variable for dense livestock monitoring. PIO is the primary dense broiler case study. PigDetect provides independent livestock-domain external validation after split-leakage auditing and cleaning.

The repository supports four connected parts of the study:

1. leakage-controlled dataset construction and audit;
2. resolution-dependent detection, scale, density, and counting diagnosis;
3. bootstrap uncertainty and repeated deployment benchmarking;
4. task-oriented resolution selection with an independent livestock-domain validation.

## Primary operating points

The prespecified primary comparison uses 800, 960, and 1280 pixels as low-, intermediate-, and high-compute operating points. The 1024-pixel PIO run is retained as a targeted sensitivity check around the intermediate-to-high region and is not part of the full primary experiment matrix.

## Dataset roles

### PIO

PIO is the main dense poultry-monitoring dataset. Its materials include the grouped leakage-controlled split, multi-seed YOLO11n repeatability, scale and density analyses, counting calibration, paired bootstrap analysis, deployment benchmarking, cross-detector check, and multi-objective selection analyses.

### PigDetect

PigDetect is the independent livestock-domain external-validation dataset. The cleaned split contains 2,431 training images, 241 validation images, and 250 test images with 5,436 test instances. The original audit found 18 SSIM-confirmed train-validation near-duplicate pairs involving 9 validation images; after cleaning, the filename, filename-stem, image-MD5, label-MD5, and SSIM-confirmed cross-split checks were all zero.

PigDetect source data cover locked test performance, scale and density stratification, 1,000-replicate paired image-level bootstrap analysis, repeated locked-weight deployment benchmarking, and the locked Fig. 6 qualitative evidence.

## Repository layout

| Path | Contents |
|---|---|
| `split_manifest/` | PIO leakage-controlled split manifests and statistics. |
| `audit_summaries/` | PIO and PigDetect leakage-audit summaries and traceability records. |
| `evaluation_scripts/` | Evaluation and diagnostic scripts released for the manuscript. |
| `figure_scripts/` | Scripts used to generate manuscript and supplementary figures where redistribution is possible. |
| `figure_source_data/` | Fixed CSV/JSON source data for manuscript tables and figures. |
| `main_figures/` | Final main-text figure files, including PigDetect Fig. 6. |
| `supplementary_figures/` | Final supplementary figures. |
| `supplementary_source_data/` | Source data for supplementary tables, including Tables S17a-S17g. |
| `external_validation/PigDetect/` | PigDetect protocol, dataset role, and reproduction notes. |
| `repository_audit/` | Cross-document numerical and traceability checks. |

## PigDetect manuscript-linked files

- `audit_summaries/PigDetect_clean_split_audit_summary.csv`
- `audit_summaries/PigDetect_source_to_table_map.csv`
- `supplementary_source_data/Table_S17a_PigDetect_clean_split_audit.csv`
- `supplementary_source_data/Table_S17b_PigDetect_training_and_locked_test_performance.csv`
- `supplementary_source_data/Table_S17c_PigDetect_scale_stratified_AP.csv`
- `supplementary_source_data/Table_S17d_PigDetect_density_stratified_AP.csv`
- `supplementary_source_data/Table_S17e_PigDetect_paired_bootstrap_summary.csv`
- `supplementary_source_data/Table_S17f_PigDetect_locked_weight_deployment_benchmark.csv`
- `supplementary_source_data/Table_S17g_PigDetect_source_to_manuscript_map.csv`
- `figure_source_data/Table12_PigDetect_main_summary_source_data.csv`
- `figure_source_data/Fig6_PigDetect_qualitative_evidence_metadata.json`
- `main_figures/Fig6_PigDetect_external_validation.pdf`
- `main_figures/Fig6_PigDetect_external_validation.png`

## Reproduction outline

1. Check out the manuscript-linked `v1.2.0` release.
2. Install the environment required by the relevant scripts; consult script headers and `evaluation_scripts/README.md` where provided.
3. Obtain PIO and PigDetect from their original sources.
4. Reconstruct or verify the released split manifests and audit outputs.
5. Run the resolution-specific evaluation using the locked protocols documented in the manuscript and source records.
6. Compare generated outputs with the fixed source data in `figure_source_data/` and `supplementary_source_data/`.

## Raw-data and redistribution policy

This repository does not redistribute raw PIO or PigDetect images, third-party dataset annotations where redistribution is restricted, Ultralytics source code, pretrained detector weights, trained checkpoints, or prediction overlays. Users must obtain third-party datasets from their original sources and comply with the relevant terms.

The repository license applies only to files authored and released in this repository. It does not relicense third-party datasets, software, pretrained models, or model weights.

## Release alignment

Release `v1.2.0` replaces the active VisDrone protocol-transfer materials with the manuscript-linked PigDetect livestock-domain external validation. Historical releases remain available through Git history and their archived tags.

## Citation

Please cite the manuscript and the manuscript-linked repository release:

> Song, Y. *Leakage-controlled and deployment-aware input-resolution selection for dense broiler detection in precision poultry monitoring*. GitHub repository release v1.2.0, 2026.

Repository: `uaena7-bit/pio-resolution-diagnosis-yolo11n`
