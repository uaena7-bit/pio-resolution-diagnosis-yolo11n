# Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection

This repository contains the manuscript-linked reproducibility materials for:

**Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection**

**Manuscript-linked release:** `v1.2.1 - submission-aligned final figures and metadata`

## Study scope

The study treats input resolution as an application and deployment design variable for dense livestock monitoring. PIO is the primary dense broiler case study. PigDetect provides independent livestock-domain external validation after split-leakage auditing and cleaning.

The repository supports four connected parts of the study:

1. leakage-controlled dataset construction and audit;
2. resolution-dependent detection, scale, density, and counting diagnosis;
3. bootstrap uncertainty and repeated deployment benchmarking;
4. objective-specific, deployment-oriented diagnosis rather than selection of one universally optimal resolution.

## Primary operating points and interpretation

The prespecified primary comparison uses 800, 960, and 1280 pixels as low-, intermediate-, and high-compute operating points. The 1024-pixel PIO run is retained only as an exploratory intermediate-resolution sensitivity check and is not treated as a primary resolution setting.

The conclusions are objective- and protocol-specific:

- **800 px** achieved the lowest counting MAE only under the adopted validation-based threshold-transfer protocol.
- **960 px** ranked highest only under the specified illustrative composite weighting.
- **1280 px** provided the strongest box-level localization performance at the highest measured computational cost.
- The study does **not** identify a universally optimal input resolution.

## Dataset roles

### PIO

PIO is the main dense poultry-monitoring dataset. Its materials include the grouped leakage-controlled split, multi-seed YOLO11n repeatability, scale and density analyses, counting calibration, paired bootstrap analysis, deployment benchmarking, cross-detector checks, and multi-objective diagnostic analyses.

### PigDetect

PigDetect is the independent livestock-domain external-validation dataset. The cleaned split contains 2,431 training images, 241 validation images, and 250 test images with 5,436 test instances. The original audit found 18 SSIM-confirmed train-validation near-duplicate pairs involving 9 validation images; after cleaning, filename, filename-stem, image-MD5, label-MD5, and SSIM-confirmed cross-split checks were all zero.

## Repository layout

| Path | Contents |
|---|---|
| `split_manifest/` | PIO leakage-controlled split manifests and statistics |
| `audit_summaries/` | PIO and PigDetect leakage-audit summaries and traceability records |
| `evaluation_scripts/` | Evaluation and diagnostic scripts released for the manuscript |
| `figure_scripts/` | Scripts used to generate quantitative manuscript and supplementary figures |
| `figure_source_data/` | Locked numerical and metadata sources for main-text figures |
| `main_figures/` | Final main-text figure files |
| `supplementary_figures/` | Final supplementary figure files |
| `supplementary_source_data/` | Locked source data for supplementary figures and tables |
| `external_validation/PigDetect/` | PigDetect protocol, dataset role, and reproduction notes |
| `repository_audit/` | Cross-document numerical and traceability checks |
| `paper_pdfs/` | Submission-aligned manuscript and supplementary-material PDFs |

## Reproduction outline

1. Check out the manuscript-linked `v1.2.1` release.
2. Install the environment required by the relevant scripts.
3. Obtain PIO and PigDetect from their original sources.
4. Reconstruct or verify the released split manifests and leakage-audit outputs.
5. Run the resolution-specific evaluation using the locked protocols documented in the manuscript.
6. Compare generated outputs with the locked source data in `figure_source_data/` and `supplementary_source_data/`.

## Raw-data and redistribution policy

This repository does not redistribute raw PIO or PigDetect images, restricted third-party annotations, Ultralytics source code, pretrained detector weights, trained checkpoints, or prediction overlays. Users must obtain third-party datasets from their original sources and comply with their terms.

The repository license applies only to files authored and released in this repository. It does not relicense third-party datasets, software, pretrained models, or model weights.

## Release alignment

Release `v1.2.1` aligns the repository title, citation metadata, final figures, locked source data, figure scripts, and paper PDFs with the final submission package prepared on 30 July 2026.

## Citation

Please cite the manuscript and manuscript-linked repository release:

> Song, Y., Xu, J., Huang, L., Lin, H., and Dong, Z. *Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection*. GitHub repository release v1.2.1, 2026.

Repository: `uaena7-bit/pio-resolution-diagnosis-yolo11n`
