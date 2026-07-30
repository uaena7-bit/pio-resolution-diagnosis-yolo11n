# Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection

This repository contains the manuscript-linked reproducibility materials for:

**Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection**

**Manuscript-linked release:** `v1.2.1 - Submission-aligned final figures and metadata`

## Study scope

This study treats input resolution as an application and deployment design variable for dense livestock monitoring.

The PIO dataset is used as the primary dense broiler detection case study. PigDetect provides independent livestock-domain external validation after split-leakage auditing and cleaning.

The repository supports four connected components of the study:

1. leakage-controlled dataset construction and auditing;
2. resolution-dependent detection, scale, density, and counting diagnosis;
3. bootstrap uncertainty analysis and repeated deployment benchmarking;
4. objective-specific, deployment-oriented diagnosis rather than selection of one universally optimal resolution.

## Primary operating points and interpretation

The prespecified primary comparison uses three input resolutions:

- **800 px:** low-compute operating point;
- **960 px:** intermediate-compute operating point;
- **1280 px:** high-compute operating point.

The **1024-pixel** PIO run is retained only as an exploratory intermediate-resolution sensitivity check and is not treated as a primary resolution setting.

The conclusions are objective- and protocol-specific:

- **800 px** achieved the lowest counting MAE only under the adopted validation-based threshold-transfer protocol.
- **960 px** ranked highest only under the specified illustrative composite weighting.
- **1280 px** provided the strongest box-level localization performance at the highest measured computational cost.
- The study does **not** identify a universally optimal input resolution.

## Dataset roles

### PIO

PIO is the main dense poultry-monitoring dataset.

The released PIO materials include:

- grouped leakage-controlled split construction;
- exact-duplicate and source-group auditing;
- multi-seed YOLO11n repeatability analysis;
- ordinary scale-stratified and density-stratified evaluation;
- strict diagnostic subgroup evaluation;
- validation-based counting calibration;
- paired bootstrap uncertainty analysis;
- deployment benchmarking;
- cross-detector sensitivity checks;
- multi-objective deployment diagnosis.

### PigDetect

PigDetect is used as an independent livestock-domain external-validation dataset.

The leakage-cleaned PigDetect split contains:

- 2,431 training images;
- 241 validation images;
- 250 test images;
- 5,436 test instances.

The original audit identified 18 SSIM-confirmed train-validation near-duplicate pairs involving 9 validation images.

After cleaning, all of the following cross-split checks were zero:

- exact filename overlap;
- filename-stem overlap;
- image-MD5 overlap;
- label-MD5 overlap;
- SSIM-confirmed near-duplicate overlap.

## Main findings

On PIO, test mAP50-95 increased with input resolution:

- 800 px: 0.7199;
- 960 px: 0.7420;
- 1280 px: 0.7543.

Higher input resolution improved strict localization performance, especially for difficult scale and density groups, but also increased:

- training time;
- GFLOPs;
- inference latency;
- GPU memory demand.

Under the adopted validation-based confidence-threshold transfer protocol, the 800-pixel condition achieved the lowest test counting MAE.

Under the specified illustrative composite weighting, the 960-pixel condition obtained the highest composite score.

The 1280-pixel condition achieved the highest box-level localization accuracy but incurred the highest measured computational cost.

PigDetect external validation showed the same resolution-dependent strict-localization trend.

These findings support objective-specific resolution diagnosis rather than a universal resolution recommendation.

## Repository layout

| Path | Contents |
|---|---|
| `split_manifest/` | PIO leakage-controlled split manifests and split statistics |
| `audit_summaries/` | PIO and PigDetect leakage-audit summaries and traceability records |
| `evaluation_scripts/` | Evaluation and diagnostic scripts used for manuscript analyses |
| `figure_scripts/` | Scripts used to generate quantitative manuscript and supplementary figures |
| `figure_source_data/` | Locked numerical and metadata sources for main-text figures |
| `main_figures/` | Final main-text figure files |
| `supplementary_figures/` | Final supplementary figure files |
| `supplementary_source_data/` | Locked source data for supplementary figures and tables |
| `supplementary_tables/` | Supplementary table materials |
| `external_validation/PigDetect/` | PigDetect protocol, dataset role, and reproduction notes |
| `counting_calibration/` | Validation-based confidence-threshold calibration materials |
| `bootstrap/` | Paired bootstrap uncertainty-analysis materials |
| `repository_audit/` | Cross-document numerical and traceability checks |
| `paper_pdfs/` | Submission-aligned manuscript and supplementary-material PDFs |
| `release_notes/` | Versioned release documentation |

## Final figure set

### Main-text figures

- `Fig1_Resolution_Diagnosis_Workflow`
- `Fig2_Accuracy_Training_Cost`
- `Fig3_Scale_Stratified_AP`
- `Fig4_Density_Stratified_AP`
- `Fig5_Dense_Scene_Qualitative_Comparison`
- `Fig6_PigDetect_external_validation`

### Supplementary figures

- `FigS1_Low_Medium_Density_Qualitative`
- `FigureS2_strict_diagnostic_AP_line_style`
- `FigS3_validation_confidence_MAE_sweep`

Important final-layout details:

- Fig. 2 uses a two-panel line-plot design rather than a bar-chart design.
- Fig. 4 uses a continuous y-axis and includes sufficient canvas margins to prevent label clipping.
- Fig. S2 includes expanded spacing between the x-axis title, legend, and adjacent panel.
- Fig. S3 includes 1024 px only in the full-range exploratory panel; the enlarged panel contains only 800, 960, and 1280 px.

## Reproduction outline

1. Check out the manuscript-linked `v1.2.1` release.
2. Install the software environment required by the relevant scripts.
3. Obtain PIO and PigDetect from their original sources.
4. Reconstruct or verify the released split manifests and leakage-audit outputs.
5. Run the resolution-specific evaluation using the locked protocols documented in the manuscript.
6. Compare generated outputs with the locked source data in:
   - `figure_source_data/`;
   - `supplementary_source_data/`.
7. Regenerate the quantitative figures using the scripts in `figure_scripts/`.

## Counting calibration

Counting evaluation uses confidence thresholds selected on the validation set and transferred unchanged to the test set.

The reported counting conclusions are therefore specific to the adopted validation-based threshold-transfer protocol.

The confidence-threshold sensitivity sweep is provided for:

- 800 px;
- 960 px;
- 1024 px;
- 1280 px.

The 1024-pixel condition is included only as an exploratory intermediate-resolution sensitivity check.

## Reproducibility and traceability

The repository provides, where redistribution is permitted:

- split manifests;
- leakage-audit summaries;
- evaluation scripts;
- locked numerical figure-source data;
- supplementary source data;
- figure-generation scripts;
- bootstrap-analysis materials;
- counting-calibration records;
- deployment-benchmark summaries;
- repository consistency checks;
- final submission-aligned figures and PDFs.

File-level SHA256 checksums for the `v1.2.1` submission-alignment package are provided in:

`FILE_MANIFEST_SHA256.csv`

## Raw-data and redistribution policy

This repository does not redistribute:

- raw PIO images;
- raw PigDetect images;
- restricted third-party annotations;
- Ultralytics source code;
- pretrained detector weights;
- trained checkpoints;
- prediction overlays containing restricted source imagery.

Users must obtain third-party datasets from their original sources and comply with the corresponding terms and licenses.

The repository license applies only to files authored and released in this repository. It does not relicense third-party datasets, software, pretrained models, or model weights.

## Release alignment

Release `v1.2.1` aligns the public repository with the final manuscript submission prepared on 30 July 2026.

The release updates:

- repository title and study positioning;
- citation metadata;
- final main-text figures;
- final supplementary figures;
- locked source data;
- quantitative plotting scripts;
- manuscript and supplementary-material PDFs;
- release documentation.

The release does not change the experimental measurements or scientific conclusions.

## Citation

Please cite the associated manuscript and manuscript-linked repository release:

> Song, Y., Xu, J., Huang, L., Lin, H., and Dong, Z.  
> *Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection*.  
> GitHub repository release v1.2.1, 2026.

Repository:

`https://github.com/uaena7-bit/pio-resolution-diagnosis-yolo11n`

## License

Repository-authored code and documentation are released under the MIT License unless otherwise stated.

Third-party datasets, pretrained models, software packages, and related assets remain subject to their original licenses and terms.
