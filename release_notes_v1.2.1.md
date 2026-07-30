# v1.2.1 - Submission-aligned final figures and metadata

Release date: 30 July 2026

This patch release aligns the public reproducibility repository with the final manuscript:

**Leakage-Controlled and Deployment-Oriented Diagnosis of Input-Resolution Trade-offs for Dense Broiler Detection**

## Changes

- Updated the repository title and citation wording from deployment-aware resolution selection to deployment-oriented diagnosis of input-resolution trade-offs.
- Preserved the primary operating points at 800, 960, and 1280 px.
- Clarified that 1024 px is only an exploratory intermediate-resolution sensitivity check.
- Locked the objective-specific interpretation:
  - 800 px has the lowest counting MAE only under the validation-based threshold-transfer protocol.
  - 960 px ranks highest only under the specified illustrative composite weighting.
  - 1280 px provides the highest box-level localization accuracy at the highest measured computational cost.
  - No universally optimal resolution is claimed.
- Replaced all main-text and supplementary figures with the final submission-aligned versions.
- Updated Fig. 4 to the continuous-axis export with complete canvas margins.
- Updated Fig. S2 spacing to prevent crowding between the x-axis title and legend.
- Updated Fig. S3 to use consistent styles across panels, include 1024 px only in the exploratory full-range panel, and remove the obstructing annotation box.
- Added locked plotting source data and reproducible plotting scripts.
- Added the final submission-aligned manuscript and ESM PDFs.
- Updated `README.md` and `CITATION.cff`.

## Integrity statement

This release does not alter experimental measurements. Figure revisions are limited to layout, typography, legend placement, black-and-white distinguishability, and final submission formatting.
