# v1.2.0 - PigDetect external-validation update

This release aligns the reproducibility repository with the revised Precision Agriculture manuscript and its supplementary material.

## Main changes

- Replaced the active VisDrone external sanity-check materials with an independent livestock-domain validation on PigDetect.
- Added the PigDetect clean-split audit and source-to-manuscript traceability records.
- Added source CSV files for Tables S17a-S17g and the rounded main-manuscript Table 12.
- Added the locked PigDetect Fig. 6 artwork and its protocol metadata.
- Updated the repository README to emphasize the dense livestock-monitoring application, the PIO/PigDetect dataset roles, and the rationale for the 800/960/1280 primary operating points.
- Removed repeated defensive wording about detector novelty and universal resolution claims from the active repository description.
- Retained PIO as the primary dataset with multi-seed repeatability, counting calibration, cross-detector analysis, and multi-objective selection.

## Experimental results

This release does not alter the locked PIO results. PigDetect results are added as the manuscript-linked external-validation evidence. PigDetect uses a fixed seed for external validation; the dedicated training-seed repeatability evidence remains the three-seed PIO experiment.

## Data redistribution

Raw PIO and PigDetect images, trained checkpoints, and prediction caches are not redistributed. Users must obtain the datasets from their original sources.

## Versioning

The previous `v1.1.0` release remains an archived record of the earlier manuscript configuration. The revised manuscript should cite `v1.2.0`.
