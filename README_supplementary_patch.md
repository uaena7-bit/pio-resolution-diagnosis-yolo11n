# Supplementary patch contents

This patch adds supplementary source data and scripts for the CSSP-oriented v1.0.0 reproducibility materials.

## Added directories and files

- `supplementary_source_data/`
  - Table S4–S12 source CSV files
  - Figure S2 strict diagnostic AP source data
- `figure_source_data/`
  - Fig. 5 qualitative case metadata
  - Fig. S1 low-/medium-density qualitative case metadata
- `evaluation_scripts/`
  - `make_figure5_qualitative_resolution_comparison.py`
  - `make_figureS1_low_medium_no_gain_examples.py`
  - `make_figureS2_strict_diagnostic_AP.py`

## Notes

Raw PIO images, model weights, and prediction overlays are not redistributed in this repository. Qualitative figure scripts require local raw PIO data and prediction-cache CSV files.

Fig. 3 and Fig. 4 in the main manuscript use ordinary subgroup AP, while Fig. S2 uses strict global-first diagnostic AP. These values should not be mixed.
