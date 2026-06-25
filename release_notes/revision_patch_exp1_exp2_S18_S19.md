# Revision patch: source data and scripts for Tables S18 and S19

## Added files

- `supplementary_source_data/TableS18_source_component_threshold_sensitivity_summary.csv`
- `supplementary_source_data/TableS19_counting_threshold_robustness_summary.csv`
- `supplementary_source_data/README_TableS18_source_component_threshold_sensitivity.md`
- `supplementary_source_data/README_TableS19_counting_threshold_robustness.md`
- `scripts/revision_experiments/source_component_threshold_sensitivity_exp1.py`
- `scripts/revision_experiments/counting_threshold_robustness_exp2.py`

## Recommended repository update

1. Copy the new CSV files into `supplementary_source_data/`.
2. Copy the two scripts into `scripts/revision_experiments/` or `evaluation_scripts/revision_experiments/`.
3. Add a short note in the repository README that Tables S18 and S19 were added for revision analyses.
4. Update the release tag or changelog, for example `v1.1.1`.
5. Keep the raw prediction CSV files local or add them only if the repository release policy allows larger intermediate outputs.

## Reproduction order

1. Run `source_component_threshold_sensitivity_exp1.py`.
2. Run `counting_threshold_robustness_exp2.py`.
3. Compare generated summary CSV files with the source tables included here.
