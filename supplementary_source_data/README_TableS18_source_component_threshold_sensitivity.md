# Table S18. Source-component sensitivity of validation-calibrated counting thresholds

This source table reports the additional validation-source-component sensitivity analysis.
For each input resolution, the counting confidence threshold was selected from the full validation set, from validation component 1 only, and from validation component 10 only. Each selected threshold was then applied once to the unchanged held-out test set.

## Input files used by the script

- `00_DATASET/PIO-GRDB-MD5-7_1_2/reports/manifest.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_val_800.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_val_960.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_val_1280.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_test_800.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_test_960.csv`
- `03_RESULTS/counting_val_calibrated_conf_3res/predictions_test_1280.csv`

## Main interpretation

The full-validation and dominant-component calibration results are identical for all three resolutions. Calibration on the smaller validation component changes the selected threshold for each resolution, showing that counting-threshold calibration is partly affected by validation-source composition.
