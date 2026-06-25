# Table S19. Counting-threshold robustness under fixed, validation-calibrated, and diagnostic test thresholds

This source table reports three counting protocols:

1. `fixed_conf_0.25`: a fixed confidence threshold used for diagnostic consistency.
2. `validation_calibrated`: the formal counting protocol, where the threshold is selected on validation data and applied to the held-out test set.
3. `test_diagnostic_best`: a diagnostic sensitivity check, where the threshold is selected directly on the test set and is not used for model selection.

## Main interpretation

At fixed confidence 0.25, the 1280-pixel model has the lowest MAE but all resolutions show positive counting bias. Under the formal validation-calibrated protocol, 800 pixels has the lowest test MAE and the smallest absolute total-count bias. The diagnostic test sweep shows that 800 and 1280 pixels are close, supporting a protocol-specific interpretation rather than a universal resolution claim.
