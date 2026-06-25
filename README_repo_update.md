# Repository update package for revision analyses

This package contains the repository-side source data and scripts corresponding to the newly inserted manuscript and supplementary results.

## What this package updates

- Table S18: validation-source-component sensitivity of counting-threshold calibration.
- Table S19: counting-threshold robustness under fixed, validation-calibrated, and diagnostic test thresholds.
- Reproduction scripts for both analyses.

## Suggested target locations

Copy the files from this package into the repository root while preserving folder names:

```text
supplementary_source_data/
scripts/revision_experiments/
release_notes/
```

## Suggested README note

```text
Revision update: Tables S18 and S19 were added to document validation-source-component sensitivity and counting-threshold robustness. The corresponding source CSV files and reproduction scripts are provided in supplementary_source_data/ and scripts/revision_experiments/.
```

## Local copy command example

From PowerShell, after extracting the package:

```powershell
Copy-Item -Path ".\supplementary_source_data\*" -Destination "D:\Broiler chicken detection dataset\pio-resolution-diagnosis-yolo11n\supplementary_source_data" -Recurse -Force
Copy-Item -Path ".\scriptsevision_experiments\*" -Destination "D:\Broiler chicken detection dataset\pio-resolution-diagnosis-yolo11n\scriptsevision_experiments" -Recurse -Force
Copy-Item -Path ".elease_notes\*" -Destination "D:\Broiler chicken detection dataset\pio-resolution-diagnosis-yolo11nelease_notes" -Recurse -Force
```
