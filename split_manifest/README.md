# Split manifest

This directory contains the fixed leakage-controlled PIO split used in the CSSP manuscript.

## Files

- `manifest.csv`: image-level split manifest for the frozen PIO-GRDB-MD5-7_1_2 partition.
- `dataset_split_statistics.csv`: split-level image and instance statistics.
- `split_stats.csv` / `split_stats.json`: machine-readable split statistics used by the reproducibility scripts.

## Notes

The PIO folder prefixes inherited from the original dataset, such as `train__` and `val__`, do not define the split assignment used in this study. The authoritative split assignment is the fixed manifest in this directory.

The split was constructed after exact-duplicate removal and source-group binding. Source-related images were kept within a single split during grouped allocation, and the final split was audited using source path, filename, image-content MD5, source-group identifiers, and near-duplicate screening.
