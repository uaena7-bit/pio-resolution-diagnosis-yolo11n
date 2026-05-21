#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S1 low-/medium-density qualitative case metadata utility.

Raw PIO images and prediction overlays are not redistributed in this repository.
This script validates the metadata used for Fig. S1 and can export a compact
summary table for manuscript/review checking. Recreating the visual panels
requires local raw PIO images and prediction results.
"""
from pathlib import Path
import argparse
import pandas as pd

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path,
                        default=repo_root() / "figure_source_data" / "FigureS1_low_medium_no_gain_cases_metadata.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "figures")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    required = {"density_group","image_name","instances","metric_conf","draw_conf","resolution_px","TP","FN","FP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "FigureS1_low_medium_no_gain_cases_summary.csv"
    df.to_csv(out, index=False)
    print(f"Validated Fig. S1 metadata: {args.metadata}")
    print(f"Saved summary: {out}")
    print("Note: raw PIO images and model predictions must be provided locally to recreate visual panels.")

if __name__ == "__main__":
    main()
