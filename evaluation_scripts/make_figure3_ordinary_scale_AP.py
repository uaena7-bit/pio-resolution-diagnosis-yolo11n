#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproducible figure-generation script for the CSSP submission.

The script reads values from CSV files in figure_source_data/.
No AP values are hard-coded inside the plotting logic.
Raw PIO images are not redistributed in this repository.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=repo_root() / "figure_source_data" / "Fig3_locked_ordinary_scale_AP_source_data.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    groups = ["tiny", "small", "medium", "large"]
    required = {"resolution_px", *groups}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.source}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    x = df["resolution_px"].tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.9), dpi=args.dpi)
    handles = []
    for group in groups:
        h, = ax.plot(x, df[group].tolist(), marker="o", linewidth=2, markersize=6, label=group)
        handles.append(h)

    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Scale-stratified AP")
    ax.set_xticks(x)
    ax.set_ylim(0.44, 0.80)
    ax.set_yticks([0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.legend(handles=handles, labels=groups, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    png = args.out_dir / "Fig3_ordinary_scale_AP.png"
    pdf = args.out_dir / "Fig3_ordinary_scale_AP.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
