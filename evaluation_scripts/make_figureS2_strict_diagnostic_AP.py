#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Figure S2: strict global-first diagnostic AP by scale and density group.

The plotted values are strict diagnostic AP values and should not be directly
compared with ordinary subgroup AP or standard global AP.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def plot_panel(ax, sub, group_order, title):
    for group in group_order:
        g = sub[sub["group"] == group].sort_values("resolution_px")
        ax.plot(g["resolution_px"], g["strict_diagnostic_AP"], marker="o", linewidth=2, markersize=5, label=group)
    ax.set_title(title)
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Strict diagnostic AP")
    ax.set_xticks([800, 960, 1280])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=repo_root() / "supplementary_source_data" / "FigureS2_strict_diagnostic_AP_source_data.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    required = {"group_type","group","resolution_px","strict_diagnostic_AP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=args.dpi)
    plot_panel(axes[0], df[df["group_type"] == "scale"], ["tiny","small","medium","large"], "(a) Scale groups")
    plot_panel(axes[1], df[df["group_type"] == "density"], ["low","medium","high","ultra_high"], "(b) Density groups")
    fig.tight_layout()

    png = args.out_dir / "FigureS2_strict_diagnostic_AP.png"
    pdf = args.out_dir / "FigureS2_strict_diagnostic_AP.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

if __name__ == "__main__":
    main()
