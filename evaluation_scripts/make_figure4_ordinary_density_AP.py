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
                        default=repo_root() / "figure_source_data" / "Fig4_locked_ordinary_density_AP_source_data.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    # Column name uses ultra_high in CSV; legend uses ultra-high in the figure.
    groups = [("low", "low"), ("medium", "medium"), ("high", "high"), ("ultra_high", "ultra-high")]
    required = {"resolution_px", *(g[0] for g in groups)}
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

    fig = plt.figure(figsize=(7.2, 5.8), dpi=args.dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.3], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    handles = []
    labels = []
    for col, label in groups:
        y = df[col].tolist()
        h, = ax_top.plot(x, y, marker="o", linewidth=2, markersize=6, label=label)
        ax_bot.plot(x, y, marker="o", linewidth=2, markersize=6)
        handles.append(h)
        labels.append(label)

    ax_top.set_ylim(0.726, 0.7725)
    ax_bot.set_ylim(0.642, 0.661)
    ax_bot.set_xticks(x)

    ax_top.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_bot.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bot.xaxis.tick_bottom()

    for ax in (ax_top, ax_bot):
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(1.0)

    # Diagonal break marks, drawn only at the left and right edges.
    kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-0.008, +0.008), (-0.015, +0.015), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((0.992, 1.008), (-0.015, +0.015), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-0.008, +0.008), (0.985, 1.015), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((0.992, 1.008), (0.985, 1.015), transform=ax_bot.transAxes, **kwargs)

    fig.text(0.03, 0.53, "Density-stratified AP", va="center", rotation="vertical")
    fig.supxlabel("Input resolution (px)", y=0.09)
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.17)

    png = args.out_dir / "Fig4_ordinary_density_AP.png"
    pdf = args.out_dir / "Fig4_ordinary_density_AP.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
