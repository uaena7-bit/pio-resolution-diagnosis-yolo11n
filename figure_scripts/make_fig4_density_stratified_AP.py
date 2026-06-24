
# -*- coding: utf-8 -*-
"""
make_fig4_density_stratified_AP_PAstyle_v2.py

Fig. 4 redrawn in a Precision Agriculture-friendly style.

v2 change:
- Explicitly assign distinct colors across the broken-axis subplots.
- This avoids low and medium using the same default Matplotlib color
  when plotted on separate axes.

Category color mapping:
low        -> blue
medium     -> orange
high       -> green
ultra-high -> red
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


DPI = 600
FIG_W_MM = 174.0
FIG_H_MM = 140.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/fig4_density_stratified_AP_PAstyle_v2_preview")


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_resolution_px": [800, 960, 1280],
            "low":        [0.6447, 0.6545, 0.6583],
            "medium":     [0.7408, 0.7402, 0.7585],
            "high":       [0.7343, 0.7516, 0.7679],
            "ultra_high": [0.7291, 0.7570, 0.7703],
        }
    )


def draw_figure(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    colors = {
        "low": "#1f77b4",
        "medium": "#ff7f0e",
        "high": "#2ca02c",
        "ultra-high": "#d62728",
    }

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.04}
    )

    x = df["input_resolution_px"].tolist()

    l1, = ax_bottom.plot(
        x, df["low"], marker="o", color=colors["low"],
        linewidth=1.25, markersize=4.7, label="low"
    )
    l2, = ax_top.plot(
        x, df["medium"], marker="s", color=colors["medium"],
        linewidth=1.25, markersize=4.7, label="medium"
    )
    l3, = ax_top.plot(
        x, df["high"], marker="^", color=colors["high"],
        linewidth=1.25, markersize=4.9, label="high"
    )
    l4, = ax_top.plot(
        x, df["ultra_high"], marker="D", color=colors["ultra-high"],
        linewidth=1.25, markersize=4.5, label="ultra-high"
    )

    ax_top.set_ylim(0.724, 0.773)
    ax_bottom.set_ylim(0.642, 0.661)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, axis="both", alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.7, width=0.8, length=3.2)
        ax.set_xlim(775, 1305)
        ax.set_xticks([800, 960, 1280])

    ax_top.set_yticks([0.73, 0.74, 0.75, 0.76, 0.77])
    ax_bottom.set_yticks([0.645, 0.650, 0.655, 0.660])

    ax_bottom.set_xlabel("Input resolution (px)", fontsize=9.5)
    fig.supylabel("Density-stratified AP", fontsize=9.5, x=0.04)

    # Broken-axis styling
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.008
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.legend(
        handles=[l1, l2, l3, l4],
        labels=["low", "medium", "high", "ultra-high"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=8.6,
        handlelength=2.1,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.13, right=0.985, top=0.975, bottom=0.15)

    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    tif = out_dir / f"{stem}.tif"
    csv = out_dir / f"{stem}_source_data.csv"

    df.to_csv(csv, index=False, encoding="utf-8")
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=DPI)
    fig.savefig(tif, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("[Fig4 PA style v2] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="Fig4_density_stratified_AP")
    args = parser.parse_args()

    df = make_data()
    draw_figure(df, Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
