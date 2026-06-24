
# -*- coding: utf-8 -*-
"""
make_figS2_strict_diagnostic_AP_PAstyle_v5.py

Supplementary Fig. S2 in Precision Agriculture / Springer style.

v6 adjustments:
1) Keep the reduced vertical gap between panel (a) and panel (b).
2) Add more bottom whitespace so the lower legend is not too close to the edge.
3) Keep panel labels fully visible and close to the plotting area.
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
FIG_H_MM = 182.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4

COLORS_SCALE = {
    "tiny": "#1f77b4",
    "small": "#ff7f0e",
    "medium": "#2ca02c",
    "large": "#d62728",
}
MARKERS_SCALE = {
    "tiny": "o",
    "small": "s",
    "medium": "^",
    "large": "D",
}

COLORS_DENSITY = {
    "low": "#1f77b4",
    "medium": "#ff7f0e",
    "high": "#2ca02c",
    "ultra-high": "#d62728",
}
MARKERS_DENSITY = {
    "low": "o",
    "medium": "s",
    "high": "^",
    "ultra-high": "D",
}


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/figS2_PAstyle_v6_preview")


def make_scale_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_resolution_px": [800, 960, 1280],
            "tiny": [0.124, 0.136, 0.142],
            "small": [0.161, 0.168, 0.174],
            "medium": [0.250, 0.270, 0.252],
            "large": [0.397, 0.358, 0.448],
        }
    )


def make_density_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_resolution_px": [800, 960, 1280],
            "low": [0.086, 0.076, 0.075],
            "medium": [0.147, 0.134, 0.146],
            "high": [0.230, 0.262, 0.259],
            "ultra-high": [0.294, 0.316, 0.316],
        }
    )


def setup_style() -> None:
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


def draw_panel(ax, df, series_names, colors, markers, panel_label):
    x = df["input_resolution_px"]

    for name in series_names:
        ax.plot(
            x,
            df[name],
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=1.25,
            markersize=4.8,
            zorder=3,
        )

    ax.set_xlim(760, 1300)
    ax.set_xticks([800, 960, 1280])
    ax.tick_params(axis="both", labelsize=8.6, width=0.8, length=3.0)
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)

    ax.text(
        -0.045, 1.01, panel_label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False
    )


def draw_figure(scale_df: pd.DataFrame, density_df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    setup_style()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_W_IN, FIG_H_IN),
        dpi=DPI,
        sharex=False
    )

    draw_panel(
        ax1, scale_df,
        ["tiny", "small", "medium", "large"],
        COLORS_SCALE, MARKERS_SCALE, "(a)"
    )
    ax1.set_ylabel("Strict diagnostic AP (mAP50-95)", fontsize=9.5)
    ax1.set_xlabel("Input resolution (px)", fontsize=9.5)
    ax1.set_ylim(0.10, 0.47)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
    )

    draw_panel(
        ax2, density_df,
        ["low", "medium", "high", "ultra-high"],
        COLORS_DENSITY, MARKERS_DENSITY, "(b)"
    )
    ax2.set_ylabel("Strict diagnostic AP (mAP50-95)", fontsize=9.5)
    ax2.set_xlabel("Input resolution (px)", fontsize=9.5)
    ax2.set_ylim(0.06, 0.34)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.175),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.0,
    )

    # Compared with v5: keep panel spacing close, but add more bottom whitespace.
    fig.subplots_adjust(left=0.12, right=0.985, top=0.955, bottom=0.13, hspace=0.42)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"
    png = out_dir / f"{stem}.png"
    tif = out_dir / f"{stem}.tif"
    csv_scale = out_dir / f"{stem}_scale_source_data.csv"
    csv_density = out_dir / f"{stem}_density_source_data.csv"

    scale_df.to_csv(csv_scale, index=False, encoding="utf-8")
    density_df.to_csv(csv_density, index=False, encoding="utf-8")

    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=DPI)
    fig.savefig(tif, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("[FigS2 PA style v6] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv_scale)
    print(csv_density)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="FigS2_strict_diagnostic_AP")
    args = parser.parse_args()

    draw_figure(make_scale_df(), make_density_df(), Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
