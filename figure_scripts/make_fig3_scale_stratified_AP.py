
# -*- coding: utf-8 -*-
"""
make_fig3_scale_stratified_AP_PAstyle.py

Fig. 3 redrawn in a Precision Agriculture-friendly style.

Source basis:
- recreated from the currently used Fig.3 trend values supplied by the user
- the figure is intended to match the manuscript plotting style used for Fig.2

Outputs:
- PDF / PNG / SVG / TIFF
- source-data CSV
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
FIG_H_MM = 100.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/fig3_scale_stratified_AP_PAstyle_preview")


def make_data() -> pd.DataFrame:
    # Reconstructed from the user's current Fig.3 panel.
    # If the user later provides the original numeric table/CSV,
    # these values can be replaced directly without changing plotting code.
    return pd.DataFrame(
        {
            "input_resolution_px": [800, 960, 1280],
            "tiny":   [0.478, 0.516, 0.524],
            "small":  [0.592, 0.626, 0.634],
            "medium": [0.697, 0.716, 0.724],
            "large":  [0.758, 0.763, 0.787],
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

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)

    x = df["input_resolution_px"].tolist()

    ax.plot(x, df["tiny"], marker="o", linewidth=1.25, markersize=4.7, label="tiny")
    ax.plot(x, df["small"], marker="s", linewidth=1.25, markersize=4.7, label="small")
    ax.plot(x, df["medium"], marker="^", linewidth=1.25, markersize=4.9, label="medium")
    ax.plot(x, df["large"], marker="D", linewidth=1.25, markersize=4.5, label="large")

    ax.set_xlabel("Input resolution (px)", fontsize=9.5)
    ax.set_ylabel("Scale-stratified AP", fontsize=9.5)

    ax.set_xlim(775, 1305)
    ax.set_xticks([800, 960, 1280])
    ax.tick_params(axis="x", labelsize=8.7, width=0.8, length=3.2)
    ax.tick_params(axis="y", labelsize=8.7, width=0.8, length=3.2)

    ax.set_ylim(0.45, 0.81)
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])

    ax.grid(True, axis="both", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=4,
        frameon=False,
        fontsize=8.6,
        handlelength=2.1,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.24, top=0.97)

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

    print("[Fig3 PA style] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="Fig3_scale_stratified_AP")
    args = parser.parse_args()

    df = make_data()
    draw_figure(df, Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
