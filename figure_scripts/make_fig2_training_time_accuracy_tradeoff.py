
# -*- coding: utf-8 -*-
"""
make_fig2_training_time_accuracy_tradeoff_PAstyle_v2.py

Updated from the PA-style version:
- move the mAP annotation labels slightly upward
- especially lift the first label (0.7199) more so it does not overlap the line
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
FIG_H_MM = 92.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/fig2_training_time_accuracy_tradeoff_PAstyle_v2_preview")


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_resolution_px": [800, 960, 1280],
            "training_time_h": [4.48, 7.29, 22.54],
            "test_mAP50_95": [0.7199, 0.7420, 0.7543],
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
            "hatch.linewidth": 0.7,
        }
    )

    fig, ax_time = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax_acc = ax_time.twinx()

    x = list(range(len(df)))
    xlabels = df["input_resolution_px"].astype(str).tolist()

    bars = ax_time.bar(
        x,
        df["training_time_h"],
        width=0.44,
        fill=False,
        hatch="///",
        linewidth=0.8,
        label="Training time",
        zorder=2,
    )

    ax_acc.plot(
        x,
        df["test_mAP50_95"],
        marker="o",
        linewidth=1.25,
        markersize=4.7,
        label="Test mAP50-95",
        zorder=4,
    )

    ax_time.set_xlabel("Input resolution (px)", fontsize=9.5)
    ax_time.set_ylabel("Training time (h)", fontsize=9.5)
    ax_acc.set_ylabel("Test mAP50-95", fontsize=9.5)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(xlabels, fontsize=8.7)
    ax_time.tick_params(axis="x", labelsize=8.7, width=0.8, length=3.2)
    ax_time.tick_params(axis="y", labelsize=8.7, width=0.8, length=3.2)
    ax_acc.tick_params(axis="y", labelsize=8.7, width=0.8, length=3.2)

    ax_time.set_ylim(0, 30)
    ax_time.set_yticks([0, 5, 10, 15, 20, 25, 30])

    ax_acc.set_ylim(0.70, 0.76)
    ax_acc.set_yticks([0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76])

    ax_time.grid(True, axis="y", alpha=0.25, linewidth=0.4)
    ax_time.set_axisbelow(True)

    # Bar labels
    for i, row in df.iterrows():
        ax_time.text(
            i,
            row["training_time_h"] + 0.45,
            f"{row['training_time_h']:.2f} h",
            ha="center",
            va="bottom",
            fontsize=8.3,
        )

    # Lift the line labels upward; first point gets extra offset
    label_offsets = [0.0030, 0.0016, 0.0016]
    for i, row in df.iterrows():
        ax_acc.text(
            i,
            row["test_mAP50_95"] + label_offsets[i],
            f"{row['test_mAP50_95']:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
        )

    handles_1, labels_1 = ax_time.get_legend_handles_labels()
    handles_2, labels_2 = ax_acc.get_legend_handles_labels()
    ax_time.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=False,
        fontsize=8.6,
        handlelength=1.8,
        borderaxespad=0.5,
        labelspacing=0.35,
    )

    fig.subplots_adjust(left=0.085, right=0.915, bottom=0.155, top=0.945)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    tif = out_dir / f"{stem}.tif"
    svg = out_dir / f"{stem}.svg"
    csv = out_dir / f"{stem}_source_data.csv"

    df.to_csv(csv, index=False, encoding="utf-8")
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=DPI)
    fig.savefig(tif, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("[Fig2 PA style v2] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="Fig2_training_time_accuracy_tradeoff")
    args = parser.parse_args()

    df = make_data()
    draw_figure(df, Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
