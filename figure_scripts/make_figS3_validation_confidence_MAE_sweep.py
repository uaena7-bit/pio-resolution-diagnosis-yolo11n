# -*- coding: utf-8 -*-
"""
make_figS3_validation_confidence_MAE_sweep_PAstyle_v6.py

Supplementary Fig. S3 in Precision Agriculture / Springer style.

This version uses the experimental MAE data from the current Fig. S3 source-data CSV.
Main adjustments vs v4:
1) enlarge the inset,
2) move the inset further left to use blank space,
3) keep the inset fully inside the main panel,
4) avoid covering the 0.45 dashed line,
5) keep highlight colors consistent with the three main curves.

Outputs:
- FigS3_validation_confidence_MAE_sweep.pdf
- FigS3_validation_confidence_MAE_sweep.svg
- FigS3_validation_confidence_MAE_sweep.png
- FigS3_validation_confidence_MAE_sweep.tif
- FigS3_validation_confidence_MAE_sweep_source_data.csv
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
FIG_H_MM = 118.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4

COLORS = {
    "800 px": "#1f77b4",
    "960 px": "#ff7f0e",
    "1280 px": "#2ca02c",
}


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/figS3_PAstyle_v6_preview")


def default_input_csv() -> Path:
    # Experimental source data used in the current figure pipeline.
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures\FigS3_validation_confidence_MAE_sweep_source_data.csv")
    return Path("/mnt/data/figS3_PAstyle_v4_preview/FigS3_validation_confidence_MAE_sweep_source_data.csv")


def fallback_data() -> pd.DataFrame:
    # Same experimental values used in the current Fig. S3 source-data file.
    return pd.DataFrame(
        {
            "confidence_threshold": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            "MAE_800px":  [75.5, 54.5, 42.9, 34.0, 26.6, 20.2, 15.8, 12.26, 11.20, 11.37, 13.0, 18.0, 25.5, 38.8],
            "MAE_960px":  [84.4, 56.5, 42.0, 33.2, 25.8, 20.3, 16.0, 13.60, 12.24, 11.84, 12.1, 15.3, 21.1, 31.6],
            "MAE_1280px": [70.1, 47.0, 35.6, 28.3, 22.5, 18.6, 14.8, 12.83, 11.43, 11.16, 11.6, 13.8, 18.1, 24.8],
        }
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = fallback_data()
    required = ["confidence_threshold", "MAE_800px", "MAE_960px", "MAE_1280px"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source data: {missing}")
    return df[required].copy()


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
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def draw_figure(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)

    series = [
        ("MAE_800px", "800 px", "o"),
        ("MAE_960px", "960 px", "s"),
        ("MAE_1280px", "1280 px", "^"),
    ]
    x = df["confidence_threshold"]

    for col, label, marker in series:
        ax.plot(
            x,
            df[col],
            marker=marker,
            linewidth=1.25,
            markersize=4.7,
            color=COLORS[label],
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Confidence threshold", fontsize=9.5)
    ax.set_ylabel("Validation counting MAE", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.7, width=0.8, length=3.2)
    ax.set_xlim(0.05, 0.70)
    ax.set_ylim(10, 86)
    ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8.6, handlelength=2.1)

    for xpos in [0.45, 0.50]:
        ax.axvline(xpos, linestyle="--", linewidth=0.8, alpha=0.55, color="0.45", zorder=1.5)

    minima = {
        "MAE_800px":  (0.45, 11.20, "800 px: min MAE = 11.20 at conf = 0.45", "800 px", "o"),
        "MAE_960px":  (0.50, 11.84, "960 px: min MAE = 11.84 at conf = 0.50", "960 px", "s"),
        "MAE_1280px": (0.50, 11.16, "1280 px: min MAE = 11.16 at conf = 0.50", "1280 px", "^"),
    }

    text_block = "\n".join(v[2] for v in minima.values())
    ax.text(
        0.062,
        11.1,
        text_block,
        fontsize=7.5,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.55", linewidth=0.6),
        zorder=8,
    )

    for _, (thr, val, _, label, marker) in minima.items():
        ax.plot(
            thr,
            val,
            marker=marker,
            markersize=8.0,
            markerfacecolor=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=7,
        )

    # Inset shifted slightly to the right while remaining fully inside the blank region.
    # Right edge stays left of the 0.45 dashed line in main-axis coordinates.
    # [x0, y0, width, height] are in axes fraction coordinates.
    axins = ax.inset_axes([0.185, 0.595, 0.395, 0.335])
    axins.set_facecolor("white")
    axins.patch.set_alpha(1.0)

    for col, label, marker in series:
        axins.plot(
            x,
            df[col],
            marker=marker,
            linewidth=1.05,
            markersize=3.6,
            color=COLORS[label],
        )

    for xpos in [0.45, 0.50]:
        axins.axvline(xpos, linestyle="--", linewidth=0.7, alpha=0.55, color="0.45")
        axins.text(xpos, 13.60, f"{xpos:.2f}", fontsize=6.7, ha="center", va="bottom", color="0.4")

    for _, (thr, val, _, label, marker) in minima.items():
        axins.plot(
            thr,
            val,
            marker=marker,
            markersize=5.3,
            markerfacecolor=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=0.6,
            linestyle="none",
            zorder=6,
        )

    axins.set_xlim(0.39, 0.53)
    axins.set_ylim(10.85, 13.80)
    axins.set_xlabel("Confidence threshold", fontsize=6.9)
    axins.set_ylabel("MAE", fontsize=6.9)
    axins.tick_params(axis="both", labelsize=6.35, width=0.6, length=2.0)
    axins.grid(True, axis="both", alpha=0.25, linewidth=0.35)

    label_specs = [
        (0.45, 11.20, "11.20", "800 px", 0.004, 0.08),
        (0.50, 11.84, "11.84", "960 px", 0.006, 0.11),
        (0.50, 11.16, "11.16", "1280 px", 0.006, -0.23),
    ]
    for thr, val, txt, label, dx, dy in label_specs:
        axins.text(
            thr + dx,
            val + dy,
            txt,
            fontsize=6.7,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=COLORS[label], linewidth=0.6),
            zorder=7,
        )

    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.145)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"
    png = out_dir / f"{stem}.png"
    tif = out_dir / f"{stem}.tif"
    csv = out_dir / f"{stem}_source_data.csv"

    df.to_csv(csv, index=False, encoding="utf-8")
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=DPI)
    fig.savefig(tif, dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("[FigS3 PA style v6] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=str(default_input_csv()))
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="FigS3_validation_confidence_MAE_sweep")
    args = parser.parse_args()

    df = load_data(Path(args.input_csv))
    draw_figure(df, Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
