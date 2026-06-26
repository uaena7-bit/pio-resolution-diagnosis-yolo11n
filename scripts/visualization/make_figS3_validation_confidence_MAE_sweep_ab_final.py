# -*- coding: utf-8 -*-
"""
make_figS3_validation_confidence_MAE_sweep_ab_final.py

Final version for Supplementary Figure S3.

Design:
- Panel (a): full-range validation confidence-threshold MAE sweep.
- Panel (b): zoomed low-error region.
- 1024 px is included as an intermediate-resolution sensitivity check.
- Panel labels (a) and (b) are aligned.
- Panel (b) label is moved slightly upward.
- Extra top whitespace prevents panel-label clipping.
- Summary box in panel (a) is moved left and does not cover the 0.45 / 0.50 dashed lines.

Final layout adjustment:
- top = 0.935: moderate whitespace above panel (a)
- right = 0.965: slightly increased right-side whitespace

Outputs:
- FigS3_validation_confidence_MAE_sweep_ab.pdf
- FigS3_validation_confidence_MAE_sweep_ab.svg
- FigS3_validation_confidence_MAE_sweep_ab.png
- FigS3_validation_confidence_MAE_sweep_ab.tif
- FigS3_validation_confidence_MAE_sweep_ab_source_data.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


DPI = 600

FIG_W_IN = 7.6
FIG_H_IN = 8.8

RESOLUTIONS = [800, 960, 1024, 1280]

COLORS = {
    "800 px": "#1f77b4",
    "960 px": "#ff7f0e",
    "1024 px": "#d62728",
    "1280 px": "#2ca02c",
}

MARKERS = {
    "800 px": "o",
    "960 px": "s",
    "1024 px": "D",
    "1280 px": "^",
}

SERIES = [
    ("MAE_800px", "800 px"),
    ("MAE_960px", "960 px"),
    ("MAE_1024px", "1024 px"),
    ("MAE_1280px", "1280 px"),
]


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/figS3_ab_final")


def default_input_csv() -> Path:
    if os.name == "nt":
        return Path(
            r"D:\Broiler chicken detection dataset\03_RESULTS\counting_val_calibrated_conf_3res"
            r"\experiment2_counting_threshold_robustness_4res\TableS18_validation_confidence_sweep.csv"
        )
    return Path("/mnt/data/TableS18_validation_confidence_sweep.csv")


def fallback_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "confidence_threshold": [
                0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
            ],
            "MAE_800px": [
                75.5, 54.5, 42.9, 34.0, 26.6, 20.2, 15.8,
                12.26, 11.20, 11.37, 13.0, 18.0, 25.5, 38.8
            ],
            "MAE_960px": [
                84.4, 56.5, 42.0, 33.2, 25.8, 20.3, 16.0,
                13.60, 12.24, 11.84, 12.1, 15.3, 21.1, 31.6
            ],
            "MAE_1024px": [
                74.8, 52.5, 40.8, 32.2, 25.5, 19.9, 15.7,
                13.10, 11.62, 11.19, 11.45, 15.0, 20.5, 30.3
            ],
            "MAE_1280px": [
                70.1, 47.0, 35.6, 28.3, 22.5, 18.6, 14.8,
                12.83, 11.43, 11.16, 11.6, 13.8, 18.1, 24.8
            ],
        }
    )


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Cannot find {label} column. "
        f"Candidates={candidates}. Available columns={list(df.columns)}"
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Supports two input formats.

    Wide format:
        confidence_threshold, MAE_800px, MAE_960px, MAE_1024px, MAE_1280px

    Long format:
        resolution, confidence_threshold/conf/threshold, MAE
    """
    if not csv_path.exists():
        print(f"[WARN] Input CSV not found: {csv_path}")
        print("[WARN] Using fallback data.")
        return fallback_data()

    raw = pd.read_csv(csv_path)

    required_wide = [
        "confidence_threshold",
        "MAE_800px",
        "MAE_960px",
        "MAE_1024px",
        "MAE_1280px",
    ]

    if all(col in raw.columns for col in required_wide):
        df = raw[required_wide].copy()
        df["confidence_threshold"] = df["confidence_threshold"].astype(float)
        return df.sort_values("confidence_threshold").reset_index(drop=True)

    res_col = pick_column(
        raw,
        ["resolution", "input_resolution_px", "imgsz", "res"],
        "resolution",
    )

    conf_col = pick_column(
        raw,
        ["confidence_threshold", "conf_threshold", "conf", "threshold", "selected_conf"],
        "confidence threshold",
    )

    mae_col = pick_column(
        raw,
        ["MAE", "mae", "validation_MAE", "val_MAE", "val_mae"],
        "MAE",
    )

    work = raw[[res_col, conf_col, mae_col]].copy()
    work.columns = ["resolution", "confidence_threshold", "MAE"]

    work["resolution"] = work["resolution"].astype(int)
    work["confidence_threshold"] = work["confidence_threshold"].astype(float)
    work["MAE"] = work["MAE"].astype(float)

    work = work[work["resolution"].isin(RESOLUTIONS)].copy()

    if work.empty:
        raise ValueError(
            f"No rows found for expected resolutions {RESOLUTIONS}. "
            f"Check input CSV: {csv_path}"
        )

    wide = (
        work.pivot_table(
            index="confidence_threshold",
            columns="resolution",
            values="MAE",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("confidence_threshold")
    )

    wide.columns = [
        "confidence_threshold" if col == "confidence_threshold" else f"MAE_{int(col)}px"
        for col in wide.columns
    ]

    missing = [col for col in required_wide if col not in wide.columns]
    if missing:
        raise ValueError(
            f"Missing columns after converting long table to wide table: {missing}. "
            f"Available columns={list(wide.columns)}"
        )

    return wide[required_wide].reset_index(drop=True)


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
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
        }
    )


def get_minima(df: pd.DataFrame) -> dict:
    minima = {}
    for col, label in SERIES:
        idx = df[col].idxmin()
        minima[label] = {
            "x": float(df.loc[idx, "confidence_threshold"]),
            "y": float(df.loc[idx, col]),
        }
    return minima


def draw_panel_a(ax, df: pd.DataFrame, minima: dict) -> None:
    x = df["confidence_threshold"]

    for col, label in SERIES:
        ax.plot(
            x,
            df[col],
            marker=MARKERS[label],
            linewidth=1.5,
            markersize=5.0,
            color=COLORS[label],
            label=label,
            zorder=3,
        )

    ax.set_xlim(0.05, 0.70)
    ax.set_ylim(10, 96)

    ax.set_xlabel("Confidence threshold", fontsize=12)
    ax.set_ylabel("Validation counting MAE", fontsize=12)

    ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])

    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3.2)
    ax.grid(True, alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    for xpos in [0.45, 0.50]:
        ax.axvline(
            xpos,
            linestyle="--",
            linewidth=0.9,
            color="0.45",
            alpha=0.50,
            zorder=1,
        )

    for label in ["800 px", "960 px", "1024 px", "1280 px"]:
        item = minima[label]
        ax.plot(
            item["x"],
            item["y"],
            marker=MARKERS[label],
            markersize=13,
            markerfacecolor=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="none",
            zorder=5,
        )

    summary = "\n".join(
        [
            f"800 px: min MAE = {minima['800 px']['y']:.2f} at conf = {minima['800 px']['x']:.2f}",
            f"960 px: min MAE = {minima['960 px']['y']:.2f} at conf = {minima['960 px']['x']:.2f}",
            f"1024 px: min MAE = {minima['1024 px']['y']:.2f} at conf = {minima['1024 px']['x']:.2f}",
            f"1280 px: min MAE = {minima['1280 px']['y']:.2f} at conf = {minima['1280 px']['x']:.2f}",
        ]
    )

    ax.text(
        0.10,
        91.2,
        summary,
        fontsize=9.2,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.7,
        ),
        zorder=8,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )

    ax.text(
        -0.045,
        1.035,
        "(a)",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )


def draw_panel_b(ax, df: pd.DataFrame, minima: dict) -> None:
    x = df["confidence_threshold"]

    for col, label in SERIES:
        ax.plot(
            x,
            df[col],
            marker=MARKERS[label],
            linewidth=1.5,
            markersize=5.0,
            color=COLORS[label],
            zorder=3,
        )

    ax.set_xlim(0.40, 0.54)
    ax.set_ylim(10.95, 14.05)

    ax.set_xlabel("Confidence threshold", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)

    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3.2)
    ax.grid(True, alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for xpos in [0.45, 0.50]:
        ax.axvline(
            xpos,
            linestyle="--",
            linewidth=0.9,
            color="0.45",
            alpha=0.50,
            zorder=1,
        )
        ax.text(
            xpos,
            14.02,
            f"{xpos:.2f}",
            fontsize=10,
            ha="center",
            va="top",
            color="0.4",
        )

    for label in ["800 px", "960 px", "1024 px", "1280 px"]:
        item = minima[label]
        ax.plot(
            item["x"],
            item["y"],
            marker=MARKERS[label],
            markersize=13,
            markerfacecolor=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="none",
            zorder=5,
        )

    ann_specs = {
        "800 px": dict(dx=0.004, dy=0.08),
        "960 px": dict(dx=0.006, dy=0.30),
        "1024 px": dict(dx=0.006, dy=0.05),
        "1280 px": dict(dx=0.006, dy=-0.16),
    }

    for label in ["800 px", "960 px", "1024 px", "1280 px"]:
        item = minima[label]
        spec = ann_specs[label]

        ax.annotate(
            f"{item['y']:.2f}",
            xy=(item["x"], item["y"]),
            xytext=(item["x"] + spec["dx"], item["y"] + spec["dy"]),
            textcoords="data",
            fontsize=9,
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.10",
                facecolor="white",
                edgecolor=COLORS[label],
                linewidth=0.7,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=COLORS[label],
                linewidth=0.8,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=6,
        )

    ax.text(
        -0.045,
        1.035,
        "(b)",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )


def draw_figure(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    setup_style()

    minima = get_minima(df)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIG_W_IN, FIG_H_IN),
        dpi=DPI,
    )

    draw_panel_a(axes[0], df, minima)
    draw_panel_b(axes[1], df, minima)

    # Final balanced layout:
    # - top=0.935 keeps moderate whitespace above panel (a)
    # - right=0.965 increases the right-side whitespace slightly
    fig.subplots_adjust(
        left=0.10,
        right=0.965,
        top=0.935,
        bottom=0.08,
        hspace=0.30,
    )

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

    print("[DONE] Saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)
    print(csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=str(default_input_csv()))
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument(
        "--stem",
        type=str,
        default="FigS3_validation_confidence_MAE_sweep_ab",
    )

    args = parser.parse_args()

    df = load_data(Path(args.input_csv))
    draw_figure(df, Path(args.out_dir), args.stem)


if __name__ == "__main__":
    main()
