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
                        default=repo_root() / "figure_source_data" / "Fig2_accuracy_cost_tradeoff_source_data.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    required = {"resolution_px", "standard_test_mAP50_95", "training_time_h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.source}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(df)))
    labels = df["resolution_px"].astype(str).tolist()
    train_time = df["training_time_h"].tolist()
    map5095 = df["standard_test_mAP50_95"].tolist()

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=args.dpi)
    ax2 = ax1.twinx()

    bars = ax1.bar(
        x, train_time, width=0.42,
        facecolor="0.90", edgecolor="0.25", linewidth=1.0,
        hatch="///", label="Training time", zorder=2
    )
    line, = ax2.plot(
        x, map5095,
        marker="o", markersize=6, linewidth=2,
        markerfacecolor="white", markeredgewidth=1.2,
        label="Test mAP50-95", zorder=3
    )

    for xi, bh in zip(x, train_time):
        ax1.text(xi, bh + 0.55, f"{bh:.2f} h", ha="center", va="bottom", fontsize=8)
    for xi, my in zip(x, map5095):
        ax2.text(xi, my + 0.0012, f"{my:.4f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Input resolution (px)")
    ax1.set_ylabel("Training time (h)")
    ax2.set_ylabel("Test mAP50-95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 30)
    ax1.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax2.set_ylim(0.70, 0.76)
    ax2.set_yticks([0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76])
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=1)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    ax1.legend([bars, line], ["Training time", "Test mAP50-95"], loc="upper left", frameon=False)
    fig.tight_layout()

    png = args.out_dir / "Fig2_accuracy_cost_tradeoff.png"
    pdf = args.out_dir / "Fig2_accuracy_cost_tradeoff.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
