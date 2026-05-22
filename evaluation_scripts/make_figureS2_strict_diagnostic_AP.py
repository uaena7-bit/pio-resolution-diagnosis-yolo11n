#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Supplementary Figure S2 in the final CSSP submission style.

Figure S2 shows strict global-first diagnostic AP by:
(a) target scale group and
(b) image density group.

Important:
- The underlying strict diagnostic AP values are unchanged.
- This script updates only the visual presentation to match the main-text
  Fig. 3 line-chart style.
- Strict diagnostic AP should not be directly compared with ordinary subgroup
  AP in Figs. 3 and 4 or with standard global AP.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def plot_panel(ax, data, order, panel_label, ylim):
    """Plot one Figure S2 panel using the locked strict diagnostic AP values."""
    for group in order:
        sub = data[data["group"] == group].sort_values("resolution_px")
        ax.plot(
            sub["resolution_px"],
            sub["strict_diagnostic_AP"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=group,
        )

    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Strict diagnostic AP (mAP50-95)")
    ax.set_xticks([800, 960, 1280])
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.text(-0.085, 1.035, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=len(order),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root() / "supplementary_source_data" / "FigureS2_strict_diagnostic_AP_source_data.csv",
        help="CSV containing locked strict diagnostic AP values.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "supplementary_figures",
        help="Directory for exported Figure S2 files.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    required = {"group_type", "group", "resolution_px", "strict_diagnostic_AP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.source}: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 10.0), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        ax.tick_params(width=0.9)

    plot_panel(
        axes[0],
        df[df["group_type"] == "scale"],
        ["tiny", "small", "medium", "large"],
        "(a)",
        (0.10, 0.47),
    )
    plot_panel(
        axes[1],
        df[df["group_type"] == "density"],
        ["low", "medium", "high", "ultra-high"],
        "(b)",
        (0.06, 0.33),
    )

    fig.tight_layout(rect=[0.04, 0.03, 0.985, 0.985], h_pad=2.3)

    base = args.out_dir / "FigureS2_strict_diagnostic_AP_line_style"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=args.dpi)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tif"), bbox_inches="tight", dpi=args.dpi)
    print(f"Saved: {base.with_suffix('.png')}")
    print(f"Saved: {base.with_suffix('.pdf')}")
    print(f"Saved: {base.with_suffix('.tif')}")


if __name__ == "__main__":
    main()
