"""
Generate a GitHub-only continuous density diagnostic figure.

This script expects per-image diagnostic source data at:
    figure_source_data/FigureS4_continuous_density_diagnostic_source_data.csv

Required columns:
    image_id, filename, gt_instances, resolution_px,
    pred_count, abs_count_error, false_negatives, false_positives,
    true_positives, recall_iou50_conf025, precision_iou50_conf025, density_group

The generated outputs are written to:
    supplementary_figures/FigureS4_continuous_density_diagnostic.png
    supplementary_figures/FigureS4_continuous_density_diagnostic.pdf

This figure is a GitHub-only diagnostic aid and is not used to change the
locked split, primary AP results, counting-calibration values, or manuscript
conclusions.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figure_source_data" / "FigureS4_continuous_density_diagnostic_source_data.csv"
OUT_DIR = ROOT / "supplementary_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "image_id",
    "filename",
    "gt_instances",
    "resolution_px",
    "pred_count",
    "abs_count_error",
    "false_negatives",
    "false_positives",
    "true_positives",
    "recall_iou50_conf025",
    "precision_iou50_conf025",
    "density_group",
}


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(
            f"Missing source data: {SRC}\n"
            "Create this CSV using the released template: "
            "figure_source_data/FigureS4_continuous_density_diagnostic_source_data_TEMPLATE.csv"
        )

    df = pd.read_csv(SRC)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["gt_instances"] = pd.to_numeric(df["gt_instances"])
    df["resolution_px"] = pd.to_numeric(df["resolution_px"])
    df["abs_count_error"] = pd.to_numeric(df["abs_count_error"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for resolution, sub in sorted(df.groupby("resolution_px")):
        ax.scatter(
            sub["gt_instances"],
            sub["abs_count_error"],
            s=18,
            alpha=0.65,
            label=f"{int(resolution)} px",
        )

    ax.set_xlabel("GT instances per image")
    ax.set_ylabel("Absolute counting error per image")
    ax.set_title("GitHub-only continuous density diagnostic")
    ax.legend(title="Input resolution", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    fig.savefig(OUT_DIR / "FigureS4_continuous_density_diagnostic.png", dpi=300)
    fig.savefig(OUT_DIR / "FigureS4_continuous_density_diagnostic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
