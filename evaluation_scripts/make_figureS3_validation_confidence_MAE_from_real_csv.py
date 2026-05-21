#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figureS3_validation_confidence_MAE_from_real_csv.py

Purpose
-------
Generate Figure S3 directly from REAL validation prediction CSV files:
    predictions_val_800.csv
    predictions_val_960.csv
    predictions_val_1280.csv

It recomputes validation-set counting MAE across confidence thresholds from
the prediction CSVs and validation labels, then draws:
    - a continuous main plot
    - a local inset around the selected-threshold region
    - PNG / TIFF / SVG / PDF outputs

No image-generation or hand-drawn curve is used.

Default project layout
----------------------
D:\Broiler chicken detection dataset
  00_DATASET\PIO-GRDB-MD5-7_1_2\dataset.yaml
  03_RESULTS\counting_val_calibrated_conf_3res\predictions_val_800.csv
  03_RESULTS\counting_val_calibrated_conf_3res\predictions_val_960.csv
  03_RESULTS\counting_val_calibrated_conf_3res\predictions_val_1280.csv
  05_PAPER\supplementary_figures

Run
---
D:\ANACONDA\envs\yolo\python.exe ^
  "D:\Broiler chicken detection dataset\01_SCRIPTS\05_visualization\make_figureS3_validation_confidence_MAE_from_real_csv.py"

If your prediction CSV names differ, pass --pred800/--pred960/--pred1280 explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt


# -----------------------------
# Defaults
# -----------------------------
PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
DATA_YAML = PROJECT_ROOT / "00_DATASET" / "PIO-GRDB-MD5-7_1_2" / "dataset.yaml"
COUNT_DIR = PROJECT_ROOT / "03_RESULTS" / "counting_val_calibrated_conf_3res"
OUT_DIR = PROJECT_ROOT / "05_PAPER" / "supplementary_figures"

DEFAULT_PREDS = {
    800: COUNT_DIR / "predictions_val_800.csv",
    960: COUNT_DIR / "predictions_val_960.csv",
    1280: COUNT_DIR / "predictions_val_1280.csv",
}

CONF_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# -----------------------------
# Robust I/O helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(data_yaml: Path, split: str) -> Tuple[Path, List[Path]]:
    data = load_yaml(data_yaml)
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_entry = data.get(split)
    if split_entry is None:
        raise RuntimeError(f"Split '{split}' not found in {data_yaml}")

    split_path = Path(split_entry)
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()

    # YOLO yaml may point to an image folder or a text file listing images.
    if split_path.is_file():
        image_paths = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (root / p).resolve()
            image_paths.append(p)
    else:
        image_paths = []
        for ext in IMG_EXTS:
            image_paths.extend(sorted(split_path.rglob(f"*{ext}")))
            image_paths.extend(sorted(split_path.rglob(f"*{ext.upper()}")))

    if not image_paths:
        raise RuntimeError(f"No images found for split={split}: {split_path}")

    return root, sorted(set(image_paths))


def image_key(path_like) -> str:
    """Use basename stem as key; this is robust for counting if validation filenames are unique."""
    return Path(str(path_like)).stem


def label_path_from_image(img_path: Path, dataset_root: Path) -> Path:
    # Common YOLO layout: images/val/xxx.jpg -> labels/val/xxx.txt
    s = str(img_path)
    s_norm = s.replace("\\", "/")
    if "/images/" in s_norm:
        label_s = s_norm.replace("/images/", "/labels/")
        return Path(label_s).with_suffix(".txt")
    # Fallback: dataset_root/images/... -> dataset_root/labels/...
    try:
        rel = img_path.resolve().relative_to(dataset_root.resolve())
        parts = list(rel.parts)
        if parts and parts[0].lower() == "images":
            parts[0] = "labels"
            return dataset_root.joinpath(*parts).with_suffix(".txt")
    except Exception:
        pass
    # Last fallback: sibling labels folder near dataset root
    return dataset_root / "labels" / "val" / f"{img_path.stem}.txt"


def load_gt_counts(data_yaml: Path, split: str = "val") -> pd.DataFrame:
    root, images = resolve_dataset_path(data_yaml, split)
    rows = []
    seen = set()
    for img in images:
        key = image_key(img)
        if key in seen:
            raise RuntimeError(
                f"Duplicate image stem '{key}' in split={split}. "
                "The script uses stem-level matching for prediction CSVs; please modify image_key() to use full canonical IDs."
            )
        seen.add(key)

        lab = label_path_from_image(img, root)
        if lab.exists():
            lines = [ln.strip() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
            gt_count = len(lines)
        else:
            gt_count = 0
        rows.append({"image_key": key, "image_path": str(img), "gt_count": gt_count})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("GT count table is empty.")
    return df


def detect_column(df: pd.DataFrame, candidates: List[str], role: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # Fuzzy contains
    for c in df.columns:
        cl = c.lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    raise RuntimeError(f"Could not detect {role} column. Available columns: {list(df.columns)}")


def load_pred_counts_by_conf(pred_csv: Path, gt_df: pd.DataFrame, conf_values: List[float]) -> Dict[float, pd.Series]:
    pred = pd.read_csv(pred_csv)
    if pred.empty:
        return {c: pd.Series(0, index=gt_df["image_key"]) for c in conf_values}

    conf_col = detect_column(pred, ["conf", "confidence", "score"], "confidence")
    id_col = detect_column(
        pred,
        ["image_key", "image_id", "image_path", "path", "filename", "file", "name"],
        "image identifier",
    )

    pred = pred.copy()
    pred["image_key"] = pred[id_col].map(image_key)
    pred[conf_col] = pd.to_numeric(pred[conf_col], errors="coerce")
    pred = pred.dropna(subset=[conf_col, "image_key"])

    valid_keys = set(gt_df["image_key"])
    overlap = len(valid_keys & set(pred["image_key"]))
    if overlap == 0:
        raise RuntimeError(
            f"No overlap between GT validation image keys and prediction CSV keys: {pred_csv}\n"
            f"GT examples: {list(gt_df['image_key'].head(5))}\n"
            f"Pred examples: {list(pred['image_key'].head(5))}"
        )

    out = {}
    for conf in conf_values:
        sub = pred[pred[conf_col] >= conf]
        cnt = sub.groupby("image_key").size()
        cnt = gt_df["image_key"].map(cnt).fillna(0).astype(int)
        out[conf] = cnt
    return out


def metric_row(resolution: int, conf: float, gt: np.ndarray, pred_count: np.ndarray) -> dict:
    err = pred_count - gt
    abs_err = np.abs(err)
    denom = np.where(gt == 0, np.nan, gt)
    mape = np.nanmean(abs_err / denom) * 100.0
    return {
        "resolution": resolution,
        "conf": float(conf),
        "images": int(len(gt)),
        "gt_total": int(gt.sum()),
        "pred_total": int(pred_count.sum()),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_percent": float(mape),
        "Bias_pred_minus_gt": float(err.mean()),
        "Relative_total_error_percent": float((pred_count.sum() - gt.sum()) / gt.sum() * 100.0),
    }


def compute_sweep(data_yaml: Path, pred_paths: Dict[int, Path], conf_values: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_df = load_gt_counts(data_yaml, split="val")
    gt = gt_df["gt_count"].to_numpy(dtype=float)

    rows = []
    for res, csv_path in pred_paths.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Prediction CSV not found for {res}: {csv_path}")
        pred_by_conf = load_pred_counts_by_conf(csv_path, gt_df, conf_values)
        for conf in conf_values:
            pred_count = pred_by_conf[conf].to_numpy(dtype=float)
            rows.append(metric_row(res, conf, gt, pred_count))

    sweep = pd.DataFrame(rows)
    selected = (
        sweep.sort_values(
            ["resolution", "MAE", "RMSE", "Bias_pred_minus_gt", "conf"],
            key=lambda s: s.abs() if s.name == "Bias_pred_minus_gt" else s,
        )
        .groupby("resolution", as_index=False)
        .first()
    )
    return sweep, selected


# -----------------------------
# Plotting
# -----------------------------
def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure_s3(sweep: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    """
    Final Figure S3 layout.

    Data source:
      - Main panel and inset are both drawn from the same sweep DataFrame.
      - No hand-drawn or image-generated curves are used.

    Layout:
      - Main panel uses a continuous y-axis.
      - Inset is placed in the upper-left blank region requested by the user.
      - Main-panel threshold labels are removed to avoid overlap.
      - Threshold labels are shown only inside the inset.
    """
    configure_style()

    colors = {
        800: "#4C78A8",
        960: "#F2A65A",
        1280: "#59A14F",
    }
    markers = {
        800: "o",
        960: "s",
        1280: "^",
    }
    labels = {
        800: "800 px",
        960: "960 px",
        1280: "1280 px",
    }

    # -------------------------------------------------------------------------
    # Main continuous plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=600)

    for res in [800, 960, 1280]:
        d = sweep[sweep["resolution"] == res].sort_values("conf")
        ax.plot(
            d["conf"],
            d["MAE"],
            marker=markers[res],
            markersize=4.0,
            linewidth=1.35,
            color=colors[res],
            label=labels[res],
        )

    # Main-panel vertical reference lines only.
    # Do NOT place 0.45 / 0.50 text on the main panel; these labels are shown in the inset.
    for x in [0.45, 0.50]:
        ax.axvline(
            x,
            linestyle="--",
            linewidth=0.8,
            color="0.35",
            alpha=0.75,
            zorder=0,
        )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Validation counting MAE")
    ax.set_xlim(0.05, 0.70)
    ax.set_xticks(CONF_VALUES)

    y_min = max(0, math.floor(float(sweep["MAE"].min())) - 1)
    y_max = math.ceil(float(sweep["MAE"].max())) + 1
    ax.set_ylim(y_min, y_max)

    ax.grid(True, axis="both", linewidth=0.35, alpha=0.30)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    # Legend stays in the upper-right and will not overlap with the inset.
    ax.legend(frameon=False, loc="upper right")

    # -------------------------------------------------------------------------
    # Summary annotation box in lower-left main panel
    # -------------------------------------------------------------------------
    selected_lines = []
    for res in [800, 960, 1280]:
        row = selected[selected["resolution"] == res].iloc[0]
        selected_lines.append(
            f"{res} px: min MAE = {row['MAE']:.2f} at conf = {row['conf']:.2f}"
        )

    summary_text = "\n".join(selected_lines)

    ax.text(
        0.065,
        y_min + 1.0,
        summary_text,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            boxstyle="square,pad=0.45",
        ),
        zorder=10,
    )

    # -------------------------------------------------------------------------
    # Inset zoom panel
    # -------------------------------------------------------------------------
    # Position requested by the user: upper-left blank region of the main panel.
    # [left, bottom, width, height] in axes fraction.
    axins = ax.inset_axes([0.17, 0.59, 0.38, 0.31])

    for res in [800, 960, 1280]:
        d = sweep[sweep["resolution"] == res].sort_values("conf")
        axins.plot(
            d["conf"],
            d["MAE"],
            marker=markers[res],
            markersize=3.2,
            linewidth=1.05,
            color=colors[res],
        )

        # Highlight selected point with the SAME marker shape and color,
        # only adding a black edge. This avoids symbol confusion.
        row = selected[selected["resolution"] == res].iloc[0]
        axins.scatter(
            [row["conf"]],
            [row["MAE"]],
            s=42,
            marker=markers[res],
            facecolor=colors[res],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    # Inset reference lines.
    for x in [0.45, 0.50]:
        axins.axvline(
            x,
            linestyle="--",
            linewidth=0.65,
            color="0.35",
            alpha=0.75,
            zorder=0,
        )

    # Inset limits: computed from the actual data around the selected-threshold region.
    axins.set_xlim(0.39, 0.53)

    local = sweep[(sweep["conf"] >= 0.39) & (sweep["conf"] <= 0.53)]
    inset_min = float(local["MAE"].min()) - 0.25
    inset_max = float(local["MAE"].max()) + 0.30
    axins.set_ylim(inset_min, inset_max)

    # Small threshold labels inside the inset only.
    inset_range = inset_max - inset_min
    for x in [0.45, 0.50]:
        axins.text(
            x,
            inset_max - 0.03 * inset_range,
            f"{x:.2f}",
            ha="center",
            va="top",
            fontsize=6.5,
            color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.15),
            zorder=6,
        )

    # Compact inset labels to reduce visual clutter.
    axins.set_xlabel("Confidence threshold", fontsize=6.5, labelpad=0.5)
    axins.set_ylabel("MAE", fontsize=6.5, labelpad=1.0)

    axins.tick_params(
        axis="both",
        which="major",
        labelsize=6.2,
        width=0.55,
        length=2.0,
        pad=1.5,
    )

    axins.grid(True, linewidth=0.30, alpha=0.28)
    axins.set_axisbelow(True)

    for spine in axins.spines.values():
        spine.set_linewidth(0.65)

    # Selected-value labels inside inset.
    # Offsets are deliberately small and separated to avoid overlap.
    # Values are computed from selected, not hard-coded.
    offsets = {
        800: (0.006, 0.08),
        960: (0.006, 0.10),
        1280: (0.006, -0.12),
    }

    for res in [800, 960, 1280]:
        row = selected[selected["resolution"] == res].iloc[0]
        dx, dy = offsets[res]

        axins.annotate(
            f"{row['MAE']:.2f}",
            xy=(row["conf"], row["MAE"]),
            xytext=(row["conf"] + dx, row["MAE"] + dy),
            fontsize=6.2,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor=colors[res],
                linewidth=0.55,
                boxstyle="round,pad=0.15",
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="0.25",
                linewidth=0.45,
                shrinkA=0,
                shrinkB=2,
            ),
            zorder=6,
        )

    # Manual layout adjustment avoids tight_layout warnings with inset axes.
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.13,
        top=0.97,
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "tif", "svg", "pdf"]:
        fig.savefig(
            out_dir / f"FigureS3_validation_confidence_MAE_curves.{ext}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=str, default=str(DATA_YAML))
    parser.add_argument("--pred800", type=str, default=str(DEFAULT_PREDS[800]))
    parser.add_argument("--pred960", type=str, default=str(DEFAULT_PREDS[960]))
    parser.add_argument("--pred1280", type=str, default=str(DEFAULT_PREDS[1280]))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--conf-values", type=str, default=",".join(map(str, CONF_VALUES)))
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    out_dir = Path(args.out_dir)
    pred_paths = {
        800: Path(args.pred800),
        960: Path(args.pred960),
        1280: Path(args.pred1280),
    }
    conf_values = [float(x.strip()) for x in args.conf_values.split(",") if x.strip()]

    print("[Figure S3] Recomputing validation confidence sweep from REAL prediction CSVs.")
    print("Data YAML:", data_yaml)
    print("Prediction CSVs:")
    for res, p in pred_paths.items():
        print(f"  {res}: {p}")

    sweep, selected = compute_sweep(data_yaml, pred_paths, conf_values)

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "FigureS3_validation_confidence_MAE_sweep.csv"
    selected_csv = out_dir / "FigureS3_validation_confidence_selected_thresholds.csv"
    meta_json = out_dir / "FigureS3_validation_confidence_MAE_metadata.json"

    sweep.to_csv(sweep_csv, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    plot_figure_s3(sweep, selected, out_dir)

    meta = {
        "source": "recomputed from validation prediction CSVs and validation labels",
        "data_yaml": str(data_yaml),
        "prediction_csvs": {str(k): str(v) for k, v in pred_paths.items()},
        "conf_values": conf_values,
        "outputs": {
            "sweep_csv": str(sweep_csv),
            "selected_csv": str(selected_csv),
            "figure_png": str(out_dir / "FigureS3_validation_confidence_MAE_curves.png"),
            "figure_tif": str(out_dir / "FigureS3_validation_confidence_MAE_curves.tif"),
            "figure_svg": str(out_dir / "FigureS3_validation_confidence_MAE_curves.svg"),
            "figure_pdf": str(out_dir / "FigureS3_validation_confidence_MAE_curves.pdf"),
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE] Figure S3 generated from real CSVs.")
    print(sweep_csv)
    print(selected_csv)
    print(out_dir / "FigureS3_validation_confidence_MAE_curves.png")
    print("\nSelected thresholds:")
    print(selected[["resolution", "conf", "MAE", "RMSE", "Bias_pred_minus_gt", "pred_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
