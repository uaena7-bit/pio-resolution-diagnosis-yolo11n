#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counting-threshold robustness comparison.

This script compares three counting protocols:
1) fixed confidence = 0.25
2) validation-calibrated threshold
3) diagnostic test-set best threshold
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
COUNT_DIR = PROJECT_ROOT / "03_RESULTS" / "counting_val_calibrated_conf_3res"
MANIFEST_CSV = PROJECT_ROOT / "00_DATASET" / "PIO-GRDB-MD5-7_1_2" / "reports" / "manifest.csv"

PRED_VAL = {
    800: COUNT_DIR / "predictions_val_800.csv",
    960: COUNT_DIR / "predictions_val_960.csv",
    1280: COUNT_DIR / "predictions_val_1280.csv",
}
PRED_TEST = {
    800: COUNT_DIR / "predictions_test_800.csv",
    960: COUNT_DIR / "predictions_test_960.csv",
    1280: COUNT_DIR / "predictions_test_1280.csv",
}
OUT_DIR = COUNT_DIR / "experiment2_counting_threshold_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONF_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
FIXED_CONF = 0.25


def image_key(name: str) -> str:
    return Path(str(name)).stem


def load_gt_from_manifest(manifest_csv: Path):
    df = pd.read_csv(manifest_csv)
    required = ["new_split", "new_image_name", "instances"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"manifest.csv missing columns: {missing}")
    df = df.copy()
    df["image_key"] = df["new_image_name"].map(image_key)
    df["gt_count"] = pd.to_numeric(df["instances"], errors="coerce").fillna(0).astype(int)
    val_df = df[df["new_split"].astype(str).str.lower().isin(["val", "validation"])].copy()
    test_df = df[df["new_split"].astype(str).str.lower().isin(["test"])].copy()
    if val_df.empty or test_df.empty:
        raise RuntimeError("Validation or test split is empty in manifest.csv")
    return val_df, test_df


def load_prediction_csv(pred_csv: Path) -> pd.DataFrame:
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")
    pred = pd.read_csv(pred_csv)
    if "image_name" not in pred.columns or "score" not in pred.columns:
        raise RuntimeError(f"{pred_csv} must contain image_name and score columns")
    pred = pred.copy()
    pred["image_key"] = pred["image_name"].map(image_key)
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    return pred.dropna(subset=["image_key", "score"])


def count_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame, conf: float) -> np.ndarray:
    counts = pred_df[pred_df["score"] >= conf].groupby("image_key").size()
    return gt_df["image_key"].map(counts).fillna(0).astype(int).to_numpy()


def compute_metrics(gt_count: np.ndarray, pred_count: np.ndarray) -> dict:
    err = pred_count - gt_count
    abs_err = np.abs(err)
    nonzero = gt_count > 0
    mape = np.mean(abs_err[nonzero] / gt_count[nonzero]) * 100 if nonzero.any() else np.nan
    gt_total = int(gt_count.sum())
    pred_total = int(pred_count.sum())
    return {
        "images": int(len(gt_count)),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_percent": float(mape),
        "Bias_pred_minus_gt": float(err.mean()),
        "Relative_total_error_percent": float((pred_total - gt_total) / gt_total * 100.0),
    }


def evaluate_at_conf(pred_df: pd.DataFrame, gt_df: pd.DataFrame, conf: float) -> dict:
    gt_count = gt_df["gt_count"].to_numpy(dtype=float)
    pred_count = count_predictions(pred_df, gt_df, conf).astype(float)
    return compute_metrics(gt_count, pred_count)


def make_sweep(resolution: int, pred_df: pd.DataFrame, gt_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    for conf in CONF_VALUES:
        metrics = evaluate_at_conf(pred_df, gt_df, conf)
        rows.append({"resolution": resolution, "split": split_name, "conf": float(conf), **metrics, "abs_Bias": abs(metrics["Bias_pred_minus_gt"])})
    return pd.DataFrame(rows)


def select_best_threshold(sweep_df: pd.DataFrame):
    return sweep_df.sort_values(["MAE", "RMSE", "abs_Bias", "conf"]).iloc[0].copy()


def main():
    val_df, test_df = load_gt_from_manifest(MANIFEST_CSV)
    summary_rows, val_sweep_rows, test_sweep_rows = [], [], []
    for res in [800, 960, 1280]:
        pred_val = load_prediction_csv(PRED_VAL[res])
        pred_test = load_prediction_csv(PRED_TEST[res])

        fixed_metrics = evaluate_at_conf(pred_test, test_df, FIXED_CONF)
        summary_rows.append({"resolution": res, "protocol": "fixed_conf_0.25", "selected_conf": FIXED_CONF, **fixed_metrics})

        val_sweep = make_sweep(res, pred_val, val_df, "validation")
        val_sweep_rows.append(val_sweep)
        val_best = select_best_threshold(val_sweep)
        val_selected_conf = float(val_best["conf"])
        val_metrics = evaluate_at_conf(pred_test, test_df, val_selected_conf)
        summary_rows.append({"resolution": res, "protocol": "validation_calibrated", "selected_conf": val_selected_conf, **val_metrics})

        test_sweep = make_sweep(res, pred_test, test_df, "test")
        test_sweep_rows.append(test_sweep)
        test_best = select_best_threshold(test_sweep)
        summary_rows.append({
            "resolution": res,
            "protocol": "test_diagnostic_best",
            "selected_conf": float(test_best["conf"]),
            "images": int(test_best["images"]),
            "gt_total": int(test_best["gt_total"]),
            "pred_total": int(test_best["pred_total"]),
            "MAE": float(test_best["MAE"]),
            "RMSE": float(test_best["RMSE"]),
            "MAPE_percent": float(test_best["MAPE_percent"]),
            "Bias_pred_minus_gt": float(test_best["Bias_pred_minus_gt"]),
            "Relative_total_error_percent": float(test_best["Relative_total_error_percent"]),
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "TableS19_counting_threshold_robustness_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(val_sweep_rows, ignore_index=True).to_csv(OUT_DIR / "TableS19_validation_confidence_sweep.csv", index=False, encoding="utf-8-sig")
    pd.concat(test_sweep_rows, ignore_index=True).to_csv(OUT_DIR / "TableS19_test_confidence_sweep.csv", index=False, encoding="utf-8-sig")
    print("Finished counting-threshold robustness comparison.")


if __name__ == "__main__":
    main()
