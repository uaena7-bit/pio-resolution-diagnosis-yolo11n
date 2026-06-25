#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source-component sensitivity of validation-calibrated counting thresholds.

For each resolution, the threshold is selected from:
1) full validation set
2) validation component 1 only
3) validation component 10 only

Each selected threshold is then applied to the unchanged held-out test set.
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
OUT_DIR = COUNT_DIR / "experiment1_source_component_threshold_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONF_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def image_key(name: str) -> str:
    return Path(str(name)).stem


def load_gt_from_manifest(manifest_csv: Path):
    df = pd.read_csv(manifest_csv)
    required = ["new_split", "component_id", "new_image_name", "source_group", "instances"]
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


def select_threshold_on_subset(pred_df: pd.DataFrame, gt_subset: pd.DataFrame):
    rows = []
    for conf in CONF_VALUES:
        m = evaluate_at_conf(pred_df, gt_subset, conf)
        rows.append({"conf": float(conf), **m, "abs_Bias": abs(m["Bias_pred_minus_gt"])})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["MAE", "RMSE", "abs_Bias", "conf"]).iloc[0].copy()
    return best, sweep


def main():
    val_df, test_df = load_gt_from_manifest(MANIFEST_CSV)
    calibration_sets = {"all_validation": val_df}
    for comp_id in sorted(val_df["component_id"].unique()):
        calibration_sets[f"component_{comp_id}_only"] = val_df[val_df["component_id"] == comp_id].copy()

    summary_rows, sweep_rows = [], []
    for res in [800, 960, 1280]:
        pred_val = load_prediction_csv(PRED_VAL[res])
        pred_test = load_prediction_csv(PRED_TEST[res])
        for calib_name, calib_df in calibration_sets.items():
            best, sweep = select_threshold_on_subset(pred_val, calib_df)
            selected_conf = float(best["conf"])
            test_metrics = evaluate_at_conf(pred_test, test_df, selected_conf)
            summary_rows.append({
                "resolution": res,
                "calibration_source": calib_name,
                "calibration_images": int(len(calib_df)),
                "selected_conf": selected_conf,
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAPE_percent": test_metrics["MAPE_percent"],
                "test_Bias_pred_minus_gt": test_metrics["Bias_pred_minus_gt"],
                "test_Relative_total_error_percent": test_metrics["Relative_total_error_percent"],
            })
            sweep = sweep.copy()
            sweep.insert(0, "resolution", res)
            sweep.insert(1, "calibration_source", calib_name)
            sweep.insert(2, "calibration_images", len(calib_df))
            sweep_rows.append(sweep)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "TableS18_source_component_threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(sweep_rows, ignore_index=True).to_csv(OUT_DIR / "TableS18_source_component_threshold_sensitivity_sweeps.csv", index=False, encoding="utf-8-sig")
    print("Finished source-component threshold sensitivity analysis.")


if __name__ == "__main__":
    main()
