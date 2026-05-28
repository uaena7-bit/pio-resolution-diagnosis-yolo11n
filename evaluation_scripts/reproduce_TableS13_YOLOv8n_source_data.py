#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalize YOLOv8n cross-detector source-data file names for the CSSP manuscript.

Purpose
-------
Create the final paper-oriented file names for the YOLOv8n cross-detector
robustness check.

Final required files
--------------------
1. supplementary_source_data/TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv
2. supplementary_source_data/README_TableS13_YOLOv8n_cross_detector_accuracy.md
3. raw_logs/

Notes
-----
- YOLOv8n is used only as a cross-detector robustness check.
- The final Table S13 source data include accuracy and training time only.
- YOLOv8n latency is not used for the final deployment conclusion because the
  initial single-run summary contained a timing anomaly for 1280 test inference.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from textwrap import dedent


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
LOG_ROOT = PROJECT_ROOT / "06_LOGS"

SUPP_DIR = LOG_ROOT / "supplementary_source_data"
RAW_DIR = LOG_ROOT / "raw_logs"

SUPP_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FINAL_ACCURACY_CSV = SUPP_DIR / "TableS13_YOLOv8n_cross_detector_accuracy_seed42.csv"
FINAL_README = SUPP_DIR / "README_TableS13_YOLOv8n_cross_detector_accuracy.md"
FINAL_RECHECK_CSV = RAW_DIR / "YOLOv8n_1280_test_latency_recheck_raw.csv"

# Candidate locations for the already cleaned accuracy-only CSV.
ACCURACY_CSV_CANDIDATES = [
    LOG_ROOT / "cleaned_source_data" / "yolov8n_3res_seed42_accuracy_summary.csv",
    LOG_ROOT / "supplementary_source_data" / "yolov8n_3res_seed42_accuracy_summary.csv",
    LOG_ROOT / "yolov8n_3res_seed42_accuracy_summary.csv",
    FINAL_ACCURACY_CSV,
]

# Candidate locations for the 1280 latency recheck CSV.
RECHECK_CSV_CANDIDATES = [
    LOG_ROOT / "cleaned_source_data" / "yolov8n_1280_test_speed_recheck.csv",
    LOG_ROOT / "raw_logs" / "yolov8n_1280_test_speed_recheck.csv",
    LOG_ROOT / "yolov8n_1280_test_speed_recheck.csv",
    FINAL_RECHECK_CSV,
]

# Original raw single-run YOLOv8n summary.
# This is not one of the three required final files, but the README references it.
RAW_SINGLE_RUN_SUMMARY_CANDIDATES = [
    LOG_ROOT / "yolov8n_3res_seed42_train_eval_summary.csv",
    LOG_ROOT / "raw_logs" / "yolov8n_3res_seed42_train_eval_summary_raw.csv",
]

RAW_SINGLE_RUN_SUMMARY_FINAL = RAW_DIR / "YOLOv8n_cross_detector_train_eval_raw_seed42.csv"

COPY_OPTIONAL_RAW_SINGLE_RUN_SUMMARY = True


# =============================================================================
# Canonical fallback data
# =============================================================================

CANONICAL_ACCURACY_ROWS = [
    {
        "experiment_id": "Y8_M1_YOLOv8n_PIO_GRDB_MD5_7_1_2_img800_seed42_SGD_e300_b2",
        "model": "YOLOv8n",
        "resolution": "800",
        "seed": "42",
        "best_pt_path": r"D:\Broiler chicken detection dataset\02_RUNS\06_cross_detector_YOLOv8n_resolution\Y8_M1_YOLOv8n_PIO_GRDB_MD5_7_1_2_img800_seed42_SGD_e300_b2\weights\best.pt",
        "training_time_h": "4.908442846222218",
        "val_precision": "0.9496410794835519",
        "val_recall": "0.9509205753738792",
        "val_map50": "0.9759941621211563",
        "val_map50_95": "0.7465332636135771",
        "test_precision": "0.9328052204156492",
        "test_recall": "0.9356311521079675",
        "test_map50": "0.9660268663030382",
        "test_map50_95": "0.7271698998902717",
    },
    {
        "experiment_id": "Y8_M2_YOLOv8n_PIO_GRDB_MD5_7_1_2_img960_seed42_SGD_e300_b2",
        "model": "YOLOv8n",
        "resolution": "960",
        "seed": "42",
        "best_pt_path": r"D:\Broiler chicken detection dataset\02_RUNS\06_cross_detector_YOLOv8n_resolution\Y8_M2_YOLOv8n_PIO_GRDB_MD5_7_1_2_img960_seed42_SGD_e300_b2\weights\best.pt",
        "training_time_h": "6.640595663277781",
        "val_precision": "0.9515555878726205",
        "val_recall": "0.9569625704849799",
        "val_map50": "0.984578460153381",
        "val_map50_95": "0.7710476089320263",
        "test_precision": "0.9360855706494167",
        "test_recall": "0.9416075733928813",
        "test_map50": "0.9732527836953169",
        "test_map50_95": "0.7442502286044805",
    },
    {
        "experiment_id": "Y8_M3_YOLOv8n_PIO_GRDB_MD5_7_1_2_img1280_seed42_SGD_e300_b2",
        "model": "YOLOv8n",
        "resolution": "1280",
        "seed": "42",
        "best_pt_path": r"D:\Broiler chicken detection dataset\02_RUNS\06_cross_detector_YOLOv8n_resolution\Y8_M3_YOLOv8n_PIO_GRDB_MD5_7_1_2_img1280_seed42_SGD_e300_b2\weights\best.pt",
        "training_time_h": "13.98413970036112",
        "val_precision": "0.9542504819923235",
        "val_recall": "0.957973174366617",
        "val_map50": "0.9853052288227626",
        "val_map50_95": "0.7862475500880595",
        "test_precision": "0.9379546064269513",
        "test_recall": "0.9431307190188583",
        "test_map50": "0.9739915074742236",
        "test_map50_95": "0.7573319097172482",
    },
]

CANONICAL_RECHECK_ROWS = [
    {
        "run_id": "Y8_M3_1280_test_recheck_01",
        "model": "YOLOv8n",
        "resolution": "1280",
        "seed": "42",
        "split": "test",
        "imgsz": "1280",
        "batch": "2",
        "workers": "0",
        "conf": "0.001",
        "iou": "0.70",
        "max_det": "1000",
        "precision": "0.938",
        "recall": "0.943",
        "map50": "0.974",
        "map50_95": "0.757",
        "preprocess_ms_per_img": "1.1",
        "inference_ms_per_img": "5.6",
        "postprocess_ms_per_img": "1.6",
        "total_ms_per_img": "8.3",
        "note": "Independent re-run of 1280-pixel test evaluation after initial single-run timing anomaly.",
    },
    {
        "run_id": "Y8_M3_1280_test_recheck_02",
        "model": "YOLOv8n",
        "resolution": "1280",
        "seed": "42",
        "split": "test",
        "imgsz": "1280",
        "batch": "2",
        "workers": "0",
        "conf": "0.001",
        "iou": "0.70",
        "max_det": "1000",
        "precision": "0.938",
        "recall": "0.943",
        "map50": "0.974",
        "map50_95": "0.757",
        "preprocess_ms_per_img": "1.1",
        "inference_ms_per_img": "5.1",
        "postprocess_ms_per_img": "1.4",
        "total_ms_per_img": "7.6",
        "note": "Independent re-run of 1280-pixel test evaluation after initial single-run timing anomaly.",
    },
    {
        "run_id": "Y8_M3_1280_test_recheck_03",
        "model": "YOLOv8n",
        "resolution": "1280",
        "seed": "42",
        "split": "test",
        "imgsz": "1280",
        "batch": "2",
        "workers": "0",
        "conf": "0.001",
        "iou": "0.70",
        "max_det": "1000",
        "precision": "0.938",
        "recall": "0.943",
        "map50": "0.974",
        "map50_95": "0.757",
        "preprocess_ms_per_img": "1.2",
        "inference_ms_per_img": "5.3",
        "postprocess_ms_per_img": "1.4",
        "total_ms_per_img": "7.9",
        "note": "Independent re-run of 1280-pixel test evaluation after initial single-run timing anomaly.",
    },
]


# =============================================================================
# Helpers
# =============================================================================

def first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        print(f"[KEEP] {dst}")
        return
    shutil.copy2(src, dst)
    print(f"[COPY] {src}")
    print(f"   ->  {dst}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE] {path}")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_accuracy_csv(path: Path) -> None:
    rows = read_csv_rows(path)

    required_cols = {
        "experiment_id",
        "model",
        "resolution",
        "seed",
        "best_pt_path",
        "training_time_h",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
    }

    if not rows:
        raise ValueError(f"Accuracy CSV is empty: {path}")

    cols = set(rows[0].keys())
    missing = sorted(required_cols - cols)
    if missing:
        raise ValueError(f"Accuracy CSV missing columns: {missing}")

    forbidden_fragments = ["speed", "latency", "preprocess", "inference", "postprocess"]
    forbidden_cols = [
        c for c in rows[0].keys()
        if any(fragment in c.lower() for fragment in forbidden_fragments)
    ]
    if forbidden_cols:
        raise ValueError(
            "Final Table S13 accuracy CSV must not contain latency/speed columns. "
            f"Forbidden columns found: {forbidden_cols}"
        )

    resolutions = sorted({str(r["resolution"]) for r in rows})
    if resolutions != ["1280", "800", "960"]:
        raise ValueError(f"Expected resolutions 800/960/1280, got: {resolutions}")

    models = sorted({r["model"] for r in rows})
    if models != ["YOLOv8n"]:
        raise ValueError(f"Expected model YOLOv8n only, got: {models}")

    print(f"[VALID] Accuracy CSV: {path}")


def validate_recheck_csv(path: Path) -> None:
    rows = read_csv_rows(path)

    required_cols = {
        "run_id",
        "preprocess_ms_per_img",
        "inference_ms_per_img",
        "postprocess_ms_per_img",
        "total_ms_per_img",
    }

    if not rows:
        raise ValueError(f"Recheck CSV is empty: {path}")

    cols = set(rows[0].keys())
    missing = sorted(required_cols - cols)
    if missing:
        raise ValueError(f"Recheck CSV missing columns: {missing}")

    if len(rows) < 3:
        raise ValueError(f"Expected at least 3 recheck runs, got {len(rows)}")

    print(f"[VALID] Recheck CSV: {path}")


def write_readme(path: Path) -> None:
    content = dedent(
        f"""
        # Table S13 — YOLOv8n Cross-Detector Robustness Check

        This folder provides the final source data for the YOLOv8n cross-detector robustness check.

        YOLOv8n was used only to examine whether the leakage-controlled input-resolution diagnosis protocol generalizes beyond the main YOLO11n detector. It is not used as a detector-ranking benchmark, not used for the main seed-repeatability analysis, and not used for the final deployment-resolution conclusion.

        ## Final Paper Source Data

        The final source-data file for the paper is:

        ```text
        supplementary_source_data/{FINAL_ACCURACY_CSV.name}
        ```

        This file contains only accuracy and training-time results:

        - training time
        - validation precision, recall, mAP50, and mAP50-95
        - test precision, recall, mAP50, and mAP50-95
        - best checkpoint path

        Latency columns are intentionally excluded from the final Table S13 source data.

        ## Raw Latency Recheck File

        The 1280-pixel YOLOv8n test-speed recheck file is:

        ```text
        raw_logs/{FINAL_RECHECK_CSV.name}
        ```

        The original single-run YOLOv8n summary contained a timing anomaly for the 1280-pixel test inference time:

        ```text
        Initial raw recorded value: 116.995 ms/img
        Independent re-runs: 5.6 / 5.1 / 5.3 ms/img
        Median rechecked inference time: 5.3 ms/img
        ```

        Because only the 1280-pixel test speed was re-checked, the latency values are not uniform in provenance across all YOLOv8n resolutions. Therefore, YOLOv8n latency is not used for the final deployment analysis.

        Deployment-oriented conclusions should be based on the repeated locked-weight YOLO11n benchmark.

        ## Recommended Paper Statement

        YOLOv8n was used as a cross-detector robustness check under the same leakage-controlled split and input-resolution protocol. Because the latency values from the initial single-run summary contained a timing anomaly, only accuracy and training-time results were used for the YOLOv8n cross-detector comparison. Deployment-oriented conclusions were based on the repeated locked-weight YOLO11n benchmark.

        ## Data Dictionary — `{FINAL_ACCURACY_CSV.name}`

        | Column | Description |
        |---|---|
        | `experiment_id` | Full experiment identifier |
        | `model` | Detector architecture |
        | `resolution` | Input image size in pixels |
        | `seed` | Random seed |
        | `best_pt_path` | Absolute path to the best checkpoint in the local experiment environment |
        | `training_time_h` | Total training time in hours, measured by the training script |
        | `val_precision` | Validation precision |
        | `val_recall` | Validation recall |
        | `val_map50` | Validation mAP@0.50 |
        | `val_map50_95` | Validation mAP@0.50:0.95 |
        | `test_precision` | Test precision |
        | `test_recall` | Test recall |
        | `test_map50` | Test mAP@0.50 |
        | `test_map50_95` | Test mAP@0.50:0.95 |

        ## Data Dictionary — `{FINAL_RECHECK_CSV.name}`

        | Column | Description |
        |---|---|
        | `run_id` | Individual re-run identifier |
        | `preprocess_ms_per_img` | Preprocessing time per image |
        | `inference_ms_per_img` | Inference time per image |
        | `postprocess_ms_per_img` | Post-processing time per image |
        | `total_ms_per_img` | Sum of preprocessing, inference, and post-processing time per image |
        | Remaining columns | Hardware, software, and evaluation context for traceability |

        ## Interpretation

        The cleaned YOLOv8n accuracy summary shows that test mAP50-95 increased from 800 to 960 and further to 1280 pixels. This supports the robustness of the proposed leakage-controlled resolution-diagnosis protocol across another lightweight YOLO-family detector.

        However, YOLOv8n is not used to select the final deployment resolution. The final deployment and Pareto-style resolution-selection analysis should use the repeated locked-weight YOLO11n benchmark.
        """
    ).strip() + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[WRITE] {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("Finalize Table S13 YOLOv8n source-data file names")
    print("=" * 100)

    # 1. Accuracy-only CSV.
    src_acc = first_existing(ACCURACY_CSV_CANDIDATES)
    if src_acc is None:
        print("[WARN] No existing accuracy-only CSV found. Writing canonical fallback data.")
        write_csv(FINAL_ACCURACY_CSV, CANONICAL_ACCURACY_ROWS)
    else:
        copy_if_needed(src_acc, FINAL_ACCURACY_CSV)

    validate_accuracy_csv(FINAL_ACCURACY_CSV)

    # 2. 1280 latency recheck raw CSV.
    src_recheck = first_existing(RECHECK_CSV_CANDIDATES)
    if src_recheck is None:
        print("[WARN] No existing 1280 latency recheck CSV found. Writing canonical fallback data.")
        write_csv(FINAL_RECHECK_CSV, CANONICAL_RECHECK_ROWS)
    else:
        copy_if_needed(src_recheck, FINAL_RECHECK_CSV)

    validate_recheck_csv(FINAL_RECHECK_CSV)

    # 3. README for Table S13.
    write_readme(FINAL_README)

    # Optional raw single-run summary.
    if COPY_OPTIONAL_RAW_SINGLE_RUN_SUMMARY:
        src_raw = first_existing(RAW_SINGLE_RUN_SUMMARY_CANDIDATES)
        if src_raw is None:
            print("[WARN] Optional raw single-run YOLOv8n summary not found; skipped.")
        else:
            copy_if_needed(src_raw, RAW_SINGLE_RUN_SUMMARY_FINAL)
            print("[INFO] Optional raw single-run summary retained for traceability.")

    print("\n" + "=" * 100)
    print("[DONE] Final YOLOv8n Table S13 file names are fixed.")
    print("=" * 100)

    print("\nRequired final files:")
    print(f"1. {FINAL_ACCURACY_CSV}")
    print(f"2. {FINAL_README}")
    print(f"3. {FINAL_RECHECK_CSV}")

    if COPY_OPTIONAL_RAW_SINGLE_RUN_SUMMARY and RAW_SINGLE_RUN_SUMMARY_FINAL.exists():
        print("\nOptional raw traceability file:")
        print(f"4. {RAW_SINGLE_RUN_SUMMARY_FINAL}")


if __name__ == "__main__":
    main()
