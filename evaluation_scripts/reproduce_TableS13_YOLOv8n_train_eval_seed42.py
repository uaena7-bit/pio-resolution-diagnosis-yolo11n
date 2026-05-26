#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n three-resolution cross-detector robustness experiment.

Experiment IDs
--------------
Y8_M1_YOLOv8n_PIO_GRDB_MD5_7_1_2_img800_seed42_SGD_e300_b2
Y8_M2_YOLOv8n_PIO_GRDB_MD5_7_1_2_img960_seed42_SGD_e300_b2
Y8_M3_YOLOv8n_PIO_GRDB_MD5_7_1_2_img1280_seed42_SGD_e300_b2

Purpose
-------
Train YOLOv8n on the leakage-controlled PIO-GRDB-MD5-7_1_2 split at
input resolutions 800, 960, and 1280.

This experiment is intended as a cross-detector robustness check for the
resolution-diagnosis protocol, not as a new detector proposal or SOTA comparison.

Dataset
-------
D:/Broiler chicken detection dataset/00_DATASET/PIO-GRDB-MD5-7_1_2/dataset.yaml

Core training protocol
----------------------
model       : YOLOv8n
imgsz       : 800, 960, 1280
seed        : 42
epochs      : 300
batch       : 2
optimizer   : SGD
lr0         : 0.01
cos_lr      : True
max_det     : 1500
workers     : 0
device      : 0
split       : PIO-GRDB-MD5-7_1_2, leakage-controlled

Final evaluation protocol
-------------------------
split       : val and test
conf        : 0.001
iou         : 0.70
max_det     : 1000
batch       : 2
workers     : 0

Outputs
-------
Training runs:
D:/Broiler chicken detection dataset/02_RUNS/06_cross_detector_YOLOv8n_resolution/

Final evaluations:
D:/Broiler chicken detection dataset/02_RUNS/06_cross_detector_YOLOv8n_resolution_eval/

Summary:
D:/Broiler chicken detection dataset/06_LOGS/yolov8n_3res_seed42_train_eval_summary.csv
"""

from __future__ import annotations

import csv
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


# =============================================================================
# Global configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")

DATA_YAML = PROJECT_ROOT / r"00_DATASET\PIO-GRDB-MD5-7_1_2\dataset.yaml"

RUNS_ROOT = PROJECT_ROOT / r"02_RUNS\06_cross_detector_YOLOv8n_resolution"
EVAL_ROOT = PROJECT_ROOT / r"02_RUNS\06_cross_detector_YOLOv8n_resolution_eval"
LOG_ROOT = PROJECT_ROOT / r"06_LOGS"

# Prefer local pretrained weight for reproducibility.
# If the local file is absent, the script falls back to "yolov8n.pt",
# which lets Ultralytics download or find it from its cache.
LOCAL_MODEL_WEIGHTS = PROJECT_ROOT / r"weights\yolov8n.pt"
MODEL_WEIGHTS = str(LOCAL_MODEL_WEIGHTS) if LOCAL_MODEL_WEIGHTS.exists() else "yolov8n.pt"

SEED = 42

EXPERIMENTS = [
    {
        "stage": "Y8_M1",
        "resolution": 800,
        "experiment_id": "Y8_M1_YOLOv8n_PIO_GRDB_MD5_7_1_2_img800_seed42_SGD_e300_b2",
    },
    {
        "stage": "Y8_M2",
        "resolution": 960,
        "experiment_id": "Y8_M2_YOLOv8n_PIO_GRDB_MD5_7_1_2_img960_seed42_SGD_e300_b2",
    },
    {
        "stage": "Y8_M3",
        "resolution": 1280,
        "experiment_id": "Y8_M3_YOLOv8n_PIO_GRDB_MD5_7_1_2_img1280_seed42_SGD_e300_b2",
    },
]

# Change this list if you want to run only one or two resolutions.
TARGET_RESOLUTIONS = [800, 960, 1280]

SKIP_TRAIN_IF_BEST_EXISTS = True

COMMON_TRAIN_ARGS = {
    "data": str(DATA_YAML),
    "epochs": 300,
    "batch": 2,
    "seed": SEED,
    "optimizer": "SGD",
    "lr0": 0.01,
    "cos_lr": True,
    "max_det": 1500,
    "project": str(RUNS_ROOT),
    "exist_ok": False,
    "pretrained": True,
    "patience": 50,
    "workers": 0,
    "device": 0,
    "verbose": True,
    "plots": True,
    "save": True,
    "save_period": -1,
    "val": True,
}

FINAL_EVAL_ARGS = {
    "data": str(DATA_YAML),
    "batch": 2,
    "device": 0,
    "workers": 0,
    "conf": 0.001,
    "iou": 0.70,
    "max_det": 1000,
    "plots": False,
    "verbose": True,
}


# =============================================================================
# Helper functions
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Set common random seeds for reproducible training setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Kept consistent with the YOLO11n baseline scripts.
    torch.backends.cudnn.benchmark = True


def check_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def check_dataset_layout() -> None:
    """Check that the generated PIO-GRDB-MD5 dataset exists before training."""
    check_file_exists(DATA_YAML, "dataset.yaml")

    dataset_root = DATA_YAML.parent

    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing required dataset directory: {d}")

    audit_file = dataset_root / "reports" / "leakage_audit.txt"
    split_stats = dataset_root / "reports" / "split_stats.csv"

    if not audit_file.exists():
        print(f"[WARN] Leakage audit file not found: {audit_file}")
    if not split_stats.exists():
        print(f"[WARN] Split statistics file not found: {split_stats}")


def collect_environment_info() -> dict:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = "unknown"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "ultralytics": ultralytics_version,
        "working_directory": str(Path.cwd()),
    }


def write_experiment_metadata(
    experiment_id: str,
    resolution: int,
    train_args: dict,
    final_eval_args: dict,
) -> Path:
    """Write metadata before training so the protocol is recorded even if training stops."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    exp_log_dir = LOG_ROOT / experiment_id
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_id": experiment_id,
        "study_stage": "YOLOv8n cross-detector resolution robustness check",
        "dataset_protocol": "PIO-GRDB-MD5-7_1_2 leakage-controlled split",
        "dataset_yaml": str(DATA_YAML),
        "model": "YOLOv8n",
        "model_weights": str(MODEL_WEIGHTS),
        "local_model_weights_exists": LOCAL_MODEL_WEIGHTS.exists(),
        "resolution": resolution,
        "seed": SEED,
        "train_args": train_args,
        "final_eval_args": final_eval_args,
        "environment": collect_environment_info(),
    }

    metadata_path = exp_log_dir / "experiment_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, exp_log_dir / script_path.name)

    return metadata_path


def extract_metrics(metrics) -> dict:
    """Extract common detection metrics from Ultralytics DetMetrics object."""
    out = {
        "precision": "",
        "recall": "",
        "map50": "",
        "map50_95": "",
    }

    try:
        out["precision"] = float(metrics.box.mp)
    except Exception:
        pass

    try:
        out["recall"] = float(metrics.box.mr)
    except Exception:
        pass

    try:
        out["map50"] = float(metrics.box.map50)
    except Exception:
        pass

    try:
        out["map50_95"] = float(metrics.box.map)
    except Exception:
        pass

    return out


def extract_speed(metrics) -> dict:
    speed = getattr(metrics, "speed", {})
    return {
        "preprocess_ms_img": float(speed.get("preprocess", 0.0)),
        "inference_ms_img": float(speed.get("inference", 0.0)),
        "postprocess_ms_img": float(speed.get("postprocess", 0.0)),
    }


def run_final_eval(
    best_pt: Path,
    experiment_id: str,
    resolution: int,
    split: str,
) -> dict:
    print("-" * 100)
    print(f"[FINAL EVAL] {experiment_id} | split={split} | imgsz={resolution}")
    print("-" * 100)

    model = YOLO(str(best_pt))

    metrics = model.val(
        **FINAL_EVAL_ARGS,
        split=split,
        imgsz=resolution,
        project=str(EVAL_ROOT),
        name=f"{experiment_id}_{split}_maxdet1000",
        save_json=(split == "test"),
    )

    values = extract_metrics(metrics)
    values.update(extract_speed(metrics))
    values["split"] = split

    return values


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_one_experiment(exp: dict) -> None:
    experiment_id = exp["experiment_id"]
    resolution = int(exp["resolution"])

    if resolution not in TARGET_RESOLUTIONS:
        print(f"[SKIP] resolution={resolution} not in TARGET_RESOLUTIONS")
        return

    run_dir = RUNS_ROOT / experiment_id
    best_pt = run_dir / "weights" / "best.pt"

    train_args = dict(COMMON_TRAIN_ARGS)
    train_args.update({
        "imgsz": resolution,
        "name": experiment_id,
    })

    metadata_path = write_experiment_metadata(
        experiment_id=experiment_id,
        resolution=resolution,
        train_args=train_args,
        final_eval_args=FINAL_EVAL_ARGS,
    )

    print("=" * 100)
    print(f"[START] {experiment_id}")
    print(f"[MODEL] YOLOv8n")
    print(f"[RESOLUTION] {resolution}")
    print(f"[SEED] {SEED}")
    print(f"[MODEL_WEIGHTS] {MODEL_WEIGHTS}")
    print(f"[RUN_DIR] {run_dir}")
    print(f"[METADATA] {metadata_path}")
    print("=" * 100)

    training_time_h = ""

    if best_pt.exists() and SKIP_TRAIN_IF_BEST_EXISTS:
        print(f"[SKIP TRAIN] Existing best.pt found: {best_pt}")
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists but best.pt was not found or skipping is disabled:\n"
                f"{run_dir}\n"
                f"Please check, rename, or delete this directory before rerunning."
            )

        set_global_seed(SEED)

        model = YOLO(MODEL_WEIGHTS)

        start_time = time.perf_counter()
        model.train(**train_args)
        elapsed_s = time.perf_counter() - start_time
        training_time_h = elapsed_s / 3600.0

        print(f"[TRAINING TIME] {experiment_id}: {training_time_h:.3f} h")

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt was not found: {best_pt}")

    val_metrics = run_final_eval(
        best_pt=best_pt,
        experiment_id=experiment_id,
        resolution=resolution,
        split="val",
    )

    test_metrics = run_final_eval(
        best_pt=best_pt,
        experiment_id=experiment_id,
        resolution=resolution,
        split="test",
    )

    summary = {
        "experiment_id": experiment_id,
        "model": "YOLOv8n",
        "resolution": resolution,
        "seed": SEED,
        "best_pt": str(best_pt),
        "training_time_h_measured_by_script": training_time_h,
        "val": val_metrics,
        "test": test_metrics,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_json = LOG_ROOT / experiment_id / "yolov8n_train_eval_summary.json"
    save_json(summary_json, summary)

    csv_row = {
        "experiment_id": experiment_id,
        "model": "YOLOv8n",
        "resolution": resolution,
        "seed": SEED,
        "best_pt": str(best_pt),
        "training_time_h_measured_by_script": training_time_h,
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_map50": val_metrics["map50"],
        "val_map50_95": val_metrics["map50_95"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_map50": test_metrics["map50"],
        "test_map50_95": test_metrics["map50_95"],
        "val_preprocess_ms_img": val_metrics["preprocess_ms_img"],
        "val_inference_ms_img": val_metrics["inference_ms_img"],
        "val_postprocess_ms_img": val_metrics["postprocess_ms_img"],
        "test_preprocess_ms_img": test_metrics["preprocess_ms_img"],
        "test_inference_ms_img": test_metrics["inference_ms_img"],
        "test_postprocess_ms_img": test_metrics["postprocess_ms_img"],
    }

    summary_csv = LOG_ROOT / "yolov8n_3res_seed42_train_eval_summary.csv"
    append_csv_row(summary_csv, csv_row)

    print("=" * 100)
    print(f"[DONE] {experiment_id}")
    print(f"[BEST_PT] {best_pt}")
    print(f"[SUMMARY JSON] {summary_json}")
    print(f"[SUMMARY CSV] {summary_csv}")
    print("=" * 100)


def main() -> None:
    print("=" * 100)
    print("YOLOv8n three-resolution cross-detector robustness experiment")
    print("=" * 100)

    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[DATA_YAML] {DATA_YAML}")
    print(f"[MODEL_WEIGHTS] {MODEL_WEIGHTS}")
    print(f"[RUNS_ROOT] {RUNS_ROOT}")
    print(f"[EVAL_ROOT] {EVAL_ROOT}")
    print(f"[LOG_ROOT] {LOG_ROOT}")
    print(f"[TARGET_RESOLUTIONS] {TARGET_RESOLUTIONS}")
    print(f"[CUDA] {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    check_dataset_layout()

    if not LOCAL_MODEL_WEIGHTS.exists():
        print(f"[WARN] Local YOLOv8n weight not found: {LOCAL_MODEL_WEIGHTS}")
        print("[WARN] Falling back to MODEL_WEIGHTS='yolov8n.pt'. Ultralytics may download it or use cache.")

    for exp in EXPERIMENTS:
        train_one_experiment(exp)

    print("=" * 100)
    print("[ALL DONE] YOLOv8n three-resolution experiment completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()