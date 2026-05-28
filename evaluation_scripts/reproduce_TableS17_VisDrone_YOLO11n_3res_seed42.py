#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11n VisDrone2019-DET cross-domain resolution demonstration.

Runs:
- YOLO11n
- resolutions: 800, 960, 1280
- seed=42
- epochs=100
- batch=2
- optimizer=SGD
- evaluates both val and test-dev splits when available in visdrone.yaml

This is a protocol-transfer demonstration, not a VisDrone SOTA benchmark.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
DATA_YAML = PROJECT_ROOT / r"00_DATASET\VisDrone2019-DET-YOLO\visdrone.yaml"
OUT_PROJECT = PROJECT_ROOT / r"02_RUNS\08_cross_dataset_VisDrone_YOLO11n_resolution"
SUMMARY_CSV = PROJECT_ROOT / r"06_LOGS\supplementary_source_data\TableS17_VisDrone_YOLO11n_cross_domain_resolution_demo.csv"

RESOLUTIONS = [800, 960, 1280]
SEED = 42
EPOCHS = 100
BATCH = 2
WORKERS = 0
OPTIMIZER = "SGD"


def metric(metrics, name: str) -> float:
    try:
        return float(getattr(metrics.box, name))
    except Exception:
        return float("nan")


def speed_metric(metrics, name: str) -> float:
    speed = getattr(metrics, "speed", {}) or {}
    try:
        return float(speed.get(name, float("nan")))
    except Exception:
        return float("nan")


def evaluate(best_pt: Path, resolution: int, split: str) -> dict:
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=resolution,
        batch=BATCH,
        workers=WORKERS,
        conf=0.001,
        iou=0.70,
        max_det=1000,
        device=0 if torch.cuda.is_available() else "cpu",
        half=True if torch.cuda.is_available() else False,
        save_json=False,
        plots=False,
        verbose=True,
    )
    return {
        f"{split}_precision": f"{metric(metrics, 'mp'):.6f}",
        f"{split}_recall": f"{metric(metrics, 'mr'):.6f}",
        f"{split}_map50": f"{metric(metrics, 'map50'):.6f}",
        f"{split}_map50_95": f"{metric(metrics, 'map'):.6f}",
        f"{split}_preprocess_ms_img": f"{speed_metric(metrics, 'preprocess'):.6f}",
        f"{split}_inference_ms_img": f"{speed_metric(metrics, 'inference'):.6f}",
        f"{split}_postprocess_ms_img": f"{speed_metric(metrics, 'postprocess'):.6f}",
    }


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {DATA_YAML}")

    print("=" * 100)
    print("VisDrone YOLO11n cross-domain resolution demonstration")
    print(f"DATA_YAML = {DATA_YAML}")
    print(f"OUT_PROJECT = {OUT_PROJECT}")
    print(f"CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU = {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    rows = []
    for resolution in RESOLUTIONS:
        run_name = f"VD_YOLO11n_img{resolution}_seed{SEED}_SGD_e{EPOCHS}_b{BATCH}"
        print("\n" + "=" * 100)
        print(f"[TRAIN] resolution={resolution}, run={run_name}")
        print("=" * 100)

        model = YOLO("yolo11n.pt")
        t0 = time.perf_counter()
        model.train(
            data=str(DATA_YAML),
            imgsz=resolution,
            epochs=EPOCHS,
            batch=BATCH,
            workers=WORKERS,
            optimizer=OPTIMIZER,
            seed=SEED,
            project=str(OUT_PROJECT),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            plots=False,
            verbose=True,
        )
        training_time_h = (time.perf_counter() - t0) / 3600.0

        best_pt = OUT_PROJECT / run_name / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Missing best.pt: {best_pt}")

        row = {
            "dataset": "VisDrone2019-DET",
            "detector": "YOLO11n",
            "resolution": resolution,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch": BATCH,
            "workers": WORKERS,
            "optimizer": OPTIMIZER,
            "training_time_h": f"{training_time_h:.6f}",
            "best_pt": str(best_pt),
            "note": "Cross-domain protocol demonstration only; not a VisDrone SOTA benchmark.",
        }

        row.update(evaluate(best_pt, resolution, "val"))
        row.update(evaluate(best_pt, resolution, "test"))

        rows.append(row)

        SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"[WRITE] {SUMMARY_CSV}")
        print(row)

    print("\nDONE.")
    print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
