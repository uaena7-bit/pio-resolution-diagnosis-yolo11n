#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import time
import gc
import statistics
import sys

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")

DATA_YAML = PROJECT_ROOT / r"00_DATASET\PIO-GRDB-MD5-7_1_2\dataset.yaml"

OUT_DIR = PROJECT_ROOT / r"02_RUNS\05_deployment_benchmark_FINAL_LOCKED_WEIGHTS"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly locked seed-42 YOLO11n weights.
# Important: 960 uses the non -2 directory.
MODEL_SPECS = [
    {
        "resolution": 800,
        "weight": PROJECT_ROOT / r"02_RUNS\00_M1_M3_resolution_ablation\M1_YOLO11n_PIO_GRDB_MD5_7_1_2_img800_seed42_SGD_e300_b2\weights\best.pt",
    },
    {
        "resolution": 960,
        "weight": PROJECT_ROOT / r"02_RUNS\00_M1_M3_resolution_ablation\M2_YOLO11n_PIO_GRDB_MD5_7_1_2_img960_seed42_SGD_e300_b2\weights\best.pt",
    },
    {
        "resolution": 1280,
        "weight": PROJECT_ROOT / r"02_RUNS\00_M1_M3_resolution_ablation\M3_YOLO11n_PIO_GRDB_MD5_7_1_2_img1280_seed42_SGD_e300_b2\weights\best.pt",
    },
]

SPLIT = "val"
BATCH = 2
DEVICE = 0
WORKERS = 0
CONF = 0.001
IOU = 0.70
MAX_DET = 1000
HALF = True

WARMUP_REPEATS = 1
FORMAL_REPEATS = 5


def check_environment():
    print("=" * 100)
    print("[ENVIRONMENT CHECK]")
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "This benchmark must be run under the conda 'yolo' GPU environment."
        )

    torch.cuda.set_device(DEVICE)
    print(f"CUDA device      : {torch.cuda.get_device_name(DEVICE)}")
    print("=" * 100)


def check_inputs():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"DATA_YAML not found: {DATA_YAML}")

    for spec in MODEL_SPECS:
        weight = spec["weight"]
        if not weight.exists():
            raise FileNotFoundError(f"Locked weight not found: {weight}")

        if "b2-2" in str(weight):
            raise ValueError(f"Forbidden b2-2 path detected: {weight}")

    print("[LOCKED WEIGHTS]")
    for spec in MODEL_SPECS:
        print(f"resolution={spec['resolution']}: {spec['weight']}")
    print("=" * 100)


def safe_get_params(model):
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return ""


def safe_get_gflops(model, imgsz):
    try:
        from ultralytics.utils.torch_utils import get_flops
        return float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        pass

    try:
        info = model.model.info(verbose=False, imgsz=imgsz)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            return float(info[3])
    except Exception:
        pass

    return ""


def safe_get_metrics(metrics):
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


def median_numeric(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(nums) if nums else ""


def prepare_cuda():
    torch.cuda.set_device(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def get_peak_memory_mib():
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
    return peak_allocated_mib, peak_reserved_mib


def run_one_val(model_path, resolution, repeat_idx, is_warmup):
    prepare_cuda()

    model = YOLO(str(model_path))

    params = safe_get_params(model)
    gflops = safe_get_gflops(model, resolution)

    start = time.perf_counter()

    metrics = model.val(
        data=str(DATA_YAML),
        split=SPLIT,
        imgsz=resolution,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        half=HALF,
        plots=False,
        save_json=False,
        project=str(OUT_DIR),
        name=f"tmp_{resolution}_repeat{repeat_idx}_{'warmup' if is_warmup else 'formal'}",
        exist_ok=True,
        verbose=False,
    )

    wall_time_s = time.perf_counter() - start

    speed = getattr(metrics, "speed", {})
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    total = preprocess + inference + postprocess

    peak_allocated_mib, peak_reserved_mib = get_peak_memory_mib()
    metric_values = safe_get_metrics(metrics)

    row = {
        "resolution": resolution,
        "repeat": repeat_idx,
        "warmup": int(is_warmup),
        "model_path": str(model_path),
        "params": params,
        "gflops": gflops,
        "precision": metric_values["precision"],
        "recall": metric_values["recall"],
        "map50": metric_values["map50"],
        "map50_95": metric_values["map50_95"],
        "preprocess_ms_img": preprocess,
        "inference_ms_img": inference,
        "postprocess_ms_img": postprocess,
        "total_ms_img": total,
        "wall_time_s": wall_time_s,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
    }

    del model
    cleanup_cuda()

    return row


def main():
    print("=" * 100)
    print("YOLO11n locked-weight deployment benchmark")
    print("=" * 100)

    check_environment()
    check_inputs()

    all_rows = []

    for spec in MODEL_SPECS:
        resolution = spec["resolution"]
        model_path = spec["weight"]

        print("\n" + "=" * 100)
        print(f"[RESOLUTION] {resolution}")
        print(f"[LOCKED WEIGHT] {model_path}")
        print("=" * 100)

        total_repeats = WARMUP_REPEATS + FORMAL_REPEATS

        for idx in range(total_repeats):
            repeat_label = idx + 1
            is_warmup = idx < WARMUP_REPEATS

            print(
                f"\n[RUN] resolution={resolution}, "
                f"repeat={repeat_label}/{total_repeats}, "
                f"warmup={is_warmup}"
            )

            row = run_one_val(
                model_path=model_path,
                resolution=resolution,
                repeat_idx=repeat_label,
                is_warmup=is_warmup,
            )

            all_rows.append(row)

            print(
                f"[RESULT] res={resolution}, warmup={is_warmup}, "
                f"P={row['precision']:.6f}, R={row['recall']:.6f}, "
                f"mAP50={row['map50']:.6f}, mAP50-95={row['map50_95']:.6f}, "
                f"pre={row['preprocess_ms_img']:.3f}, "
                f"inf={row['inference_ms_img']:.3f}, "
                f"post={row['postprocess_ms_img']:.3f}, "
                f"total={row['total_ms_img']:.3f}, "
                f"peak_alloc={row['peak_allocated_mib']:.1f} MiB, "
                f"peak_reserved={row['peak_reserved_mib']:.1f} MiB"
            )

    raw_csv = OUT_DIR / "deployment_benchmark_raw_repeats_locked_weights.csv"
    summary_csv = OUT_DIR / "deployment_benchmark_summary_median_locked_weights.csv"

    fieldnames = [
        "resolution",
        "repeat",
        "warmup",
        "model_path",
        "params",
        "gflops",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "preprocess_ms_img",
        "inference_ms_img",
        "postprocess_ms_img",
        "total_ms_img",
        "wall_time_s",
        "peak_allocated_mib",
        "peak_reserved_mib",
    ]

    with raw_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []

    for spec in MODEL_SPECS:
        resolution = spec["resolution"]
        rows = [
            r for r in all_rows
            if r["resolution"] == resolution and r["warmup"] == 0
        ]

        summary_rows.append({
            "resolution": resolution,
            "model_path": rows[0]["model_path"],
            "params": rows[0]["params"],
            "gflops": rows[0]["gflops"],
            "precision_median": median_numeric([r["precision"] for r in rows]),
            "recall_median": median_numeric([r["recall"] for r in rows]),
            "map50_median": median_numeric([r["map50"] for r in rows]),
            "map50_95_median": median_numeric([r["map50_95"] for r in rows]),
            "preprocess_median_ms_img": median_numeric([r["preprocess_ms_img"] for r in rows]),
            "inference_median_ms_img": median_numeric([r["inference_ms_img"] for r in rows]),
            "postprocess_median_ms_img": median_numeric([r["postprocess_ms_img"] for r in rows]),
            "total_median_ms_img": median_numeric([r["total_ms_img"] for r in rows]),
            "peak_allocated_median_mib": median_numeric([r["peak_allocated_mib"] for r in rows]),
            "peak_reserved_median_mib": median_numeric([r["peak_reserved_mib"] for r in rows]),
            "wall_time_median_s": median_numeric([r["wall_time_s"] for r in rows]),
            "formal_repeats": len(rows),
            "warmup_repeats": WARMUP_REPEATS,
            "split": SPLIT,
            "batch": BATCH,
            "workers": WORKERS,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "half": HALF,
        })

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 100)
    print("[DONE] Locked-weight deployment benchmark completed.")
    print("=" * 100)
    print(f"Raw repeats saved to: {raw_csv}")
    print(f"Median summary saved to: {summary_csv}")

    print("\n[SUMMARY]")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
