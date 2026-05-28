#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisDrone2019-DET split-leakage audit.

Purpose:
- Check whether the official VisDrone train/val/test-dev splits used in the
  cross-domain demonstration contain obvious cross-split leakage.
- This is an audit for the external demonstration only, not a replacement for
  the full PIO leakage-controlled split construction.

Checks:
1. basename overlap across splits
2. exact image-content MD5 overlap across splits
3. exact annotation-content MD5 overlap across splits
4. coarse near-duplicate candidates using dHash Hamming distance
5. optional SSIM confirmation for dHash candidates

Expected input:
D:\Broiler chicken detection dataset\00_DATASET\VisDrone2019-DET\
    VisDrone2019-DET-train\images, annotations
    VisDrone2019-DET-val\images, annotations
    VisDrone2019-DET-test-dev\images, annotations

Output:
D:\Broiler chicken detection dataset\06_LOGS\visdrone_leakage_audit\
    VisDrone_split_leakage_audit_summary.csv
    VisDrone_cross_split_exact_duplicate_images.csv
    VisDrone_cross_split_basename_overlap.csv
    VisDrone_cross_split_annotation_md5_overlap.csv
    VisDrone_cross_split_near_duplicate_candidates.csv
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
import time
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np


PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
RAW_ROOT = PROJECT_ROOT / r"00_DATASET\VisDrone2019-DET"
OUT_DIR = PROJECT_ROOT / r"06_LOGS\visdrone_leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": RAW_ROOT / "VisDrone2019-DET-train",
    "val": RAW_ROOT / "VisDrone2019-DET-val",
    "test-dev": RAW_ROOT / "VisDrone2019-DET-test-dev",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DHASH_SIZE = 8
DHASH_THRESHOLD = 6
SSIM_THRESHOLD = 0.95
MAX_SSIM_CONFIRMATIONS = 5000  # safety guard


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = DHASH_SIZE) -> int:
    # Difference hash, returns 64-bit integer by default.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im = im.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simple_ssim(path_a: Path, path_b: Path, size: int = 256) -> float:
    # Lightweight grayscale SSIM implementation on resized images.
    with Image.open(path_a) as ia:
        ia = ImageOps.exif_transpose(ia).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        xa = np.asarray(ia, dtype=np.float64)
    with Image.open(path_b) as ib:
        ib = ImageOps.exif_transpose(ib).convert("L").resize((size, size), Image.Resampling.BILINEAR)
        xb = np.asarray(ib, dtype=np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = xa.mean()
    mu_y = xb.mean()
    var_x = xa.var()
    var_y = xb.var()
    cov_xy = ((xa - mu_x) * (xb - mu_y)).mean()
    return float(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))


def collect_images() -> list[dict]:
    rows = []
    for split, root in SPLITS.items():
        img_dir = root / "images"
        ann_dir = root / "annotations"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {img_dir}")
        if not ann_dir.exists():
            raise FileNotFoundError(f"Missing annotations folder: {ann_dir}")

        image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        for idx, img_path in enumerate(image_paths, start=1):
            ann_path = ann_dir / f"{img_path.stem}.txt"
            row = {
                "split": split,
                "image_path": str(img_path),
                "basename": img_path.name,
                "stem": img_path.stem,
                "image_md5": file_md5(img_path),
                "annotation_path": str(ann_path) if ann_path.exists() else "",
                "annotation_md5": file_md5(ann_path) if ann_path.exists() else "",
                "dhash": dhash(img_path),
            }
            rows.append(row)
            if idx % 500 == 0 or idx == len(image_paths):
                print(f"[COLLECT] {split}: {idx}/{len(image_paths)}")
    return rows


def cross_split_pairs(rows: list[dict], key: str) -> list[dict]:
    by_key = {}
    for r in rows:
        if not r[key]:
            continue
        by_key.setdefault(r[key], []).append(r)

    out = []
    for value, items in by_key.items():
        splits = sorted({r["split"] for r in items})
        if len(splits) <= 1:
            continue
        for a, b in itertools.combinations(items, 2):
            if a["split"] == b["split"]:
                continue
            out.append({
                "key": value,
                "split_a": a["split"],
                "image_a": a["image_path"],
                "split_b": b["split"],
                "image_b": b["image_path"],
            })
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)
    print(f"[WRITE] {path} ({len(rows)} rows)")


def near_duplicate_candidates(rows: list[dict]) -> list[dict]:
    # Bucket by high 16 bits of dHash to reduce comparisons, then compare cross-split.
    # This is conservative and fast; exact dHash neighbors with different high bits may be missed.
    buckets = {}
    for r in rows:
        prefix = int(r["dhash"]) >> 48
        buckets.setdefault(prefix, []).append(r)

    candidates = []
    confirmations = 0

    for prefix, items in buckets.items():
        if len(items) < 2:
            continue
        for a, b in itertools.combinations(items, 2):
            if a["split"] == b["split"]:
                continue
            dist = hamming(int(a["dhash"]), int(b["dhash"]))
            if dist <= DHASH_THRESHOLD:
                ssim = ""
                confirmed = False
                if confirmations < MAX_SSIM_CONFIRMATIONS:
                    try:
                        ssim_val = simple_ssim(Path(a["image_path"]), Path(b["image_path"]))
                        ssim = f"{ssim_val:.6f}"
                        confirmed = ssim_val >= SSIM_THRESHOLD
                    except Exception as e:
                        ssim = f"ERROR: {e}"
                    confirmations += 1
                candidates.append({
                    "split_a": a["split"],
                    "image_a": a["image_path"],
                    "split_b": b["split"],
                    "image_b": b["image_path"],
                    "dhash_hamming": dist,
                    "ssim_256": ssim,
                    "confirmed_ssim_ge_0_95": confirmed,
                })

    return candidates


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 100)
    print("VisDrone2019-DET split-leakage audit")
    print(f"RAW_ROOT = {RAW_ROOT}")
    print(f"OUT_DIR  = {OUT_DIR}")
    print("=" * 100)

    rows = collect_images()

    basename_overlap = cross_split_pairs(rows, "basename")
    image_md5_overlap = cross_split_pairs(rows, "image_md5")
    ann_md5_overlap = cross_split_pairs(rows, "annotation_md5")
    near_dups = near_duplicate_candidates(rows)

    # Filter confirmed near duplicates.
    confirmed_near = [r for r in near_dups if str(r["confirmed_ssim_ge_0_95"]) == "True"]

    summary_rows = []
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        summary_rows.append({
            "split": split,
            "images": len(subset),
            "unique_basenames": len({r["basename"] for r in subset}),
            "unique_image_md5": len({r["image_md5"] for r in subset}),
            "unique_annotation_md5": len({r["annotation_md5"] for r in subset if r["annotation_md5"]}),
        })

    summary_rows.append({
        "split": "cross_split_basename_overlap_pairs",
        "images": len(basename_overlap),
        "unique_basenames": "",
        "unique_image_md5": "",
        "unique_annotation_md5": "",
    })
    summary_rows.append({
        "split": "cross_split_image_md5_duplicate_pairs",
        "images": len(image_md5_overlap),
        "unique_basenames": "",
        "unique_image_md5": "",
        "unique_annotation_md5": "",
    })
    summary_rows.append({
        "split": "cross_split_annotation_md5_duplicate_pairs",
        "images": len(ann_md5_overlap),
        "unique_basenames": "",
        "unique_image_md5": "",
        "unique_annotation_md5": "",
    })
    summary_rows.append({
        "split": f"cross_split_near_duplicate_candidates_dhash_le_{DHASH_THRESHOLD}",
        "images": len(near_dups),
        "unique_basenames": "",
        "unique_image_md5": "",
        "unique_annotation_md5": "",
    })
    summary_rows.append({
        "split": f"cross_split_confirmed_near_duplicates_ssim_ge_{SSIM_THRESHOLD}",
        "images": len(confirmed_near),
        "unique_basenames": "",
        "unique_image_md5": "",
        "unique_annotation_md5": "",
    })

    write_csv(OUT_DIR / "VisDrone_split_leakage_audit_summary.csv", summary_rows)
    write_csv(OUT_DIR / "VisDrone_cross_split_basename_overlap.csv", basename_overlap,
              ["key", "split_a", "image_a", "split_b", "image_b"])
    write_csv(OUT_DIR / "VisDrone_cross_split_exact_duplicate_images.csv", image_md5_overlap,
              ["key", "split_a", "image_a", "split_b", "image_b"])
    write_csv(OUT_DIR / "VisDrone_cross_split_annotation_md5_overlap.csv", ann_md5_overlap,
              ["key", "split_a", "image_a", "split_b", "image_b"])
    write_csv(OUT_DIR / "VisDrone_cross_split_near_duplicate_candidates.csv", near_dups,
              ["split_a", "image_a", "split_b", "image_b", "dhash_hamming", "ssim_256", "confirmed_ssim_ge_0_95"])

    print("=" * 100)
    print("[DONE] VisDrone audit completed")
    print(f"Elapsed: {(time.perf_counter() - t0):.1f} s")
    print(f"basename overlap pairs: {len(basename_overlap)}")
    print(f"image MD5 duplicate pairs: {len(image_md5_overlap)}")
    print(f"annotation MD5 duplicate pairs: {len(ann_md5_overlap)}")
    print(f"near-duplicate candidates dHash<={DHASH_THRESHOLD}: {len(near_dups)}")
    print(f"confirmed near duplicates SSIM>={SSIM_THRESHOLD}: {len(confirmed_near)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
