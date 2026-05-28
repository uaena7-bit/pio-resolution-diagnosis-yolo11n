#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare VisDrone2019-DET for YOLO format.

Expected input:
D:\Broiler chicken detection dataset\00_DATASET\VisDrone2019-DET\
    VisDrone2019-DET-train\images, annotations
    VisDrone2019-DET-val\images, annotations
    VisDrone2019-DET-test-dev\images, annotations

Output:
D:\Broiler chicken detection dataset\00_DATASET\VisDrone2019-DET-YOLO\
    images\train, images\val, images\test
    labels\train, labels\val, labels\test
    visdrone.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
RAW_ROOT = PROJECT_ROOT / r"00_DATASET\VisDrone2019-DET"
OUT_ROOT = PROJECT_ROOT / r"00_DATASET\VisDrone2019-DET-YOLO"
YAML_PATH = OUT_ROOT / "visdrone.yaml"

NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def find_split_dir(dirname: str) -> Path | None:
    p = RAW_ROOT / dirname
    if p.exists():
        return p
    matches = list(RAW_ROOT.rglob(dirname))
    return matches[0] if matches else None


def convert_one_annotation(ann_path: Path, img_path: Path, label_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as im:
        w_img, h_img = im.size

    out_lines = []
    ignored = 0

    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                ignored += 1
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                bw = float(parts[2])
                bh = float(parts[3])
                category = int(float(parts[5]))
            except Exception:
                ignored += 1
                continue

            # VisDrone: category 0 is ignored region, 1..10 are valid, 11 is others.
            if category < 1 or category > 10:
                ignored += 1
                continue
            if bw <= 1 or bh <= 1:
                ignored += 1
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(w_img), x + bw)
            y2 = min(float(h_img), y + bh)
            bw2 = x2 - x1
            bh2 = y2 - y1
            if bw2 <= 1 or bh2 <= 1:
                ignored += 1
                continue

            cls = category - 1
            xc = (x1 + x2) / 2.0 / w_img
            yc = (y1 + y2) / 2.0 / h_img
            wn = bw2 / w_img
            hn = bh2 / h_img
            out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return len(out_lines), ignored


def prepare_split(split: str, src_dir: Path) -> dict:
    src_images = src_dir / "images"
    src_anns = src_dir / "annotations"
    if not src_images.exists():
        raise FileNotFoundError(f"Missing images folder: {src_images}")
    if not src_anns.exists():
        raise FileNotFoundError(f"Missing annotations folder: {src_anns}")

    dst_images = OUT_ROOT / "images" / split
    dst_labels = OUT_ROOT / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in src_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_files:
        raise RuntimeError(f"No images found in {src_images}")

    n_boxes = 0
    n_ignored = 0

    for idx, img_path in enumerate(image_files, start=1):
        dst_img = dst_images / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        ann_path = src_anns / f"{img_path.stem}.txt"
        label_path = dst_labels / f"{img_path.stem}.txt"
        boxes, ignored = convert_one_annotation(ann_path, img_path, label_path)
        n_boxes += boxes
        n_ignored += ignored

        if idx % 500 == 0 or idx == len(image_files):
            print(f"[{split}] {idx}/{len(image_files)} images converted")

    return {
        "split": split,
        "images": len(image_files),
        "boxes": n_boxes,
        "ignored_or_invalid": n_ignored,
        "source": str(src_dir),
    }


def write_yaml() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    names_text = "\n".join([f"  {i}: {name}" for i, name in enumerate(NAMES)])
    text = (
        f"path: {OUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_text}\n"
    )
    YAML_PATH.write_text(text, encoding="utf-8")
    print(f"[WRITE] {YAML_PATH}")


def main() -> None:
    print("=" * 100)
    print("Prepare VisDrone2019-DET for YOLO")
    print(f"RAW_ROOT = {RAW_ROOT}")
    print(f"OUT_ROOT = {OUT_ROOT}")
    print("=" * 100)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {RAW_ROOT}")

    summaries = []
    for split, dirname in SPLITS.items():
        src_dir = find_split_dir(dirname)
        if src_dir is None:
            raise FileNotFoundError(f"Could not find split directory: {dirname}")
        print(f"[INFO] {split}: {src_dir}")
        summaries.append(prepare_split(split, src_dir))

    write_yaml()

    print("\nSummary:")
    for s in summaries:
        print(s)
    print("\nDONE.")


if __name__ == "__main__":
    main()
