# -*- coding: utf-8 -*-
"""
Figure 5 qualitative resolution-comparison generator.

This repository version is path-parameterized and does not redistribute raw PIO
images, model weights, or prediction overlays. To reproduce the qualitative
figure, download the original PIO dataset and provide local paths to the dataset
YAML and prediction-cache CSV files.

The companion CSV `figure_source_data/Fig5_qualitative_cases_metadata.csv`
records the fixed qualitative cases and the TP/FN/FP counts reported in the
manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


PROJECT_ROOT = Path(".").resolve()
DATA_YAML = PROJECT_ROOT / "00_DATASET" / "PIO-GRDB-MD5-7_1_2" / "dataset.yaml"
PRED_CACHE_DIR = PROJECT_ROOT / "04_AUDIT" / "04_groupwise_evaluation" / "prediction_cache"
OUT_DIR = PROJECT_ROOT / "figures"
GROUP_DEFS_PATH = PROJECT_ROOT / "group_definitions" / "P0_P1_group_definitions_test.json"

EXPERIMENTS = [
    {
        "key": "M1",
        "name": "M1_800_seed42",
        "short": "800",
        "imgsz": 800,
        "cache_glob": "M1_800_seed42_test_imgsz800_conf0.001_iou0.7_maxdet1000_cls0_fp16_cuda_FINAL_CANONICAL_IMAGE_ID_V4_predictions.csv",
    },
    {
        "key": "M2",
        "name": "M2_960_seed42",
        "short": "960",
        "imgsz": 960,
        "cache_glob": "M2_960_seed42_test_imgsz960_conf0.001_iou0.7_maxdet1000_cls0_fp16_cuda_FINAL_CANONICAL_IMAGE_ID_V4_predictions.csv",
    },
    {
        "key": "M3",
        "name": "M3_1280_seed42",
        "short": "1280",
        "imgsz": 1280,
        "cache_glob": "M3_1280_seed42_test_imgsz1280_conf0.001_iou0.7_maxdet1000_cls0_fp32_cpu_FINAL_CANONICAL_IMAGE_ID_V4_predictions.csv",
    },
]

# Fixed three images for the final paper figure
FIXED_CASES = [
    {
        "row_name": "Case A",
        "filename": "val__C-W2-V0007.jpg",
        "display_title": "Case A: 960 markedly reduces missed detections in an ultra-high-density scene",
        "caption_short": "ultra_high density, 571 instances",
        "case_type": "A",
    },
    {
        "row_name": "Case B",
        "filename": "train__K-513.jpg",
        "display_title": "Case B: 960 and 1280 produce nearly identical detections",
        "caption_short": "ultra_high density, 602 instances",
        "case_type": "B",
    },
    {
        "row_name": "Case C",
        "filename": "val__C-W2-V0036.jpg",
        "display_title": "Case C: 960 improves substantially over 800, while 1280 is only marginally better",
        "caption_short": "ultra_high density, fixed third-row replacement",
        "case_type": "A",  # use Case A style cropping logic
    },
]


@dataclass
class GTBox:
    idx: int
    image_id: str
    cls: int
    xyxy: np.ndarray
    scale_group: str
    density_group: str


@dataclass
class PredBox:
    image_id: str
    cls: int
    xyxy: np.ndarray
    conf: float


def log(msg: str) -> None:
    print(f"[FINAL-FIG5] {msg}", flush=True)


def read_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(data_yaml: Path, raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    data = read_yaml(data_yaml)
    base = data_yaml.parent
    if data.get("path"):
        base = Path(data["path"])
        if not base.is_absolute():
            base = data_yaml.parent / base
    for p in [base / raw, data_yaml.parent / raw, PROJECT_ROOT / raw]:
        if p.exists():
            return p
    return base / raw


def load_image_list(data_yaml: Path, split: str) -> List[Path]:
    data = read_yaml(data_yaml)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    split_path = resolve_dataset_path(data_yaml, data[split])
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            s = line.strip().strip('"').strip("'")
            if not s:
                continue
            p = Path(s)
            if not p.is_absolute():
                candidates = [
                    split_path.parent / p,
                    data_yaml.parent / p,
                    resolve_dataset_path(data_yaml, p),
                    PROJECT_ROOT / p,
                ]
                p = next((c for c in candidates if c.exists()), candidates[0])
            images.append(p.resolve())
    elif split_path.is_dir():
        images = [p.resolve() for p in split_path.rglob("*") if p.suffix.lower() in exts]
    else:
        raise FileNotFoundError(f"Cannot resolve split path: {split_path}")

    images = sorted(set(images))
    if not images:
        raise RuntimeError(f"No images found for split={split}")
    return images


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lower = [x.lower() for x in parts]
    if "images" in lower:
        i = lower.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, w: int, h: int) -> np.ndarray:
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return np.array(
        [np.clip(x1, 0, w), np.clip(y1, 0, h), np.clip(x2, 0, w), np.clip(y2, 0, h)],
        dtype=np.float32,
    )


def assign_group(v: float, q1: float, q2: float, q3: float, names: Tuple[str, str, str, str]) -> str:
    if v <= q1:
        return names[0]
    if v <= q2:
        return names[1]
    if v <= q3:
        return names[2]
    return names[3]


def load_group_defs(gt_df: pd.DataFrame, density_df: pd.DataFrame) -> Dict:
    group_path = GROUP_DEFS_PATH
    if group_path.exists():
        log(f"Loading group definitions: {group_path}")
        return json.loads(group_path.read_text(encoding="utf-8"))

    log("Group definition file not found. Computing quartiles from current split.")
    sq1, sq2, sq3 = np.quantile(gt_df["sqrt_area"].to_numpy(np.float64), [0.25, 0.50, 0.75]).tolist()
    dq1, dq2, dq3 = np.quantile(density_df["num_instances"].to_numpy(np.float64), [0.25, 0.50, 0.75]).tolist()
    return {
        "scale": {"q1": sq1, "q2": sq2, "q3": sq3},
        "density": {"q1": dq1, "q2": dq2, "q3": dq3},
    }


def collect_gt(image_paths: List[Path], class_id: Optional[int]) -> Tuple[List[GTBox], pd.DataFrame]:
    raw_gt_rows = []
    density_rows = []

    for img in tqdm(image_paths, desc="Read GT"):
        w, h = image_size(img)
        image_id = str(img.resolve())
        label = image_to_label_path(img)
        lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []

        count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if class_id is not None and cls != class_id:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            xyxy = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            area = max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))
            raw_gt_rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_id,
                    "cls": cls,
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "area": area,
                    "sqrt_area": math.sqrt(area),
                    "img_w": w,
                    "img_h": h,
                }
            )
            count += 1

        density_rows.append({"image_id": image_id, "image_path": image_id, "num_instances": count, "img_w": w, "img_h": h})

    gt_df = pd.DataFrame(raw_gt_rows)
    density_df = pd.DataFrame(density_rows)
    if gt_df.empty:
        raise RuntimeError("No GT boxes found.")

    defs = load_group_defs(gt_df, density_df)
    sq1, sq2, sq3 = defs["scale"]["q1"], defs["scale"]["q2"], defs["scale"]["q3"]
    dq1, dq2, dq3 = defs["density"]["q1"], defs["density"]["q2"], defs["density"]["q3"]

    gt_df["scale_group"] = gt_df["sqrt_area"].apply(
        lambda x: assign_group(x, sq1, sq2, sq3, ("tiny", "small", "medium", "large"))
    )
    density_df["density_group"] = density_df["num_instances"].apply(
        lambda x: assign_group(x, dq1, dq2, dq3, ("low", "medium", "high", "ultra_high"))
    )
    img_to_density = dict(zip(density_df["image_id"], density_df["density_group"]))
    gt_df["density_group"] = gt_df["image_id"].map(img_to_density)

    gt_list = []
    for idx, r in enumerate(gt_df.itertuples(index=False)):
        gt_list.append(
            GTBox(
                idx=idx,
                image_id=str(r.image_id),
                cls=int(r.cls),
                xyxy=np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float32),
                scale_group=str(r.scale_group),
                density_group=str(r.density_group),
            )
        )
    return gt_list, density_df


def find_cache(exp: Dict, split: str) -> Path:
    exact = PRED_CACHE_DIR / exp["cache_glob"].replace("_test_", f"_{split}_")
    if exact.exists():
        return exact

    pattern = f"{exp['name']}_{split}_imgsz{exp['imgsz']}*FINAL_CANONICAL_IMAGE_ID_V4_predictions.csv"
    matches = sorted(PRED_CACHE_DIR.glob(pattern))
    if matches:
        return matches[-1]

    raise FileNotFoundError(
        f"Prediction cache not found for {exp['name']} and split={split}.\nPattern: {pattern}"
    )


def load_predictions(split: str, class_id: Optional[int]) -> Dict[str, List[PredBox]]:
    preds_by_exp = {}
    for exp in EXPERIMENTS:
        cache = find_cache(exp, split)
        log(f"Loading predictions: {exp['name']} <- {cache.name}")
        df = pd.read_csv(cache)
        if class_id is not None and "cls" in df.columns:
            df = df[df["cls"].astype(int) == class_id]

        # Load the full FINAL_SAFE cache. Metric filtering and drawing filtering are applied separately.
        preds = []
        for r in df.itertuples(index=False):
            preds.append(
                PredBox(
                    image_id=str(r.image_id),
                    cls=int(r.cls),
                    xyxy=np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float32),
                    conf=float(r.conf),
                )
            )
        preds_by_exp[exp["name"]] = preds
        log(f"  loaded predictions: {len(preds)}")
    return preds_by_exp


def filter_preds(preds: List[PredBox], conf_thr: float) -> List[PredBox]:
    """Filter predictions by confidence for metric calculation or drawing."""
    return [p for p in preds if p.conf >= float(conf_thr)]


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    a2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-16)


def match_one_image(gt_boxes: List[GTBox], pred_boxes: List[PredBox], iou_thr: float) -> Dict:
    pred_sorted = sorted(pred_boxes, key=lambda p: p.conf, reverse=True)
    matched_gt: Set[int] = set()
    matched_pairs = []

    gt_by_cls: Dict[int, List[GTBox]] = {}
    for g in gt_boxes:
        gt_by_cls.setdefault(g.cls, []).append(g)

    for p in pred_sorted:
        candidates = gt_by_cls.get(p.cls, [])
        best_gt = None
        best_iou = -1.0
        if candidates:
            boxes = np.asarray([g.xyxy for g in candidates], dtype=np.float32)
            ious = box_iou_one_to_many(p.xyxy, boxes)
            for j, g in enumerate(candidates):
                if g.idx in matched_gt:
                    continue
                val = float(ious[j])
                if val > best_iou:
                    best_iou = val
                    best_gt = g

        if best_gt is not None and best_iou >= iou_thr:
            matched_gt.add(best_gt.idx)
            matched_pairs.append((best_gt.idx, best_iou, best_gt.scale_group))

    n_gt = len(gt_boxes)
    tp = len(matched_gt)
    fp = max(0, len(pred_sorted) - tp)
    fn = max(0, n_gt - tp)

    scale_counts = {s: 0 for s in ["tiny", "small", "medium", "large"]}
    scale_tp = {s: 0 for s in ["tiny", "small", "medium", "large"]}
    scale_iou_sum = {s: 0.0 for s in ["tiny", "small", "medium", "large"]}

    for g in gt_boxes:
        scale_counts[g.scale_group] += 1
    for gt_idx, iou, sg in matched_pairs:
        scale_tp[sg] += 1
        scale_iou_sum[sg] += float(iou)

    out = {
        "n_gt": n_gt,
        "n_pred": len(pred_sorted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / n_gt if n_gt else 0.0,
        "precision": tp / len(pred_sorted) if len(pred_sorted) else 0.0,
        "matched_gt_ids": matched_gt,
        "matched_pairs": matched_pairs,
    }

    for s in ["tiny", "small", "medium", "large"]:
        out[f"{s}_gt"] = scale_counts[s]
        out[f"{s}_tp"] = scale_tp[s]
        out[f"{s}_recall"] = scale_tp[s] / scale_counts[s] if scale_counts[s] else 0.0
        out[f"{s}_mean_iou"] = scale_iou_sum[s] / scale_tp[s] if scale_tp[s] else 0.0

    return out


def safe_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def box_intersects_crop(box: np.ndarray, crop: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = crop
    return not (box[2] < x1 or box[0] > x2 or box[3] < y1 or box[1] > y2)


def shift_box(box: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, _, _ = crop
    return np.array([box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1], dtype=np.float32)


def crop_from_interesting_gt(img_w: int, img_h: int, gt_boxes: List[GTBox], gt_ids: Set[int], min_size: int = 760, pad: int = 140) -> Optional[Tuple[int, int, int, int]]:
    boxes = [g.xyxy for g in gt_boxes if g.idx in gt_ids]
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=np.float32)
    x1, y1 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x2, y2 = float(arr[:, 2].max()), float(arr[:, 3].max())

    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    w = max(w, min_size)
    h = max(h, min_size)
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def find_dense_crop(img_w: int, img_h: int, gt_boxes: List[GTBox], crop_size: int = 900, stride: int = 180) -> Optional[Tuple[int, int, int, int]]:
    if not gt_boxes:
        return None

    centers = np.asarray(
        [[(g.xyxy[0] + g.xyxy[2]) / 2.0, (g.xyxy[1] + g.xyxy[3]) / 2.0] for g in gt_boxes],
        dtype=np.float32,
    )
    crop_w = min(int(crop_size), img_w)
    crop_h = min(int(crop_size), img_h)

    xs = list(range(0, max(1, img_w - crop_w + 1), max(1, stride)))
    ys = list(range(0, max(1, img_h - crop_h + 1), max(1, stride)))
    if xs[-1] != max(0, img_w - crop_w):
        xs.append(max(0, img_w - crop_w))
    if ys[-1] != max(0, img_h - crop_h):
        ys.append(max(0, img_h - crop_h))

    best = None
    best_score = -1
    for x1 in xs:
        for y1 in ys:
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            inside = (
                (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
            )
            count = int(inside.sum())
            cx = x1 + crop_w / 2.0
            cy = y1 + crop_h / 2.0
            center_bias = -abs(cx - img_w / 2.0) * 0.001 - abs(cy - img_h / 2.0) * 0.001
            score = count + center_bias
            if score > best_score:
                best_score = score
                best = (int(x1), int(y1), int(x2), int(y2))
    return best


def draw_panel(
    img: Image.Image,
    title: str,
    gt_boxes: List[GTBox],
    pred_boxes: List[PredBox],
    crop: Optional[Tuple[int, int, int, int]],
    draw_gt: bool,
    draw_conf: float,
    max_draw: int,
    target_w: int = 640,
) -> Image.Image:
    if crop is not None:
        panel = img.crop(crop).convert("RGB")
    else:
        panel = img.convert("RGB")

    scale = target_w / panel.width
    target_h = int(round(panel.height * scale))
    panel = panel.resize((target_w, target_h))

    draw = ImageDraw.Draw(panel)
    font = safe_font(18)
    small_font = safe_font(14)

    if crop is None:
        crop0 = (0, 0, img.width, img.height)
    else:
        crop0 = crop

    def transform(b: np.ndarray) -> Tuple[int, int, int, int]:
        if crop is not None:
            b = shift_box(b, crop0)
        return (
            int(round(b[0] * scale)),
            int(round(b[1] * scale)),
            int(round(b[2] * scale)),
            int(round(b[3] * scale)),
        )

    if draw_gt:
        for g in gt_boxes:
            if crop is not None and not box_intersects_crop(g.xyxy, crop0):
                continue
            xy = transform(g.xyxy)
            draw.rectangle(xy, outline=(255, 220, 0), width=1)

    drawn = 0
    for p in sorted(pred_boxes, key=lambda x: x.conf, reverse=True):
        if p.conf < draw_conf:
            continue
        if crop is not None and not box_intersects_crop(p.xyxy, crop0):
            continue
        xy = transform(p.xyxy)
        draw.rectangle(xy, outline=(0, 255, 255), width=2)
        drawn += 1
        if drawn >= max_draw:
            break

    draw.rectangle((0, 0, panel.width, 32), fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(255, 255, 255), font=font)
    draw.text((8, panel.height - 20), "GT yellow | Pred cyan", fill=(255, 255, 255), font=small_font)
    return panel


def find_image_path(image_paths: List[Path], filename: str) -> Path:
    matches = [p for p in image_paths if p.name == filename]
    if not matches:
        raise FileNotFoundError(f"Cannot find fixed image in split list: {filename}")
    if len(matches) > 1:
        log(f"Warning: multiple matches for {filename}, using first one.")
    return matches[0]


def build_metrics_for_image(
    image_id: str,
    gt_boxes: List[GTBox],
    preds_by_exp: Dict[str, List[PredBox]],
    metric_conf: float,
) -> Dict:
    out = {}
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_preds = [p for p in preds_by_exp[exp_name] if p.image_id == image_id and p.conf >= float(metric_conf)]
        out[exp_name] = match_one_image(gt_boxes, exp_preds, iou_thr=0.50)
    return out


def choose_crop(case_type: str, img: Image.Image, gt_boxes: List[GTBox], metrics: Dict) -> Optional[Tuple[int, int, int, int]]:
    if case_type == "A":
        m1 = metrics["M1_800_seed42"]
        m2 = metrics["M2_960_seed42"]
        interesting = set(m2["matched_gt_ids"]) - set(m1["matched_gt_ids"])
        crop = crop_from_interesting_gt(img.width, img.height, gt_boxes, interesting, min_size=760, pad=140)
        if crop is None:
            crop = find_dense_crop(img.width, img.height, gt_boxes, crop_size=900, stride=180)
        return crop
    elif case_type == "B":
        return find_dense_crop(img.width, img.height, gt_boxes, crop_size=900, stride=180)
    else:
        return find_dense_crop(img.width, img.height, gt_boxes, crop_size=900, stride=180)



def _animals_strip_panel_title_bar(panel: Image.Image) -> Image.Image:
    """
    Remove the dark title strip produced by the old draw_panel() function.
    If no dark title strip is detected, the original panel is returned.
    """
    from PIL import ImageStat

    img = panel.convert("RGB")
    gray = img.convert("L")

    max_scan = min(90, max(20, img.height // 5))
    means = []
    for y in range(max_scan):
        row = gray.crop((0, y, img.width, y + 1))
        means.append(ImageStat.Stat(row).mean[0])

    # Old panel title bar is nearly black at the top.
    dark_top = sum(m < 75 for m in means[:12]) >= 6
    if not dark_top:
        return img

    cut = None
    for y, m in enumerate(means):
        if y > 8 and m > 110:
            cut = y
            break

    if cut is not None and 12 <= cut <= 85:
        return img.crop((0, cut, img.width, img.height))

    return img



def _animals_strip_panel_footer_note(panel: Image.Image) -> Image.Image:
    """
    Remove the old per-panel footer note such as 'GT yellow | Pred cyan'.
    The global legend at the bottom of the full figure is kept instead.
    """
    img = panel.convert("RGB")

    # Crop a very small bottom strip where the old footer note was drawn.
    # This avoids duplicate tiny text in every panel.
    crop_h = 22
    if img.height > 120:
        return img.crop((0, 0, img.width, img.height - crop_h))

    return img


def _animals_text_bbox(draw: ImageDraw.ImageDraw, xy, text: str, font):
    """
    Pillow compatibility helper.
    """
    if hasattr(draw, "textbbox"):
        return draw.textbbox(xy, text, font=font)
    w, h = draw.textsize(text, font=font)
    x, y = xy
    return (x, y, x + w, y + h)


def _animals_add_panel_label(panel: Image.Image, label: str) -> Image.Image:
    """
    Add a clean white label box in the upper-left corner.
    """
    img = panel.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    font = safe_font(18)
    lines = label.splitlines()

    pad_x = 8
    pad_y = 6
    line_gap = 3

    line_boxes = [_animals_text_bbox(draw, (0, 0), line, font) for line in lines]
    line_ws = [b[2] - b[0] for b in line_boxes]
    line_hs = [b[3] - b[1] for b in line_boxes]

    box_w = max(line_ws) + 2 * pad_x
    box_h = sum(line_hs) + line_gap * (len(lines) - 1) + 2 * pad_y

    x0, y0 = 8, 8
    x1, y1 = x0 + box_w, y0 + box_h

    # Semi-transparent white label background.
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 218))

    y = y0 + pad_y
    for line, lh in zip(lines, line_hs):
        draw.text((x0 + pad_x, y), line, fill=(0, 0, 0, 255), font=font)
        y += lh + line_gap

    return Image.alpha_composite(img, overlay).convert("RGB")


def compose_final_figure(rows: List[Image.Image], out_path: Path) -> None:
    """
    Manuscript layout:
    - clean white background
    - compact row spacing
    - one global legend
    - PNG + TIFF + PDF outputs
    """
    row_gap = 14
    legend_h = 42

    max_w = max(im.width for im in rows)
    total_h = sum(im.height for im in rows) + row_gap * (len(rows) - 1) + legend_h

    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for im in rows:
        x = (max_w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + row_gap

    # One clean global legend at the bottom.
    # Detailed TP/FN/FP protocol is reported in the figure caption, not inside the figure.
    legend_font = safe_font(18)
    legend_y = total_h - legend_h + 10

    # Centered legend group.
    legend_text_w = 300
    x0 = max(14, (max_w - legend_text_w) // 2)

    # GT symbol
    draw.rectangle((x0, legend_y + 3, x0 + 18, legend_y + 17), outline=(220, 190, 0), width=3)
    draw.text((x0 + 26, legend_y), "GT", fill=(0, 0, 0), font=legend_font)

    # Prediction symbol
    x1 = x0 + 92
    draw.rectangle((x1, legend_y + 3, x1 + 18, legend_y + 17), outline=(0, 210, 210), width=3)
    draw.text((x1 + 26, legend_y), "Prediction", fill=(0, 0, 0), font=legend_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_path.with_suffix(".png")
    tif_path = out_path.with_suffix(".tif")
    pdf_path = out_path.with_suffix(".pdf")

    canvas.save(png_path, dpi=(600, 600))
    canvas.save(tif_path, dpi=(600, 600), compression="tiff_lzw")
    canvas.save(pdf_path, "PDF", resolution=600.0)

    log(f"Saved Figure 5 PNG: {png_path}")
    log(f"Saved Figure 5 TIFF: {tif_path}")
    log(f"Saved Figure 5 PDF: {pdf_path}")


def make_row_figure(
    case_info: Dict,
    image_path: Path,
    gt_boxes: List[GTBox],
    preds_by_exp: Dict[str, List[PredBox]],
    metrics: Dict,
    draw_gt: bool,
    draw_conf: float,
    max_draw: int,
    panel_target_w: int,
) -> Image.Image:
    """
    Manuscript row layout:
    - 3 panels per row
    - no black panel bars
    - concise white labels inside each panel
    - row header only says Case A/B/C and instance count
    """
    image_id = str(image_path.resolve())
    img = Image.open(image_path).convert("RGB")
    crop = choose_crop(case_info["case_type"], img, gt_boxes, metrics)

    preds_by_exp_img = {
        exp["name"]: [p for p in preds_by_exp[exp["name"]] if p.image_id == image_id]
        for exp in EXPERIMENTS
    }

    panels = []
    for exp in EXPERIMENTS:
        short = exp["short"]
        m = metrics[exp["name"]]

        # Use empty title to avoid old black title text, then strip any remaining bar.
        panel = draw_panel(
            img=img,
            title="",
            gt_boxes=gt_boxes,
            pred_boxes=preds_by_exp_img[exp["name"]],
            crop=crop,
            draw_gt=draw_gt,
            draw_conf=draw_conf,
            max_draw=max_draw,
            target_w=panel_target_w,
        )

        panel = _animals_strip_panel_title_bar(panel)
        panel = _animals_strip_panel_footer_note(panel)

        res_label = str(short)
        if not res_label.endswith("px"):
            res_label = f"{res_label} px"

        label = f"{res_label}\nTP={int(m['tp'])}, FN={int(m['fn'])}, FP={int(m['fp'])}"
        panel = _animals_add_panel_label(panel, label)
        panels.append(panel)

    panel_gap = 8
    row_header_h = 34
    row_w = sum(p.width for p in panels) + panel_gap * (len(panels) - 1)
    row_h = max(p.height for p in panels) + row_header_h

    row_canvas = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    draw = ImageDraw.Draw(row_canvas)

    case_font = safe_font(22)
    inst_font = safe_font(18)

    # Compact row header.
    case_title = case_info["row_name"]
    inst_text = f"{len(gt_boxes)} GT instances"

    draw.text((6, 4), case_title, fill=(0, 0, 0), font=case_font)
    draw.text((110, 6), inst_text, fill=(70, 70, 70), font=inst_font)

    x = 0
    for p in panels:
        row_canvas.paste(p, (x, row_header_h))
        x += p.width + panel_gap

    return row_canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=Path(".").resolve(),
                        help="Local project root containing the PIO dataset and prediction cache.")
    parser.add_argument("--data_yaml", type=Path, default=None,
                        help="Path to dataset.yaml. Defaults to <project_root>/00_DATASET/PIO-GRDB-MD5-7_1_2/dataset.yaml.")
    parser.add_argument("--pred_cache_dir", type=Path, default=None,
                        help="Directory containing FINAL_CANONICAL_IMAGE_ID_V4 prediction cache CSV files.")
    parser.add_argument("--group_definitions", type=Path, default=None,
                        help="Path to P0_P1_group_definitions_test.json.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory. Defaults to <project_root>/figures.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--metric_conf", type=float, default=0.25, help="Confidence threshold used to compute TP/FN/FP shown in titles.")
    parser.add_argument("--draw_conf", type=float, default=0.50, help="Confidence threshold for drawing prediction boxes.")
    parser.add_argument("--draw_gt", type=int, default=1)
    parser.add_argument("--max_draw", type=int, default=120)
    parser.add_argument("--panel_target_w", type=int, default=640)
    args = parser.parse_args()

    global PROJECT_ROOT, DATA_YAML, PRED_CACHE_DIR, OUT_DIR, GROUP_DEFS_PATH
    PROJECT_ROOT = args.project_root.resolve()
    DATA_YAML = args.data_yaml if args.data_yaml is not None else PROJECT_ROOT / "00_DATASET" / "PIO-GRDB-MD5-7_1_2" / "dataset.yaml"
    PRED_CACHE_DIR = args.pred_cache_dir if args.pred_cache_dir is not None else PROJECT_ROOT / "04_AUDIT" / "04_groupwise_evaluation" / "prediction_cache"
    OUT_DIR = args.out_dir if args.out_dir is not None else PROJECT_ROOT / "figures"
    GROUP_DEFS_PATH = args.group_definitions if args.group_definitions is not None else PROJECT_ROOT / "group_definitions" / "P0_P1_group_definitions_test.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "Figure5_qualitative_resolution_comparison_animals.png"
    meta_path = OUT_DIR / "Figure5_qualitative_resolution_comparison_animals_metadata.json"

    log(f"Output directory: {OUT_DIR}")
    log(f"Split: {args.split}")
    log(f"metric_conf={args.metric_conf}, draw_conf={args.draw_conf}, max_draw={args.max_draw}")
    log(f"Fixed three-image mode enabled.")

    class_id = None if int(args.class_id) < 0 else int(args.class_id)

    image_paths = load_image_list(DATA_YAML, args.split)
    log(f"Loaded images: {len(image_paths)}")

    gt_list, density_df = collect_gt(image_paths, class_id=class_id)
    gt_by_img: Dict[str, List[GTBox]] = {}
    for g in gt_list:
        gt_by_img.setdefault(g.image_id, []).append(g)

    preds_by_exp = load_predictions(split=args.split, class_id=class_id)

    # Build map for density metadata
    density_map = density_df.set_index("image_id").to_dict(orient="index")

    rows = []
    meta_rows = []

    for case in FIXED_CASES:
        img_path = find_image_path(image_paths, case["filename"])
        image_id = str(img_path.resolve())
        gts = gt_by_img.get(image_id, [])
        if not gts:
            raise RuntimeError(f"No GT boxes found for fixed image: {case['filename']}")

        metrics = build_metrics_for_image(image_id, gts, preds_by_exp, metric_conf=float(args.metric_conf))
        row_im = make_row_figure(
            case_info=case,
            image_path=img_path,
            gt_boxes=gts,
            preds_by_exp=preds_by_exp,
            metrics=metrics,
            draw_gt=bool(args.draw_gt),
            draw_conf=float(args.draw_conf),
            max_draw=int(args.max_draw),
            panel_target_w=int(args.panel_target_w),
        )
        rows.append(row_im)

        den = density_map.get(image_id, {})
        meta_rows.append({
            "row_name": case["row_name"],
            "filename": case["filename"],
            "image_id": image_id,
            "display_title": case["display_title"],
            "density_group": den.get("density_group", ""),
            "num_instances": int(den.get("num_instances", len(gts))),
            "800": {"TP": int(metrics["M1_800_seed42"]["tp"]), "FN": int(metrics["M1_800_seed42"]["fn"]), "FP": int(metrics["M1_800_seed42"]["fp"])},
            "960": {"TP": int(metrics["M2_960_seed42"]["tp"]), "FN": int(metrics["M2_960_seed42"]["fn"]), "FP": int(metrics["M2_960_seed42"]["fp"])},
            "1280": {"TP": int(metrics["M3_1280_seed42"]["tp"]), "FN": int(metrics["M3_1280_seed42"]["fn"]), "FP": int(metrics["M3_1280_seed42"]["fp"])},
        })

    compose_final_figure(rows, out_path)

    meta = {
        "script": "make_figure5_qualitative_resolution_comparison.py",
        "split": args.split,
        "metric_conf": args.metric_conf,
        "draw_conf": args.draw_conf,
        "draw_gt": args.draw_gt,
        "max_draw": args.max_draw,
        "panel_target_w": args.panel_target_w,
        "output_figure": str(out_path),
        "rows": meta_rows,
        "notes": [
            "This figure is a fixed three-image paper figure.",
            "GT boxes are yellow; predicted boxes are cyan.",
            "Rows 1 and 3 emphasize the practical benefit of 960 over 800.",
            "Row 2 emphasizes the limited marginal gain of 1280 over 960.",
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved metadata: {meta_path}")
    log("Done.")


if __name__ == "__main__":
    main()
