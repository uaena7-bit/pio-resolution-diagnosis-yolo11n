#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a clean PIO-GRDB-MD5 7:1:2 dataset from the original PIO train/val folders.

This script is tailored for your current layout:

D:/Broiler chicken detection dataset/data/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  dataset.yaml
  classes.txt

Goal
----
Pool the original train + val images together, ignore the old split, then create a new
leakage-audited YOLO dataset:

D:/Broiler chicken detection dataset/00_DATASET/PIO-GRDB-MD5-7_1_2/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  dataset.yaml
  reports/

PIO-GRDB-MD5 means:
1. MD5 content deduplication: same image content must stay in the same split.
2. Group constraint: same inferred source sequence/group must stay in the same split.
3. Ratio-density balance: greedily balance image count, instance count, and object density.
4. Audit: check path/name/MD5/group leakage after splitting.

Safety
------
The script NEVER deletes, moves, or modifies the original data folder.
It only reads from data/images/train, data/images/val, data/labels/train, data/labels/val.

Recommended command, dry-run first:

D:\\ANACONDA\\envs\\yolo\\python.exe "D:\\Broiler chicken detection dataset\\01_SCRIPTS\\00_dataset_audit\\create_pio_grdb_md5_split.py" ^
  --src-root "D:\\Broiler chicken detection dataset\\data" ^
  --out-root "D:\\Broiler chicken detection dataset\\00_DATASET\\PIO-GRDB-MD5-7_1_2" ^
  --ratios 0.70 0.10 0.20 ^
  --seed 42 ^
  --mode copy ^
  --dry-run

Then remove --dry-run to really create the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    idx: int
    old_split: str
    image_path: Path
    label_path: Path
    rel_from_old_split: Path
    output_stem: str
    output_image_name: str
    output_label_name: str
    basename_lower: str
    md5: str
    source_group: str
    instances: int


@dataclass
class Component:
    cid: int
    sample_indices: List[int]
    images: int
    instances: int
    md5_keys: List[str]
    source_groups: List[str]

    @property
    def density(self) -> float:
        return self.instances / self.images if self.images else 0.0


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel_posix(path: Path) -> str:
    return path.as_posix()


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def count_yolo_instances(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    n = 0
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                n += 1
    return n


def is_yolo_label_file(label_path: Path) -> Tuple[bool, str]:
    """Light validation: each non-empty line should have at least 5 numeric fields."""
    if not label_path.exists():
        return True, "missing label treated as empty"
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                return False, f"line {line_no}: fewer than 5 fields"
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                return False, f"line {line_no}: non-numeric YOLO fields"
            if cls < 0:
                return False, f"line {line_no}: negative class id"
            # Allow a tiny tolerance because some exported labels may contain 1.0000001.
            if any(v < -1e-6 or v > 1.000001 for v in vals):
                return False, f"line {line_no}: bbox value outside [0, 1]"
    return True, "ok"


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.as_posix().lower(),
    )


def infer_source_group(stem: str, group_regex: Optional[str]) -> str:
    """
    Infer near-source group from filename stem.

    If you know the filename rule, use --group-regex.
    Example: house01_cam02_000123 -> --group-regex "^(?P<group>.+?)_\\d+$"

    Default rule strips common trailing frame/index tokens.
    """
    raw = stem
    s = stem.lower()

    if group_regex:
        m = re.search(group_regex, raw)
        if not m:
            return s
        if "group" in m.groupdict():
            return str(m.group("group")).lower()
        if m.groups():
            return str(m.group(1)).lower()
        return m.group(0).lower()

    patterns = [
        r"^(?P<group>.+?)[_\-\s]+(?:frame|frm|img|image)[_\-\s]*\d{2,8}$",
        r"^(?P<group>.+?)[_\-\s]+\d{2,8}$",
        r"^(?P<group>.+?)(?:frame|frm|img|image)\d{2,8}$",
    ]
    for pat in patterns:
        m = re.match(pat, s)
        if m and m.group("group"):
            return m.group("group")
    return s


def make_safe_output_stem(old_split: str, rel_image: Path) -> str:
    """
    Make a unique flat output stem.
    We prefix the old split to avoid filename collision when old train/val have same basename.
    """
    parts = [old_split] + list(rel_image.with_suffix("").parts)
    joined = "__".join(parts)
    joined = re.sub(r"[^A-Za-z0-9_\-.]+", "_", joined)
    return joined


def load_names_from_yaml(yaml_path: Optional[Path]) -> Optional[List[str]]:
    if yaml_path is None or not yaml_path.exists() or yaml is None:
        return None
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        names = data.get("names")
        if isinstance(names, list):
            return [str(x) for x in names]
        if isinstance(names, dict):
            def sort_key(k):
                return int(k) if str(k).isdigit() else str(k)
            return [str(names[k]) for k in sorted(names.keys(), key=sort_key)]
    except Exception:
        return None
    return None


def load_names_from_classes_txt(classes_txt: Path) -> Optional[List[str]]:
    if not classes_txt.exists():
        return None
    names = [x.strip() for x in classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    return names or None


def load_class_names(src_root: Path, explicit_yaml: Optional[str], fallback: List[str]) -> List[str]:
    yaml_path = Path(explicit_yaml) if explicit_yaml else src_root / "dataset.yaml"
    names = load_names_from_yaml(yaml_path)
    if names:
        return names
    names = load_names_from_classes_txt(src_root / "classes.txt")
    if names:
        return names
    return fallback


def collect_samples(
    src_root: Path,
    old_image_splits: Sequence[str],
    old_label_splits: Sequence[str],
    group_regex: Optional[str],
    validate_labels: bool,
) -> Tuple[List[Sample], List[str]]:
    warnings: List[str] = []
    samples: List[Sample] = []
    idx = 0

    if len(old_image_splits) != len(old_label_splits):
        fail("--old-image-splits and --old-label-splits must have the same length.")

    for old_split, img_subdir, lbl_subdir in zip(old_image_splits, old_image_splits, old_label_splits):
        images_dir = src_root / "images" / img_subdir
        labels_dir = src_root / "labels" / lbl_subdir

        if not images_dir.exists():
            fail(f"Missing source image directory: {images_dir}")
        if not labels_dir.exists():
            fail(f"Missing source label directory: {labels_dir}")

        images = find_images(images_dir)
        if not images:
            warnings.append(f"No images found in: {images_dir}")

        for image_path in images:
            rel_image = image_path.relative_to(images_dir)
            label_path = labels_dir / rel_image.with_suffix(".txt")

            if not label_path.exists():
                warnings.append(f"Missing label, treated as empty: old_split={old_split}, image={rel_posix(rel_image)}")
            elif validate_labels:
                ok, reason = is_yolo_label_file(label_path)
                if not ok:
                    warnings.append(f"Suspicious YOLO label: {label_path} -> {reason}")

            md5 = compute_md5(image_path)
            source_group = infer_source_group(image_path.stem, group_regex)
            instances = count_yolo_instances(label_path)
            output_stem = make_safe_output_stem(old_split, rel_image)

            samples.append(
                Sample(
                    idx=idx,
                    old_split=old_split,
                    image_path=image_path,
                    label_path=label_path,
                    rel_from_old_split=rel_image,
                    output_stem=output_stem,
                    output_image_name=output_stem + image_path.suffix.lower(),
                    output_label_name=output_stem + ".txt",
                    basename_lower=image_path.name.lower(),
                    md5=md5,
                    source_group=source_group,
                    instances=instances,
                )
            )
            idx += 1

    if not samples:
        fail("No source images found. Check --src-root and old split directories.")
    return samples, warnings


def build_components(samples: Sequence[Sample]) -> List[Component]:
    """
    Build unsplittable components.

    A component is the atomic unit of splitting. Images in the same component
    are forced into the same new split.

    Constraints used here:
    1. Same image MD5.
    2. Same inferred source group.
    3. Same original basename.

    The basename constraint is intentionally included because your audit protocol
    checks filename leakage as well as content leakage.
    """
    uf = UnionFind(len(samples))
    by_md5: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    by_basename: Dict[str, int] = {}

    for s in samples:
        if s.md5 in by_md5:
            uf.union(s.idx, by_md5[s.md5])
        else:
            by_md5[s.md5] = s.idx

        if s.source_group in by_group:
            uf.union(s.idx, by_group[s.source_group])
        else:
            by_group[s.source_group] = s.idx

        if s.basename_lower in by_basename:
            uf.union(s.idx, by_basename[s.basename_lower])
        else:
            by_basename[s.basename_lower] = s.idx

    buckets: Dict[int, List[int]] = defaultdict(list)
    for s in samples:
        buckets[uf.find(s.idx)].append(s.idx)

    components: List[Component] = []
    for cid, (_, idxs) in enumerate(sorted(buckets.items(), key=lambda kv: min(kv[1]))):
        components.append(
            Component(
                cid=cid,
                sample_indices=sorted(idxs),
                images=len(idxs),
                instances=sum(samples[i].instances for i in idxs),
                md5_keys=sorted({samples[i].md5 for i in idxs}),
                source_groups=sorted({samples[i].source_group for i in idxs}),
            )
        )
    return components


def assign_components(
    components: Sequence[Component],
    ratios: Sequence[float],
    seed: int,
    w_images: float = 1.0,
    w_instances: float = 1.0,
    w_density: float = 0.35,
) -> Dict[int, str]:
    """
    Assign components to train/val/test.

    The earlier pure greedy version can collapse a smaller split to zero when
    components are highly uneven. This version is safer:

    1. Select validation components to approximate the val target.
    2. Select test components from the remaining pool to approximate the test target.
    3. Put the rest into train.
    4. Run a small local search to improve the global image/instance/density balance.

    Components remain indivisible, so leakage constraints are preserved.
    """
    total_images = sum(c.images for c in components)
    total_instances = sum(c.instances for c in components)
    global_density = total_instances / total_images if total_images else 0.0

    targets = {
        split: {
            "images": total_images * r,
            "instances": total_instances * r,
        }
        for split, r in zip(SPLITS, ratios)
    }

    comp_by_id = {c.cid: c for c in components}
    rng = random.Random(seed)

    def subset_score(cids: set[int], split: str) -> float:
        imgs = sum(comp_by_id[cid].images for cid in cids)
        inst = sum(comp_by_id[cid].instances for cid in cids)
        target_img = max(targets[split]["images"], 1.0)
        target_inst = max(targets[split]["instances"], 1.0)
        img_err = ((imgs - target_img) / target_img) ** 2
        inst_err = ((inst - target_inst) / target_inst) ** 2
        dens = inst / imgs if imgs else 0.0
        dens_err = ((dens - global_density) / global_density) ** 2 if global_density > 0 and imgs > 0 else 1.0
        empty_penalty = 100.0 if imgs == 0 else 0.0
        return w_images * img_err + w_instances * inst_err + w_density * dens_err + empty_penalty

    def global_score(assign: Dict[int, str]) -> float:
        score = 0.0
        for split in SPLITS:
            cids = {cid for cid, sp in assign.items() if sp == split}
            score += subset_score(cids, split)
        return score

    def greedy_pick(target_split: str, available: set[int]) -> set[int]:
        """Greedily pick a non-empty subset for one split."""
        chosen: set[int] = set()
        current = subset_score(chosen, target_split)

        # First seed: choose the single component that best reduces target error.
        best_seed = None
        best_seed_score = current
        for cid in available:
            sc = subset_score({cid}, target_split)
            if sc < best_seed_score:
                best_seed_score = sc
                best_seed = cid
        if best_seed is not None:
            chosen.add(best_seed)
            available.remove(best_seed)
            current = best_seed_score

        # Then add components while they improve the target score.
        improved = True
        while improved and available:
            improved = False
            best_cid = None
            best_score = current
            # Shuffle equal-quality candidates deterministically.
            cand = list(available)
            rng.shuffle(cand)
            for cid in cand:
                sc = subset_score(chosen | {cid}, target_split)
                if sc < best_score:
                    best_score = sc
                    best_cid = cid
            if best_cid is not None:
                chosen.add(best_cid)
                available.remove(best_cid)
                current = best_score
                improved = True

        return chosen

    # Build initial solution: reserve small splits first, then train gets the rest.
    available = {c.cid for c in components}
    val_cids = greedy_pick("val", available)
    test_cids = greedy_pick("test", available)

    assignment: Dict[int, str] = {}
    for cid in val_cids:
        assignment[cid] = "val"
    for cid in test_cids:
        assignment[cid] = "test"
    for cid in available:
        assignment[cid] = "train"

    # Local search: move one component at a time if global objective improves.
    best_score = global_score(assignment)
    changed = True
    rounds = 0
    max_rounds = 25
    while changed and rounds < max_rounds:
        changed = False
        rounds += 1
        cids = list(assignment.keys())
        rng.shuffle(cids)
        # Try larger components first because they dominate ratio error.
        cids.sort(key=lambda cid: (comp_by_id[cid].images, comp_by_id[cid].instances), reverse=True)
        for cid in cids:
            old_split = assignment[cid]
            for new_split in SPLITS:
                if new_split == old_split:
                    continue
                trial = dict(assignment)
                trial[cid] = new_split
                # Do not allow any split to become empty.
                if any(not any(sp == split for sp in trial.values()) for split in SPLITS):
                    continue
                sc = global_score(trial)
                if sc + 1e-12 < best_score:
                    assignment = trial
                    best_score = sc
                    changed = True
                    old_split = new_split

    return assignment


def component_lookup(components: Sequence[Component]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in components:
        for idx in c.sample_indices:
            out[idx] = c.cid
    return out


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_empty_label(dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        dst.write_text("", encoding="utf-8")


def create_dataset(
    samples: Sequence[Sample],
    components: Sequence[Component],
    comp_assignment: Dict[int, str],
    out_root: Path,
    mode: str,
    dry_run: bool,
) -> Dict[int, str]:
    sample_to_comp = component_lookup(components)
    sample_split: Dict[int, str] = {}

    if not dry_run:
        for split in SPLITS:
            ensure_dir(out_root / "images" / split)
            ensure_dir(out_root / "labels" / split)

    used_output_names: Dict[str, int] = {}

    for s in samples:
        split = comp_assignment[sample_to_comp[s.idx]]
        sample_split[s.idx] = split

        # Safety check for output filename collision.
        key = f"{split}/{s.output_image_name.lower()}"
        if key in used_output_names:
            fail(f"Output filename collision: {key}. Please check source filenames.")
        used_output_names[key] = s.idx

        if dry_run:
            continue

        dst_img = out_root / "images" / split / s.output_image_name
        dst_lbl = out_root / "labels" / split / s.output_label_name
        materialize_file(s.image_path, dst_img, mode)
        if s.label_path.exists():
            materialize_file(s.label_path, dst_lbl, mode)
        else:
            write_empty_label(dst_lbl)

    return sample_split


def write_dataset_yaml(out_root: Path, names: List[str], dry_run: bool) -> None:
    if dry_run:
        return
    content = {
        "path": out_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    if yaml is not None:
        with (out_root / "dataset.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    else:
        lines = [
            f"path: {content['path']}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            "names:",
        ]
        for i, name in enumerate(names):
            lines.append(f"  {i}: {name}")
        (out_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_stats(samples: Sequence[Sample], sample_split: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    total_images = len(samples)
    total_instances = sum(s.instances for s in samples)
    stats: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        idxs = [s.idx for s in samples if sample_split[s.idx] == split]
        images = len(idxs)
        instances = sum(samples[i].instances for i in idxs)
        stats[split] = {
            "images": images,
            "image_ratio": images / total_images if total_images else 0.0,
            "instances": instances,
            "instance_ratio": instances / total_instances if total_instances else 0.0,
            "avg_instances_per_image": instances / images if images else 0.0,
        }
    stats["total"] = {
        "images": total_images,
        "image_ratio": 1.0,
        "instances": total_instances,
        "instance_ratio": 1.0,
        "avg_instances_per_image": total_instances / total_images if total_images else 0.0,
    }
    return stats


def keys_crossing_splits(samples: Sequence[Sample], sample_split: Dict[int, str], key_func) -> Dict[str, List[str]]:
    buckets: Dict[str, set[str]] = defaultdict(set)
    for s in samples:
        buckets[str(key_func(s))].add(sample_split[s.idx])
    return {k: sorted(v) for k, v in buckets.items() if len(v) > 1}


def audit_leakage(samples: Sequence[Sample], sample_split: Dict[int, str]) -> Tuple[bool, List[str]]:
    ok = True
    lines: List[str] = []

    checks = [
        ("absolute source image path", lambda s: s.image_path.resolve().as_posix().lower()),
        ("original basename", lambda s: s.basename_lower),
        ("image content MD5", lambda s: s.md5),
        ("inferred source group", lambda s: s.source_group),
    ]

    for name, key_func in checks:
        leaks = keys_crossing_splits(samples, sample_split, key_func)
        if leaks:
            ok = False
            lines.append(f"[FAIL] Cross-split leakage by {name}: {len(leaks)} keys")
            for k, splits in list(leaks.items())[:80]:
                lines.append(f"  - {k} -> {splits}")
            if len(leaks) > 80:
                lines.append(f"  ... {len(leaks) - 80} more")
        else:
            lines.append(f"[PASS] No cross-split leakage by {name}")

    return ok, lines


def write_reports(
    samples: Sequence[Sample],
    components: Sequence[Component],
    comp_assignment: Dict[int, str],
    sample_split: Dict[int, str],
    out_root: Path,
    ratios: Sequence[float],
    warnings: Sequence[str],
    dry_run: bool,
) -> bool:
    stats = split_stats(samples, sample_split)
    ok, audit_lines = audit_leakage(samples, sample_split)
    sample_to_comp = component_lookup(components)

    print("\n=== New PIO-GRDB-MD5 split statistics ===")
    print(f"Target ratios: train={ratios[0]:.2f}, val={ratios[1]:.2f}, test={ratios[2]:.2f}")
    for split in (*SPLITS, "total"):
        x = stats[split]
        print(
            f"{split:>5s}: images={int(x['images']):5d} "
            f"({x['image_ratio']:.3f}), instances={int(x['instances']):7d} "
            f"({x['instance_ratio']:.3f}), density={x['avg_instances_per_image']:.2f}"
        )

    print("\n=== Leakage audit ===")
    for line in audit_lines:
        print(line)

    if warnings:
        print("\n=== Warnings ===")
        for w in warnings[:120]:
            print(f"[WARN] {w}")
        if len(warnings) > 120:
            print(f"[WARN] ... {len(warnings) - 120} more warnings")

    if dry_run:
        return ok

    reports_dir = out_root / "reports"
    ensure_dir(reports_dir)

    with (reports_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with (reports_dir / "split_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "images", "image_ratio", "instances", "instance_ratio", "avg_instances_per_image"])
        for split in (*SPLITS, "total"):
            x = stats[split]
            writer.writerow([
                split,
                int(x["images"]),
                f"{x['image_ratio']:.8f}",
                int(x["instances"]),
                f"{x['instance_ratio']:.8f}",
                f"{x['avg_instances_per_image']:.8f}",
            ])

    with (reports_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "new_split", "component_id", "old_split", "source_image_path", "source_label_path",
            "old_rel_image", "new_image_name", "new_label_name", "md5", "source_group", "instances"
        ])
        for s in samples:
            writer.writerow([
                sample_split[s.idx],
                sample_to_comp[s.idx],
                s.old_split,
                str(s.image_path),
                str(s.label_path),
                rel_posix(s.rel_from_old_split),
                s.output_image_name,
                s.output_label_name,
                s.md5,
                s.source_group,
                s.instances,
            ])

    with (reports_dir / "component_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component_id", "new_split", "images", "instances", "density", "num_md5", "num_source_groups", "source_groups_preview"])
        for c in components:
            writer.writerow([
                c.cid,
                comp_assignment[c.cid],
                c.images,
                c.instances,
                f"{c.density:.8f}",
                len(c.md5_keys),
                len(c.source_groups),
                " | ".join(c.source_groups[:10]),
            ])

    with (reports_dir / "leakage_audit.txt").open("w", encoding="utf-8") as f:
        f.write("PIO-GRDB-MD5 7:1:2 leakage audit\n")
        f.write("=" * 40 + "\n\n")
        f.write("\n".join(audit_lines) + "\n")
        if warnings:
            f.write("\nWarnings\n")
            f.write("-" * 16 + "\n")
            for w in warnings:
                f.write(f"[WARN] {w}\n")

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clean PIO-GRDB-MD5 7:1:2 YOLO dataset from original train/val folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-root", required=True, type=str, help="Source data root containing images/train, images/val, labels/train, labels/val.")
    parser.add_argument("--out-root", required=True, type=str, help="Output root for new clean dataset.")
    parser.add_argument("--old-image-splits", nargs="+", default=["train", "val"], help="Original image split subfolders under images/ to pool.")
    parser.add_argument("--old-label-splits", nargs="+", default=["train", "val"], help="Original label split subfolders under labels/ to pool.")
    parser.add_argument("--data-yaml", default=None, type=str, help="Optional source dataset.yaml for class names.")
    parser.add_argument("--class-names", nargs="+", default=["broiler"], help="Fallback class names if dataset.yaml/classes.txt cannot be read.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.10, 0.20], metavar=("TRAIN", "VAL", "TEST"), help="New split ratios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic tie-breaking.")
    parser.add_argument("--group-regex", default=None, type=str, help="Regex for source-group inference. Use named group (?P<group>...) or first capture group.")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy", help="How to materialize output files. copy is safest.")
    parser.add_argument("--dry-run", action="store_true", help="Only compute split and audit; do not write dataset files.")
    parser.add_argument("--allow-existing-out", action="store_true", help="Allow non-empty output directory. Use carefully.")
    parser.add_argument("--no-label-validate", action="store_true", help="Skip light YOLO label validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not src_root.exists():
        fail(f"Source root does not exist: {src_root}")
    if out_root == src_root or src_root in out_root.parents:
        # This is allowed for your project only if out_root is outside the data folder.
        # Example: project/00_DATASET/... is fine, project/data/... is not.
        if out_root == src_root or out_root.parent == src_root or str(out_root).lower().startswith(str(src_root).lower() + os.sep):
            fail("Output root must not be inside the source data root. Put it under 00_DATASET, not under data.")

    if out_root.exists() and any(out_root.iterdir()) and not args.allow_existing_out and not args.dry_run:
        fail(
            f"Output directory already exists and is not empty: {out_root}\n"
            "Use a new --out-root, delete the incomplete output manually, or pass --allow-existing-out."
        )

    ratios = [float(x) for x in args.ratios]
    if any(r <= 0 for r in ratios):
        fail("All ratios must be positive.")
    ratio_sum = sum(ratios)
    ratios = [r / ratio_sum for r in ratios]

    print("=== Create clean PIO-GRDB-MD5 dataset from original train/val ===")
    print(f"Source root     : {src_root}")
    print(f"Input images    : {[str(src_root / 'images' / s) for s in args.old_image_splits]}")
    print(f"Input labels    : {[str(src_root / 'labels' / s) for s in args.old_label_splits]}")
    print(f"Output root     : {out_root}")
    print(f"New ratios      : train={ratios[0]:.4f}, val={ratios[1]:.4f}, test={ratios[2]:.4f}")
    print(f"Seed            : {args.seed}")
    print(f"Mode            : {args.mode}")
    print(f"Dry run         : {args.dry_run}")

    print("\n[1/5] Pooling original train + val, reading labels, computing MD5...")
    samples, warnings = collect_samples(
        src_root=src_root,
        old_image_splits=args.old_image_splits,
        old_label_splits=args.old_label_splits,
        group_regex=args.group_regex,
        validate_labels=not args.no_label_validate,
    )
    print(f"Pooled images       : {len(samples)}")
    print(f"Total instances     : {sum(s.instances for s in samples)}")
    print(f"Unique image MD5    : {len(set(s.md5 for s in samples))}")
    print(f"Unique source groups: {len(set(s.source_group for s in samples))}")

    print("\n[2/5] Building unsplittable MD5/source-group components...")
    components = build_components(samples)
    largest = max((c.images for c in components), default=0)
    print(f"Components          : {len(components)}")
    print(f"Largest component   : {largest} images")
    if largest > max(1, int(len(samples) * 0.20)):
        warnings.append(
            f"Largest component has {largest} images, over 20% of all images. "
            "Your --group-regex may be too broad, or source groups are very large."
        )

    print("\n[3/5] Assigning components to new train/val/test with 7:1:2 target...")
    comp_assignment = assign_components(components, ratios, args.seed)

    print("\n[4/5] Creating output YOLO dataset...")
    sample_split = create_dataset(samples, components, comp_assignment, out_root, args.mode, args.dry_run)

    class_names = load_class_names(src_root, args.data_yaml, args.class_names)
    write_dataset_yaml(out_root, class_names, args.dry_run)
    print(f"Class names      : {class_names}")

    print("\n[5/5] Writing reports and running leakage audit...")
    ok = write_reports(samples, components, comp_assignment, sample_split, out_root, ratios, warnings, args.dry_run)

    if ok:
        if args.dry_run:
            print("\n[DONE] Dry-run completed. No output dataset was written.")
            print("If the four leakage checks are PASS, remove --dry-run and run again.")
        else:
            print("\n[DONE] Clean PIO-GRDB-MD5 7:1:2 dataset created successfully.")
            print(f"Dataset YAML : {out_root / 'dataset.yaml'}")
            print(f"Audit report : {out_root / 'reports' / 'leakage_audit.txt'}")
            print(f"Split stats  : {out_root / 'reports' / 'split_stats.csv'}")
        return 0

    print("\n[FAILED] Leakage audit failed. Do not train on this split before fixing the reported issue.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
