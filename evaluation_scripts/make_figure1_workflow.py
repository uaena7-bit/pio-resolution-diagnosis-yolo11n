#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 workflow generator for the CSSP submission.

This script creates a deterministic four-module workflow schematic:
PIO dataset -> leakage-controlled split -> YOLO11n training -> evaluation outputs.

The generated figure uses a real PIO image panel when a local image is provided
and contains no quantitative result panels. Quantitative figures are generated separately from CSV source data. It is intended for workflow visualization only.

Expected local input:
    figure_assets/PIO_dataset_example.jpg or .png, or --pio-image PATH

Outputs:
    Figure1_workflow_font_consistent.png
    Figure1_workflow_font_consistent.tif
    Figure1_workflow_font_consistent.svg
    Figure1_workflow_font_consistent.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
from matplotlib.lines import Line2D


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(r"D:\Broiler chicken detection dataset")
ASSET_DIR = PROJECT_ROOT / "05_PAPER" / "figure_assets"
DEFAULT_PIO_IMAGE = ASSET_DIR / "PIO_dataset_example.jpg"
DEFAULT_PIO_IMAGE_ALT = ASSET_DIR / "PIO_dataset_example.jpg.png"
DEFAULT_OUT_DIR = PROJECT_ROOT / "05_PAPER" / "figures"
DEFAULT_BASENAME = "Figure1_workflow_font_consistent"

# Crop removes the left timestamp-heavy region in the original PIO frame.
# If your image is already cropped, run with --no-crop.
DEFAULT_CROP_FRAC = (0.29, 0.01, 0.995, 0.985)

DPI = 600
MM = 1 / 25.4

TEAL = "#176f7a"
TEAL_DARK = "#0e4f56"
GREEN = "#5EA34B"
BLUE = "#4E86BF"
YELLOW = "#E0B21A"
EDGE = "#141414"
LIGHT_EDGE = "#b8b8b8"
LIGHT_FILL = "#fbfbfb"


# =============================================================================
# Utility
# =============================================================================

def resolve_pio_image(user_path: Path | None) -> Path:
    if user_path is not None:
        return user_path
    if DEFAULT_PIO_IMAGE.exists():
        return DEFAULT_PIO_IMAGE
    if DEFAULT_PIO_IMAGE_ALT.exists():
        return DEFAULT_PIO_IMAGE_ALT
    return DEFAULT_PIO_IMAGE


def crop_fraction(img: Image.Image, frac_box: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    l, t, r, b = frac_box
    return img.crop((int(l*w), int(t*h), int(r*w), int(b*h)))


def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    w, h = img.size
    aspect = w / h
    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / target_aspect)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


# =============================================================================
# Drawing helpers
# =============================================================================

def round_box(ax, x, y, w, h, radius=0.018, lw=0.95, fc="white", ec=EDGE, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def step_badge(ax, x, y, label):
    c = Circle((x, y), radius=0.014, facecolor=TEAL, edgecolor=TEAL_DARK, linewidth=0.45, zorder=9)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=6.9, color="white", fontweight="bold", zorder=10)


def flow_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13.5,
        linewidth=1.25,
        color=TEAL,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    ))


def cylinder(ax, cx, cy, r, hh, color):
    ax.add_patch(Rectangle((cx-r, cy-hh/2), 2*r, hh, facecolor=color, edgecolor="0.18", linewidth=0.55, zorder=4))
    ax.add_patch(Ellipse((cx, cy+hh/2), width=2*r, height=0.50*r, facecolor=color, edgecolor="0.18", linewidth=0.55, zorder=5))
    ax.add_patch(Ellipse((cx, cy), width=2*r, height=0.50*r, facecolor="none", edgecolor="0.18", linewidth=0.42, zorder=5))
    ax.add_patch(Ellipse((cx, cy-hh/2), width=2*r, height=0.50*r, facecolor="none", edgecolor="0.18", linewidth=0.42, zorder=5))


def add_bullet_section(ax, x, y, w, h, title, bullets):
    round_box(ax, x, y, w, h, radius=0.006, lw=0.55, fc="white", ec=LIGHT_EDGE, z=4)
    ax.text(x + w*0.055, y + h*0.70, title, ha="left", va="center",
            fontsize=6.7, fontweight="bold", color="0.02", zorder=6)
    text = "\n".join([f"鈥?{b}" for b in bullets])
    ax.text(x + w*0.065, y + h*0.34, text, ha="left", va="center",
            fontsize=5.55, linespacing=1.15, color="0.02", zorder=6)


# =============================================================================
# Panels
# =============================================================================

def panel_pio(ax, rect, img):
    x, y, w, h = rect
    round_box(ax, x, y, w, h)

    img_x = x + w*0.035
    img_y = y + h*0.275
    img_w = w*0.930
    img_h = h*0.500
    img2 = center_crop_to_aspect(img, img_w/img_h)

    # Keep the original data appearance. No artificial enhancement.
    ax.imshow(np.asarray(img2), extent=(img_x, img_x+img_w, img_y, img_y+img_h), aspect="auto", zorder=3)
    round_box(ax, img_x, img_y, img_w, img_h, radius=0.007, lw=0.35, fc="none", ec="0.25", z=5)

    ax.text(x+w/2, y+h*0.095, "PIO dataset", ha="center", va="center",
            fontsize=9.4, fontweight="bold", color="0.02", zorder=6)


def panel_split(ax, rect):
    x, y, w, h = rect
    round_box(ax, x, y, w, h)
    ax.text(x+w/2, y+h*0.905, "Leakage-controlled split", ha="center", va="center",
            fontsize=7.55, fontweight="bold", color="0.02", zorder=6)

    cx = [x+w*0.245, x+w*0.500, x+w*0.755]
    colors = [GREEN, BLUE, YELLOW]
    labels = ["Source A", "Source B", "Source C"]
    splits = ["Train", "Val", "Test"]

    for c, col, lab in zip(cx, colors, labels):
        ax.text(c, y+h*0.775, lab, ha="center", va="center", fontsize=5.5, color="0.02", zorder=6)
        cylinder(ax, c, y+h*0.675, w*0.039, h*0.070, col)
        ax.plot([c, c], [y+h*0.600, y+h*0.465], color="0.18", lw=0.48, linestyle=(0, (3.2, 2.1)), zorder=4)

    ax.plot([cx[0], cx[2]], [y+h*0.535, y+h*0.535], color="0.18", lw=0.48, linestyle=(0, (3.2, 2.1)), zorder=4)

    for c, col, lab in zip(cx, colors, splits):
        bw, bh = w*0.190, h*0.250
        bx, by = c-bw/2, y+h*0.270
        round_box(ax, bx, by, bw, bh, radius=0.006, lw=0.72, fc="white", ec=col, z=4)
        ax.text(c, by+bh*0.78, lab, ha="center", va="center", fontsize=6.3, fontweight="bold", color="0.02", zorder=6)

        # 3x3 solid dots in every split, including Test.
        for rr in range(3):
            for cc in range(3):
                ax.plot(
                    bx + bw*(0.25 + 0.25*cc),
                    by + bh*(0.28 + 0.19*rr),
                    marker="o",
                    markersize=2.85,
                    markerfacecolor=col,
                    markeredgecolor=col,
                    markeredgewidth=0,
                    zorder=6
                )

    ax.text(x+w/2, y+h*0.105,
            "MD5 deduplication\nsource-group binding\ndensity-balanced allocation",
            ha="center", va="center", fontsize=6.35, linespacing=1.13, color="0.02", zorder=6)


def panel_training(ax, rect):
    x, y, w, h = rect
    round_box(ax, x, y, w, h)
    ax.text(x+w/2, y+h*0.875, "YOLO11n\ntraining", ha="center", va="center",
            fontsize=7.8, fontweight="bold", linespacing=1.05, color="0.02", zorder=6)

    lx, mx, rx = x+w*0.23, x+w*0.50, x+w*0.77
    ys_l = [y+h*0.61, y+h*0.54, y+h*0.47]
    ys_m = [y+h*0.62, y+h*0.54, y+h*0.46]
    ys_r = [y+h*0.58, y+h*0.50]

    for yl in ys_l:
        for ym in ys_m:
            ax.add_line(Line2D([lx, mx], [yl, ym], color="0.12", lw=0.42, zorder=3))
    for ym in ys_m:
        for yr in ys_r:
            ax.add_line(Line2D([mx, rx], [ym, yr], color="0.12", lw=0.42, zorder=3))

    for yl in ys_l:
        ax.add_patch(Circle((lx, yl), radius=w*0.045, facecolor="white", edgecolor="0.12", linewidth=0.62, zorder=5))
    for ym in ys_m:
        ax.add_patch(Circle((mx, ym), radius=w*0.047, facecolor="#c6e3df", edgecolor="0.12", linewidth=0.62, zorder=5))
    for yr in ys_r:
        ax.add_patch(Circle((rx, yr), radius=w*0.045, facecolor="white", edgecolor="0.12", linewidth=0.62, zorder=5))

    ax.text(lx-w*0.038, y+h*0.410, "...", ha="center", va="center", fontsize=7.5, zorder=5)
    ax.text(rx+w*0.038, y+h*0.410, "...", ha="center", va="center", fontsize=7.5, zorder=5)

    # Monitor icon
    mon_x, mon_y = x+w*0.28, y+h*0.220
    mon_w, mon_h = w*0.44, h*0.145
    ax.add_patch(Rectangle((mon_x, mon_y), mon_w, mon_h, facecolor="white", edgecolor="0.12", linewidth=0.70, zorder=5))
    ax.add_patch(Rectangle((mon_x+mon_w*0.085, mon_y+mon_h*0.15), mon_w*0.83, mon_h*0.70,
                           facecolor=LIGHT_FILL, edgecolor="0.35", linewidth=0.38, zorder=5))
    cx, cy = mon_x+mon_w/2, mon_y+mon_h/2
    ax.add_patch(Circle((cx, cy), radius=w*0.038, facecolor=TEAL, edgecolor="none", zorder=6))
    ax.add_patch(Circle((cx, cy), radius=w*0.017, facecolor="white", edgecolor="none", zorder=7))
    ax.add_line(Line2D([mon_x+mon_w/2, mon_x+mon_w/2], [mon_y, mon_y-h*0.025], color="0.12", lw=0.60, zorder=5))
    ax.add_line(Line2D([mon_x+mon_w*0.33, mon_x+mon_w*0.67], [mon_y-h*0.025, mon_y-h*0.025], color="0.12", lw=0.60, zorder=5))

    ax.text(x+w/2, y+h*0.105, "800 / 960 / 1280 px", ha="center", va="center",
            fontsize=6.25, color="0.02", zorder=6)


def panel_outputs(ax, rect):
    x, y, w, h = rect
    round_box(ax, x, y, w, h)
    ax.text(x+w/2, y+h*0.93, "Evaluation outputs", ha="center", va="center",
            fontsize=7.7, fontweight="bold", color="0.02", zorder=6)

    row_x = x + w*0.075
    row_w = w*0.850
    row_h = h*0.145
    row_ys = [y+h*0.720, y+h*0.535, y+h*0.350, y+h*0.175]

    data = [
        ("Standard test", ["global AP / AP50", "max_det sensitivity"]),
        ("Scale-density diagnosis", ["ordinary AP", "strict diagnostic AP"]),
        ("Uncertainty analysis", ["bootstrap CIs"]),
        ("Deployment analysis", ["counting / latency / VRAM"]),
    ]
    for (title, bullets), cy in zip(data, row_ys):
        add_bullet_section(ax, row_x, cy-row_h/2, row_w, row_h, title, bullets)


def build_figure(pio_image_path: Path, out_dir: Path, basename: str, crop=True):
    if not pio_image_path.exists():
        raise FileNotFoundError(
            f"PIO image not found: {pio_image_path}\n"
            f"Tried default paths:\n  {DEFAULT_PIO_IMAGE}\n  {DEFAULT_PIO_IMAGE_ALT}\n"
            "Use --pio-image to pass the exact path."
        )

    pio = Image.open(pio_image_path).convert("RGB")
    if crop:
        pio = crop_fraction(pio, DEFAULT_CROP_FRAC)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 4:3 figure layout for manuscript workflow visualization.
    fig, ax = plt.subplots(figsize=(180*MM, 135*MM), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.225
    h = 0.565
    panels = [
        (0.018, y, 0.255, h),   # PIO
        (0.305, y, 0.250, h),   # split
        (0.585, y, 0.155, h),   # training
        (0.770, y, 0.210, h),   # outputs
    ]

    for i, (x, yy, w, hh) in enumerate(panels, start=1):
        step_badge(ax, x+0.010, yy+hh+0.030, str(i))

    panel_pio(ax, panels[0], pio)
    panel_split(ax, panels[1])
    panel_training(ax, panels[2])
    panel_outputs(ax, panels[3])

    for i in range(3):
        x1, y1, w1, h1 = panels[i]
        x2, y2, w2, h2 = panels[i+1]
        flow_arrow(ax, x1+w1+0.011, y1+h1*0.53, x2-0.011, y2+h2*0.53)

    for ext in ["png", "tif", "svg", "pdf"]:
        fig.savefig(out_dir / f"{basename}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.030)
    plt.close(fig)

    print("[DONE] Figure 1 generated:")
    for ext in ["png", "tif", "svg", "pdf"]:
        print(out_dir / f"{basename}.{ext}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pio-image", type=Path, default=None, help="Path to PIO example image.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--basename", type=str, default=DEFAULT_BASENAME, help="Output basename.")
    parser.add_argument("--no-crop", action="store_true", help="Disable default left-side crop.")
    return parser.parse_args()


def main():
    args = parse_args()
    pio_path = resolve_pio_image(args.pio_image)
    build_figure(pio_path, args.out_dir, args.basename, crop=not args.no_crop)


if __name__ == "__main__":
    main()
