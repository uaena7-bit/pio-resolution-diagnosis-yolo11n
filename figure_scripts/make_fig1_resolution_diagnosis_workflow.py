
# -*- coding: utf-8 -*-
"""
make_fig1_precision_agriculture_v23_final_refined.py

v25 changes:
1) Widen the Controlled-resolution panel to add left/right title margins.
2) Further enlarge Detector and Resolution note boxes.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D

DPI = 600
FIG_W_MM = 174.0
FIG_H_MM = 90.0
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4

COL = {
    "bg": "#f6f6f4",
    "panel_fill": "#ffffff",
    "panel_edge": "#1f1f1f",
    "ink": "#171717",
    "muted": "#4f565d",
    "muted2": "#5e666d",
    "accent": "#1b7f86",
    "green": "#5ca649",
    "blue": "#4f86cb",
    "yellow": "#d9af18",
    "soft": "#eef6f5",
    "soft_gray": "#f8f8f8",
    "box": "#b8b8b8",
    "light_edge": "#d1d1d1",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Broiler chicken detection dataset\05_PAPER\figures")
    return Path("/mnt/data/fig1_precision_agriculture_v23_preview")

def rounded_box(ax, x, y, w, h, fc=None, ec=None, lw=1.0, r=0.018, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc if fc is not None else COL["panel_fill"],
        edgecolor=ec if ec is not None else COL["panel_edge"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch

def text(ax, x, y, s, size=8, weight="normal", color=None,
         ha="left", va="center", ls=1.12, z=6):
    ax.text(
        x, y, s,
        fontsize=size,
        fontweight=weight,
        color=color or COL["ink"],
        ha=ha,
        va=va,
        linespacing=ls,
        zorder=z,
    )

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=19,
        linewidth=1.35,
        color=COL["accent"],
        shrinkA=0, shrinkB=0, zorder=12,
        capstyle="round", joinstyle="round",
    ))

def stat_row(ax, x, y, w, h, label, value):
    rounded_box(ax, x, y, w, h, fc="#fcfcfc", ec="#cccccc", lw=0.82, r=0.011, z=4)
    text(ax, x + w * 0.080, y + h * 0.52, label,
         size=6.35, weight="bold", color=COL["muted"], ha="left")
    text(ax, x + w * 0.930, y + h * 0.52, value,
         size=9.25, weight="bold", color=COL["ink"], ha="right")

def draw_panel1(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, lw=1.28, r=0.024)
    text(ax, x + w / 2, y + h * 0.922, "PIO broiler\nmonitoring dataset",
         size=8.15, weight="bold", ha="center", ls=1.0)

    rounded_box(ax, x + w * 0.105, y + h * 0.735, w * 0.790, h * 0.087,
                fc=COL["soft"], ec="#bedad7", lw=0.78, r=0.012, z=4)
    text(ax, x + w / 2, y + h * 0.779,
         "Dense broiler-house monitoring\nin practical farm conditions",
         size=5.25, weight="bold", color=COL["accent"], ha="center", ls=1.03)

    sx = x + w * 0.10
    sw = w * 0.80
    sh = h * 0.108
    stat_row(ax, sx, y + h * 0.570, sw, sh, "Images", "1,487")
    stat_row(ax, sx, y + h * 0.430, sw, sh, "Instances", "327,288")
    stat_row(ax, sx, y + h * 0.290, sw, sh, "Inst./image", "220.10")

    rounded_box(ax, x + w * 0.130, y + h * 0.096, w * 0.740, h * 0.088,
                fc="#fbfbfb", ec="#d0d0d0", lw=0.70, r=0.012, z=4)
    text(ax, x + w / 2, y + h * 0.140,
         "Dataset basis for\nresolution selection",
         size=5.55, weight="bold", color=COL["muted2"], ha="center", ls=1.05)

def draw_database(ax, cx, cy, color):
    ww, hh = 0.023, 0.048
    ax.add_patch(Rectangle((cx - ww / 2, cy - hh / 2), ww, hh,
                           facecolor=color, edgecolor="#454545", linewidth=0.86, zorder=4))
    ax.add_patch(Ellipse((cx, cy + hh / 2), ww, 0.013,
                         facecolor=color, edgecolor="#454545", linewidth=0.86, zorder=5))
    ax.add_patch(Ellipse((cx, cy), ww, 0.013,
                         facecolor="none", edgecolor="#454545", linewidth=0.74, zorder=5))
    ax.add_patch(Ellipse((cx, cy - hh / 2), ww, 0.013,
                         facecolor="#ffffff", edgecolor="#454545", linewidth=0.74, zorder=5))

def split_box(ax, x, y, w, h, label, color):
    rounded_box(ax, x, y, w, h, fc="#fbfbfb", ec=color, lw=1.16, r=0.009, z=4)
    text(ax, x + w / 2, y + h * 0.75, label, size=7.35, weight="bold", ha="center")
    xs = [x + w * 0.28, x + w * 0.50, x + w * 0.72]
    ys = [y + h * 0.56, y + h * 0.40, y + h * 0.24]
    for yy in ys:
        for xx in xs:
            ax.add_patch(Circle((xx, yy), 0.00280, facecolor=color, edgecolor=color, zorder=5))

def draw_panel2(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, lw=1.28, r=0.024)
    text(ax, x + w / 2, y + h * 0.922, "Leakage-controlled\nsplit",
         size=8.10, weight="bold", ha="center", ls=1.0)

    centers = [x + w * 0.24, x + w * 0.50, x + w * 0.76]
    colors = [COL["green"], COL["blue"], COL["yellow"]]
    for cx, lab, col in zip(centers, ["Source A", "Source B", "Source C"], colors):
        text(ax, cx, y + h * 0.760, lab, size=6.10, ha="center")
        draw_database(ax, cx, y + h * 0.650, col)

    bw = w * 0.195
    bh = h * 0.245
    by = y + h * 0.305
    box_top = by + bh + 0.016
    src_bottom = y + h * 0.610

    for cx, lab, col in zip(centers, ["Train", "Val", "Test"], colors):
        bx = cx - bw / 2
        ax.add_line(Line2D([cx, cx], [src_bottom, box_top],
                           linestyle=(0, (4, 3)), color="#5a5a5a", linewidth=0.80, zorder=4))
        split_box(ax, bx, by, bw, bh, lab, col)

    ax.add_line(Line2D([centers[0], centers[-1]], [box_top, box_top],
                       linestyle=(0, (4, 3)), color="#5a5a5a", linewidth=0.80, zorder=4))

    rounded_box(ax, x + w * 0.125, y + h * 0.086, w * 0.750, h * 0.112,
                fc=COL["soft_gray"], ec=COL["light_edge"], lw=0.70, r=0.012, z=3)
    text(ax, x + w / 2, y + h * 0.142,
         "Remove duplicates\nGroup-aware split\nBalance density",
         size=5.50, weight="bold", ha="center", color=COL["muted2"], ls=1.10)

def draw_network(ax, cx, cy, scale=1.0):
    left = [(cx - 0.034 * scale, cy + 0.027 * scale),
            (cx - 0.034 * scale, cy),
            (cx - 0.034 * scale, cy - 0.027 * scale)]
    mid = [(cx, cy + 0.033 * scale), (cx, cy), (cx, cy - 0.033 * scale)]
    right = [(cx + 0.034 * scale, cy + 0.017 * scale),
             (cx + 0.034 * scale, cy - 0.017 * scale)]

    for a in left:
        for b in mid:
            ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color="#252525", linewidth=1.00, zorder=4))
    for a in mid:
        for b in right:
            ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color="#252525", linewidth=1.00, zorder=4))

    for p in left + right:
        ax.add_patch(Circle(p, 0.0062 * scale, facecolor="#ffffff", edgecolor="#252525", linewidth=0.98, zorder=5))
    for p in mid:
        ax.add_patch(Circle(p, 0.0068 * scale, facecolor="#d7ece8", edgecolor="#252525", linewidth=0.98, zorder=5))

def draw_monitor(ax, cx, cy, scale=1.0):
    ww, hh = 0.072 * scale, 0.052 * scale
    ax.add_patch(Rectangle((cx - ww / 2, cy - hh / 2), ww, hh, fill=False, edgecolor="#252525", linewidth=1.10, zorder=4))
    ax.add_patch(Rectangle((cx - ww * 0.34, cy - hh * 0.25), ww * 0.68, hh * 0.50, fill=False, edgecolor="#969696", linewidth=0.76, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.0050 * scale, facecolor="none", edgecolor=COL["accent"], linewidth=1.70, zorder=5))
    ax.add_line(Line2D([cx, cx], [cy - hh / 2, cy - hh / 2 - 0.013 * scale], color="#252525", linewidth=1.10, zorder=4))
    ax.add_line(Line2D([cx - 0.011 * scale, cx + 0.011 * scale], [cy - hh / 2 - 0.013 * scale, cy - hh / 2 - 0.013 * scale], color="#252525", linewidth=1.10, zorder=4))

def draw_panel3(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, lw=1.28, r=0.024)

    # moved downward to create more space from top border
    text(ax, x + w / 2, y + h * 0.922, "Controlled resolution\nexperiment",
         size=7.15, weight="bold", ha="center", ls=1.0)

    # enlarged label box to prevent overflow
    rounded_box(ax, x + w * 0.095, y + h * 0.742, w * 0.810, h * 0.076,
                fc=COL["soft_gray"], ec=COL["light_edge"], lw=0.68, r=0.010, z=3)
    text(ax, x + w / 2, y + h * 0.780, "Detector: YOLO11n",
         size=5.95, weight="bold", ha="center", color=COL["muted2"])

    draw_network(ax, x + w / 2, y + h * 0.565, scale=1.48)
    text(ax, x + w * 0.205, y + h * 0.425, "...", size=8.9)
    text(ax, x + w * 0.795, y + h * 0.425, "...", size=8.9, ha="right")
    draw_monitor(ax, x + w / 2, y + h * 0.315, scale=1.48)

    text(ax, x + w / 2, y + h * 0.176, "800 / 960 / 1280 px",
         size=6.25, ha="center", weight="bold")

    rounded_box(ax, x + w * 0.125, y + h * 0.054, w * 0.750, h * 0.063,
                fc="#ffffff", ec="#dddddd", lw=0.56, r=0.010, z=3)
    text(ax, x + w / 2, y + h * 0.085,
         "Resolution\nvaries only",
         size=4.95, weight="bold", ha="center", color=COL["muted2"], ls=0.98)

def output_box(ax, x, y, w, h, title, bullets, title_size=6.0, bullet_size=5.15):
    rounded_box(ax, x, y, w, h, fc="#fcfcfc", ec=COL["box"], lw=0.84, r=0.010, z=4)
    text(ax, x + w * 0.060, y + h * 0.710, title, size=title_size, weight="bold", ls=1.0)
    text(ax, x + w * 0.092, y + h * 0.365, bullets, size=bullet_size, ls=1.18)

def draw_panel4(ax, x, y, w, h):
    rounded_box(ax, x, y, w, h, lw=1.28, r=0.024)
    text(ax, x + w / 2, y + h * 0.922, "Evaluation outputs",
         size=8.70, weight="bold", ha="center")

    bx, bw, bh = x + w * 0.080, w * 0.840, h * 0.145
    ys = [y + h * 0.650, y + h * 0.460, y + h * 0.270, y + h * 0.080]
    titles = ["Detection performance", "Scale-density analysis", "Counting reliability", "Deployment efficiency"]
    bullets = [
        "鈥?AP / AP50\n鈥?max_det",
        "鈥?ordinary AP\n鈥?strict AP",
        "鈥?bootstrap CIs",
        "鈥?latency\n鈥?VRAM",
    ]
    title_sizes = [5.75, 5.45, 5.75, 5.75]
    bullet_sizes = [5.30, 5.30, 5.45, 5.30]
    for yy, title_, bullet_, ts, bs in zip(ys, titles, bullets, title_sizes, bullet_sizes):
        output_box(ax, bx, yy, bw, bh, title_, bullet_, title_size=ts, bullet_size=bs)

def compose(out_dir: Path, stem: str):
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=COL["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COL["bg"])

    left = 0.016
    gap = 0.028
    widths = [0.247, 0.236, 0.198, 0.205]
    xs = [left]
    for i in range(1, 4):
        xs.append(xs[-1] + widths[i - 1] + gap)

    y, hh = 0.035, 0.93
    draw_panel1(ax, xs[0], y, widths[0], hh)
    draw_panel2(ax, xs[1], y, widths[1], hh)
    draw_panel3(ax, xs[2], y, widths[2], hh)
    draw_panel4(ax, xs[3], y, widths[3], hh)

    cy = y + hh * 0.505
    for i in range(3):
        gap_left = xs[i] + widths[i]
        gap_right = xs[i + 1]
        x0 = gap_left + gap * 0.10
        x1 = gap_right - gap * 0.10
        arrow(ax, x0, cy, x1, cy)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"
    png = out_dir / f"{stem}.png"
    tif = out_dir / f"{stem}.tif"

    fig.savefig(pdf, facecolor=COL["bg"], bbox_inches=None, pad_inches=0)
    fig.savefig(svg, facecolor=COL["bg"], bbox_inches=None, pad_inches=0)
    fig.savefig(png, dpi=DPI, facecolor=COL["bg"], bbox_inches=None, pad_inches=0)
    fig.savefig(tif, dpi=DPI, facecolor=COL["bg"], bbox_inches=None, pad_inches=0,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("[Fig1 resolution diagnosis workflow] saved:")
    print(pdf)
    print(svg)
    print(png)
    print(tif)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--stem", type=str, default="Fig1_resolution_diagnosis_workflow")
    args = parser.parse_args()
    compose(Path(args.out_dir), args.stem)

if __name__ == "__main__":
    main()

