from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "supplementary_figures"
out.mkdir(exist_ok=True)
sources = [
    ("a", ROOT / "supplementary_source_data" / "FigS2a_scale_defined_SGF_AP_source_data.csv",
     [("tiny","o","-"),("small","s","--"),("medium","^","-."),("large","D",":")]),
    ("b", ROOT / "supplementary_source_data" / "FigS2b_density_defined_SGF_AP_source_data.csv",
     [("low","o","-"),("medium","s","--"),("high","^","-."),("ultra-high","D",":")]),
]
pngs = []
for panel, path, styles in sources:
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.85, 3.86))
    for name, marker, ls in styles:
        ax.plot(df["resolution_px"], df[name], marker=marker, linestyle=ls,
                linewidth=1.3, markersize=5, label=name)
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Strict diagnostic AP (mAP50-95)")
    ax.set_xticks([800, 960, 1280])
    ax.grid(alpha=0.25)
    ax.text(0.0, 1.02, f"({panel})", transform=ax.transAxes, fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.33))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.37)
    p = out / f"FigS2{panel}.png"
    fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / f"FigS2{panel}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    pngs.append(Image.open(p).convert("RGB"))
gap = round(14 / 25.4 * 600)
combined = Image.new("RGB", (max(i.width for i in pngs), sum(i.height for i in pngs)+gap), "white")
combined.paste(pngs[0], ((combined.width-pngs[0].width)//2, 0))
combined.paste(pngs[1], ((combined.width-pngs[1].width)//2, pngs[0].height+gap))
combined.save(out / "FigureS2_strict_diagnostic_AP_line_style.png", dpi=(600,600))
combined.save(out / "FigureS2_strict_diagnostic_AP_line_style.tif", dpi=(600,600), compression="tiff_lzw")
