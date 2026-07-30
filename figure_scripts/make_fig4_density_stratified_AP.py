from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "figure_source_data" / "Fig4_density_stratified_AP_source_data.csv")
out = ROOT / "main_figures"
out.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6.85, 4.25))
styles = [("low","o","-"),("medium","s","--"),("high","^","-."),("ultra-high","D",":")]
for name, marker, ls in styles:
    ax.plot(df["resolution_px"], df[name], marker=marker, linestyle=ls,
            linewidth=1.3, markersize=5, label=name)
ax.set_xlabel("Input resolution (px)")
ax.set_ylabel("Ordinary density-stratified AP (mAP50-95)", labelpad=10)
ax.set_xticks([800, 960, 1280])
ax.set_ylim(0.635, 0.78)
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.26))
fig.subplots_adjust(left=0.15, right=0.98, top=0.965, bottom=0.30)
for ext in ["png", "pdf", "tif"]:
    fig.savefig(out / f"Fig4_Density_Stratified_AP.{ext}", dpi=600,
                bbox_inches=None, pad_inches=0.12)
plt.close(fig)
