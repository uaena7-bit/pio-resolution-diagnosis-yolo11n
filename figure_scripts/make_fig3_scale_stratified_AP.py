from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "figure_source_data" / "Fig3_scale_stratified_AP_source_data.csv")
out = ROOT / "main_figures"
out.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6.7, 3.8))
styles = [("tiny","o","-"),("small","s","--"),("medium","^","-."),("large","D",":")]
for name, marker, ls in styles:
    ax.plot(df["resolution_px"], df[name], marker=marker, linestyle=ls,
            linewidth=1.4, markersize=5, label=name)
ax.set_xlabel("Input resolution (px)")
ax.set_ylabel("Ordinary scale-stratified AP (mAP50-95)")
ax.set_xticks([800, 960, 1280])
ax.grid(alpha=0.25)
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.30)
for ext in ["png", "pdf", "tif"]:
    fig.savefig(out / f"Fig3_Scale_Stratified_AP.{ext}", dpi=600)
plt.close(fig)
