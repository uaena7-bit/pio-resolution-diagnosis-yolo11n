from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "supplementary_source_data" / "FigS3_validation_confidence_MAE_sweep_source_data.csv")
out = ROOT / "supplementary_figures"
out.mkdir(exist_ok=True)

styles = {
    "800": ("o", "-"),
    "960": ("s", "--"),
    "1024": ("^", "-."),
    "1280": ("D", ":"),
}
fig, axes = plt.subplots(2, 1, figsize=(6.85, 6.75))
for resolution in ["800","960","1024","1280"]:
    col = f"MAE_{resolution}px"
    marker, ls = styles[resolution]
    axes[0].plot(df["confidence_threshold"], df[col], marker=marker,
                 linestyle=ls, linewidth=1.3, markersize=4, label=f"{resolution} px")
for resolution in ["800","960","1280"]:
    col = f"MAE_{resolution}px"
    marker, ls = styles[resolution]
    axes[1].plot(df["confidence_threshold"], df[col], marker=marker,
                 linestyle=ls, linewidth=1.3, markersize=4, label=f"{resolution} px")
axes[0].set_ylabel("Validation counting MAE")
axes[1].set_ylabel("Validation counting MAE")
axes[1].set_xlabel("Confidence threshold")
axes[0].set_title("(a) Full-range sweep including exploratory 1024 px", loc="left")
axes[1].set_title("(b) Enlarged low-error region for primary settings", loc="left")
axes[1].set_xlim(0.35, 0.60)
axes[1].set_ylim(10.5, 18.5)
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center")
fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.09, hspace=0.35)
for ext in ["png","pdf","tif"]:
    fig.savefig(out / f"FigS3_validation_confidence_MAE_sweep.{ext}", dpi=600)
plt.close(fig)
