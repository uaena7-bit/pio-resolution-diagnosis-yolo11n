from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "figure_source_data" / "Fig2_training_time_accuracy_tradeoff_source_data.csv")
out = ROOT / "main_figures"
out.mkdir(exist_ok=True)

primary = df[df["resolution_px"].isin([800, 960, 1280])].sort_values("resolution_px")
exploratory = df[df["resolution_px"] == 1024]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75))
for ax, y, ylabel in [
    (axes[0], "test_mAP50_95", "Test mAP50-95"),
    (axes[1], "training_time_h", "Training time (h)"),
]:
    ax.plot(primary["resolution_px"], primary[y], marker="o", linewidth=1.4,
            label="Primary operating points")
    ax.plot(exploratory["resolution_px"], exploratory[y], marker="D",
            markerfacecolor="white", linestyle="None", label="Exploratory 1024 px")
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([800, 960, 1024, 1280])
    ax.grid(alpha=0.25)
axes[0].text(0.0, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
axes[1].text(0.0, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.25, wspace=0.28)
for ext in ["png", "pdf", "tif"]:
    fig.savefig(out / f"Fig2_Accuracy_Training_Cost.{ext}", dpi=600)
plt.close(fig)
