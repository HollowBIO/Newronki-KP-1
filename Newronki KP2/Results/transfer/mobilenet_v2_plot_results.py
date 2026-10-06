from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent

data = pd.read_excel(RESULTS_DIR / "mobilenet_v2_epochs.xlsx")
fine_tune_start = data.loc[data["phase"] == 2, "epoch"].min()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(data["epoch"], data["loss"], label="train")
axes[0].plot(data["epoch"], data["val_loss"], label="val")
axes[0].set_yscale("log")
axes[0].set_title("Loss (cross-entropy, log scale)")
axes[0].set_xlabel("epoch")

axes[1].plot(data["epoch"], data["accuracy"], label="train")
axes[1].plot(data["epoch"], data["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")

for ax in axes:
    ax.axvline(fine_tune_start - 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tuning starts")
    ax.legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "mobilenet_v2_history.png")
