import pandas as pd
import matplotlib.pyplot as plt

d00 = pd.read_excel("relu_model_epochs.xlsx")
d02 = pd.read_excel("dropout02_model_epochs.xlsx")
d05 = pd.read_excel("dropout05_model_epochs.xlsx")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(d00["epoch"], d00["val_loss"], label="dropout=0.0")
axes[0].plot(d02["epoch"], d02["val_loss"], label="dropout=0.2")
axes[0].plot(d05["epoch"], d05["val_loss"], label="dropout=0.5")
axes[0].set_title("Validation Loss (MSE)")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(d00["epoch"], d00["val_mae"], label="dropout=0.0")
axes[1].plot(d02["epoch"], d02["val_mae"], label="dropout=0.2")
axes[1].plot(d05["epoch"], d05["val_mae"], label="dropout=0.5")
axes[1].set_title("Validation MAE")
axes[1].set_xlabel("epoch")
axes[1].legend()

fig.savefig("dropout_comparison_plot.png")
