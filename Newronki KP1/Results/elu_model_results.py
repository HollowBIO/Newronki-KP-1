import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("elu_model_epochs.xlsx")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(data["epoch"], data["loss"], label="train")
axes[0].plot(data["epoch"], data["val_loss"], label="val")
axes[0].set_title("Loss (MSE)")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(data["epoch"], data["mae"], label="train")
axes[1].plot(data["epoch"], data["val_mae"], label="val")
axes[1].set_title("MAE")
axes[1].set_xlabel("epoch")
axes[1].legend()

fig.savefig("elu_model_history.png")
