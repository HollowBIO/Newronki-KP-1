import pandas as pd
import matplotlib.pyplot as plt

relu = pd.read_excel("relu_model_epochs.xlsx")
tanh = pd.read_excel("tanh_model_epochs.xlsx")
elu = pd.read_excel("elu_model_epochs.xlsx")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(relu["epoch"], relu["val_loss"], label="relu")
axes[0].plot(tanh["epoch"], tanh["val_loss"], label="tanh")
axes[0].plot(elu["epoch"], elu["val_loss"], label="elu")
axes[0].set_title("Validation Loss (MSE)")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(relu["epoch"], relu["val_mae"], label="relu")
axes[1].plot(tanh["epoch"], tanh["val_mae"], label="tanh")
axes[1].plot(elu["epoch"], elu["val_mae"], label="elu")
axes[1].set_title("Validation MAE")
axes[1].set_xlabel("epoch")
axes[1].legend()

fig.savefig("study_a_comparison.png")
