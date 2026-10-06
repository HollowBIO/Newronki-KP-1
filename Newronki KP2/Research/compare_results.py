from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from architectures import VARIANTS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results" / "research"
GROUPS = {
    "Кількість блоків згортка + пулінг": ["one_block", "base", "five_blocks"],
    "Порядок шарів згортки та пулінгу": ["pool_first", "base", "pool_last"],
    "Склад блоку": ["base", "double_conv_pool"],
}
# Trained and kept in research_summary.xlsx, but only mentioned in the report text
NOT_CHARTED = {"double_conv"}
# Fixed colour per variant, so base looks the same in every panel
COLORS = {name: f"C{i}" for i, name in enumerate(VARIANTS)}

done = [n for n in VARIANTS if (RESULTS_DIR / f"{n}_test_metrics.xlsx").exists()]
summary = pd.concat(
    [pd.read_excel(RESULTS_DIR / f"{n}_test_metrics.xlsx", sheet_name="summary") for n in done],
    ignore_index=True,
)
summary["val_test_gap"] = summary["val_accuracy"] - summary["test_accuracy"]
# Median, because epochs that save a checkpoint (269 MB for one_block) take up to 2× longer
summary["epoch_time_s"] = [pd.read_excel(RESULTS_DIR / f"{n}_epochs.xlsx")["epoch_time_s"].median()
                           for n in summary["variant"]]
summary.to_excel(RESULTS_DIR / "research_summary.xlsx", index=False)
print(summary[["variant", "layers", "params", "receptive_field_px", "epochs", "epoch_time_s",
               "val_accuracy", "test_accuracy", "test_loss", "test_macro_f1"]].to_string(index=False))

charted = summary[~summary["variant"].isin(NOT_CHARTED)].reset_index(drop=True)

# Figure 1: validation vs test accuracy (the gap = overfitting) and time per epoch
labels = [f"{r.variant}  {r.layers}" for r in charted.itertuples()]
y = range(len(charted))[::-1]
fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(12, 0.55 * len(charted) + 1.6), sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1]})
ax_acc.hlines(y, charted["test_accuracy"], charted["val_accuracy"], color="lightgray", linewidth=2, zorder=1)
ax_acc.scatter(charted["val_accuracy"], y, color="C1", s=60, label="val accuracy", zorder=2)
ax_acc.scatter(charted["test_accuracy"], y, color="C2", s=60, label="test accuracy", zorder=2)
for yi, acc in zip(y, charted["test_accuracy"]):
    ax_acc.annotate(f"{acc:.1%}".replace(".", ","), (acc, yi), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9)
ax_acc.set_yticks(list(y), labels, fontsize=9)
ax_acc.set_xlim(min(0.5, charted["test_accuracy"].min() - 0.08), 1.02)
ax_acc.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax_acc.set_title("Точність: валідація vs тест (відрізок = розрив)")
ax_acc.grid(axis="x", alpha=0.3)
ax_acc.legend(loc="lower left")

ax_time.barh(list(y), charted["epoch_time_s"], color="C0", height=0.5)
for yi, t in zip(y, charted["epoch_time_s"]):
    ax_time.annotate(f"{t:.0f} с", (t, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
ax_time.set_xlim(0, charted["epoch_time_s"].max() * 1.25)
ax_time.set_title("Час однієї епохи, с")
ax_time.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "research_accuracy_time.png", dpi=150)
plt.close(fig)

# Figure 2: validation loss dynamics per group
fig, axes = plt.subplots(1, len(GROUPS), figsize=(5 * len(GROUPS), 4), sharey=True)
for ax, (title, names) in zip(axes, GROUPS.items()):
    for n in names:
        if n not in done:
            continue
        epochs = pd.read_excel(RESULTS_DIR / f"{n}_epochs.xlsx")
        ax.plot(epochs["epoch"], epochs["val_loss"], color=COLORS[n], linewidth=2, label=n)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("val loss (log)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "research_val_loss.png", dpi=150)
plt.close(fig)
