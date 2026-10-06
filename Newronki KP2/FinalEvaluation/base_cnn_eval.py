from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
RESULTS_DIR = KP2_DIR / "Results"
IMG_SIZE = (150, 150)  # must match training
BATCH_SIZE = 32

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "rps-test-set", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH_SIZE, shuffle=False)
class_names = test_ds.class_names

model = keras.models.load_model(RESULTS_DIR / "base_cnn.keras")

test_loss, test_accuracy = model.evaluate(test_ds)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = model.predict(test_ds)
y_pred = y_prob.argmax(axis=1)

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
per_class = pd.DataFrame({
    "class": class_names, "precision": precision, "recall": recall, "f1": f1, "support": support,
})
summary = pd.DataFrame([{
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "macro_precision": precision.mean(),
    "macro_recall": recall.mean(),
    "macro_f1": f1.mean(),
}])
cm = confusion_matrix(y_true, y_pred)
cm_table = pd.DataFrame(cm, index=[f"true {c}" for c in class_names],
                        columns=[f"pred {c}" for c in class_names])

with pd.ExcelWriter(RESULTS_DIR / "base_cnn_test_metrics.xlsx") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    per_class.to_excel(writer, sheet_name="per_class", index=False)
    cm_table.to_excel(writer, sheet_name="confusion_matrix")

print(summary.to_string(index=False))
print(per_class.to_string(index=False))
print(cm_table)

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
ax.set_title(f"Base CNN, test set (accuracy {test_accuracy:.1%})")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "base_cnn_confusion_matrix.png")
plt.close(fig)

# Show what the model gets wrong, with its confidence
wrong = np.flatnonzero(y_pred != y_true)
if len(wrong):
    shown = wrong[:12]
    cols = 6
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, shown):
        ax.imshow(keras.utils.load_img(test_ds.file_paths[i]))
        ax.set_title(f"true {class_names[y_true[i]]}\npred {class_names[y_pred[i]]} ({y_prob[i].max():.0%})",
                     fontsize=8)
    fig.suptitle(f"Misclassified test images ({len(shown)} of {len(wrong)})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "base_cnn_misclassified.png")
    plt.close(fig)
