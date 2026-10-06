"""Report figure: one misclassified test image per (test hand, wrong class), most confident first."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
RESULTS_DIR = KP2_DIR / "Results"
CLASSES = ["paper", "rock", "scissors"]

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "rps-test-set", image_size=(150, 150), batch_size=32, shuffle=False)
model = keras.models.load_model(RESULTS_DIR / "base_cnn.keras")
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)


def hand(i):
    return re.match(r"^(.+)-\d+\.png$", Path(test_ds.file_paths[i]).name).group(1)


picked, seen = [], set()
for i in sorted(np.flatnonzero(y_pred != y_true), key=lambda i: -y_prob[i].max()):
    if (hand(i), y_pred[i]) not in seen:
        seen.add((hand(i), y_pred[i]))
        picked.append(i)
picked = picked[:6]

fig, axes = plt.subplots(1, len(picked), figsize=(2 * len(picked), 2.5))
for ax, i in zip(np.atleast_1d(axes), picked):
    ax.imshow(keras.utils.load_img(test_ds.file_paths[i]))
    ax.set_title(f"справжній: {CLASSES[y_true[i]]}\nпрогноз: {CLASSES[y_pred[i]]} ({y_prob[i].max():.0%})",
                 fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "base_cnn_error_examples.png", dpi=150)
