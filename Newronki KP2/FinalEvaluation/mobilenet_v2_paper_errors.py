import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
RESULTS_DIR = KP2_DIR / "Results" / "transfer"
CLASSES = ["paper", "rock", "scissors"]
TRAIN_EXAMPLE = DATA_DIR / "train" / "paper" / "paper05-060.png"  # typical training "paper": spread fingers

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "rps-test-set", image_size=(224, 224), batch_size=32, shuffle=False)
y_true = np.concatenate([y.numpy() for _, y in test_ds])
hands = np.array([re.match(r"^(.+)-\d+\.png$", Path(p).name).group(1) for p in test_ds.file_paths])

# Final model = after fine-tuning (phase 2)
model = keras.models.load_model(RESULTS_DIR / "mobilenet_v2.keras")
y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

paper_hands = sorted(set(hands[y_true == 0]))
fig, axes = plt.subplots(1, len(paper_hands) + 1, figsize=(2.4 * (len(paper_hands) + 1), 2.9))
for ax in axes:
    ax.axis("off")
axes[0].imshow(keras.utils.load_img(TRAIN_EXAMPLE))
axes[0].set_title("навчальна вибірка:\npaper", fontsize=9)
for ax, hand in zip(axes[1:], paper_hands):
    idx = np.flatnonzero(hands == hand)
    wrong = [i for i in idx if y_pred[i] != y_true[i]]
    i = wrong[len(wrong) // 2]
    ax.imshow(keras.utils.load_img(test_ds.file_paths[i]))
    ax.set_title(f"тест {hand}: → {CLASSES[y_pred[i]]}\n(помилок {len(wrong)}/{len(idx)})", fontsize=9)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "mobilenet_v2_paper_errors.png", dpi=150)
