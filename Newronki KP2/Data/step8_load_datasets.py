from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parent
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH_SIZE, shuffle=False)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "rps-test-set", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
print(f"\nclass names (label 0, 1, 2): {class_names}")
assert val_ds.class_names == class_names, "train/val class order differs"
assert test_ds.class_names == class_names, "train/test class order differs"

for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    labels = np.concatenate([y.numpy() for _, y in ds])
    per_class = Counter(class_names[i] for i in labels)
    print(f"{name:<5} {len(labels):>5} images  {dict(sorted(per_class.items()))}")

file_names = {name: {Path(p).name for p in ds.file_paths}
              for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]}
print(f"files in both train and val:  {len(file_names['train'] & file_names['val'])}")
print(f"files in both train and test: {len(file_names['train'] & file_names['test'])}")

images, labels = next(iter(train_ds))
print(f"batch shape: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
print(f"batch dtype: {images.dtype}, value range [{images.numpy().min():.0f}, {images.numpy().max():.0f}]")
