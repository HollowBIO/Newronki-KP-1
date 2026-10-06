import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
RESULTS_DIR = KP2_DIR / "Results"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

keras.utils.set_random_seed(SEED)

# Load unbatched and unshuffled so the decoded images can be cached in memory,
# then reshuffle the train set every epoch
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=None, shuffle=False)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "val", image_size=IMG_SIZE, color_mode="rgb",
    batch_size=None, shuffle=False)
n_train = len(train_ds.file_paths)

train_ds = train_ds.cache().shuffle(n_train, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.layers.Input(shape=IMG_SIZE + (3,)),
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True,
)
# Saves the model only when val_loss improves, so the file always holds the best epoch
checkpoint = keras.callbacks.ModelCheckpoint(
    RESULTS_DIR / "base_cnn.keras", monitor="val_loss", save_best_only=True,
)
timer = EpochTimer()

RESULTS_DIR.mkdir(exist_ok=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, checkpoint, timer],
)

results = pd.DataFrame(history.history)
results["epoch_time_s"] = timer.times
results.index.name = "epoch"
results.to_excel(RESULTS_DIR / "base_cnn_epochs.xlsx")

best = results["val_loss"].idxmin()
print(f"best epoch: {best}, val_loss {results.loc[best, 'val_loss']:.4f}, "
      f"val_accuracy {results.loc[best, 'val_accuracy']:.4f}")
