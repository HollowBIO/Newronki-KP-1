import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow import keras

from architectures import IMG_SIZE, VARIANTS, build_model, flatten_size, receptive_field

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
OUT_DIR = KP2_DIR / "Results" / "research"
BATCH_SIZE = 32
SEED = 42


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)


def load(folder):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / folder, image_size=IMG_SIZE, color_mode="rgb",
        batch_size=None, shuffle=False)


def run(name, train_images, n_train, val_ds, test_ds, y_test, class_names):
    spec = VARIANTS[name]
    print(f"\n===== {name}: {spec} =====")
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    # Fresh shuffle per variant, so every variant sees the same order of batches
    train_ds = train_images.shuffle(n_train, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model(spec)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    weights_path = OUT_DIR / f"{name}.weights.h5"
    timer = EpochTimer()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(weights_path, monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        timer,
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks, verbose=2)
    model.load_weights(weights_path)

    epochs = pd.DataFrame(history.history)
    epochs["epoch_time_s"] = timer.times
    epochs.index.name = "epoch"
    epochs.to_excel(OUT_DIR / f"{name}_epochs.xlsx")

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    best = epochs["val_loss"].idxmin()
    tokens = spec.split("-")

    summary = pd.DataFrame([{
        "variant": name,
        "layers": spec,
        "conv_layers": sum(t != "P" for t in tokens),
        "pool_layers": tokens.count("P"),
        "params": model.count_params(),
        "flatten_size": flatten_size(model),
        "receptive_field_px": receptive_field(spec),
        "epochs": len(epochs),
        "best_epoch": best,
        "epoch_time_s": epochs["epoch_time_s"].mean(),
        "train_time_s": epochs["epoch_time_s"].sum(),
        "train_accuracy": epochs.loc[best, "accuracy"],
        "val_accuracy": epochs.loc[best, "val_accuracy"],
        "val_loss": epochs.loc[best, "val_loss"],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_macro_f1": f1_score(y_test, y_pred, average="macro"),
    }])
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=[f"true {c}" for c in class_names],
                      columns=[f"pred {c}" for c in class_names])
    with pd.ExcelWriter(OUT_DIR / f"{name}_test_metrics.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        cm.to_excel(writer, sheet_name="confusion_matrix")

    print(summary.T.to_string(header=False))


if __name__ == "__main__":
    # Usage: python train_variant.py base pool_first ...   or   python train_variant.py all
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        # pool_last is by far the slowest (convolutions on full-size images), so it goes last
        names = [n for n in VARIANTS if n != "pool_last"] + ["pool_last"]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants: {unknown}, choose from {list(VARIANTS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = load("train")
    n_train = len(train_raw.file_paths)
    train_images = train_raw.cache()
    val_ds = load("val").cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_raw = load("rps-test-set")
    class_names = test_raw.class_names
    test_ds = test_raw.batch(BATCH_SIZE).cache()
    y_test = np.concatenate([y.numpy() for _, y in test_ds])

    for name in names:
        run(name, train_images, n_train, val_ds, test_ds, y_test, class_names)
