import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow import keras

KP2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = KP2_DIR / "Data"
RESULTS_DIR = KP2_DIR / "Results" / "transfer"
IMG_SIZE = (224, 224)  # MobileNetV2 was pretrained on 224×224 ImageNet images
BATCH_SIZE = 32
SEED = 42
FINE_TUNE_LAYERS = 30  # top layers of MobileNetV2 unfrozen in phase 2

keras.utils.set_random_seed(SEED)


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)


def load(folder):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / folder, image_size=IMG_SIZE, color_mode="rgb", batch_size=None, shuffle=False)


train_raw = load("train")
n_train = len(train_raw.file_paths)
train_ds = train_raw.cache().shuffle(n_train, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = load("val").cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_raw = load("rps-test-set")
class_names = test_raw.class_names
test_ds = test_raw.batch(BATCH_SIZE).cache()
y_test = np.concatenate([y.numpy() for _, y in test_ds])

# Pretrained body without its 1000-class ImageNet head
base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)  # same as mobilenet_v2.preprocess_input: [-1, 1]
x = base(x, training=False)  # keeps BatchNorm in inference mode, also while fine-tuning
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_phase(phase, learning_rate, epochs, initial_epoch=0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"\n===== phase {phase}: lr {learning_rate}, trainable params {trainable:,} =====")

    weights_path = RESULTS_DIR / f"mobilenet_v2_phase{phase}.weights.h5"
    timer = EpochTimer()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(weights_path, monitor="val_loss",
                                            save_best_only=True, save_weights_only=True),
            timer,
        ],
    )
    model.load_weights(weights_path)

    epochs_df = pd.DataFrame(history.history)
    epochs_df.index = range(initial_epoch, initial_epoch + len(epochs_df))
    epochs_df["epoch_time_s"] = timer.times
    epochs_df["phase"] = phase

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    best = epochs_df["val_loss"].idxmin()
    summary = {
        "phase": phase,
        "learning_rate": learning_rate,
        "trainable_params": trainable,
        "epochs": len(epochs_df),
        "best_epoch": best,
        "epoch_time_s": epochs_df["epoch_time_s"].median(),
        "train_time_s": epochs_df["epoch_time_s"].sum(),
        "train_accuracy": epochs_df.loc[best, "accuracy"],
        "val_accuracy": epochs_df.loc[best, "val_accuracy"],
        "val_loss": epochs_df.loc[best, "val_loss"],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_macro_f1": f1.mean(),
    }
    per_class = pd.DataFrame({"phase": phase, "class": class_names,
                              "precision": precision, "recall": recall, "f1": f1})
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      index=[f"true {c}" for c in class_names],
                      columns=[f"pred {c}" for c in class_names])
    print(pd.Series(summary).to_string())
    print(cm)
    return epochs_df, summary, per_class, cm


# Phase 1: only the new head learns, the pretrained body is frozen
epochs1, summary1, per_class1, cm1 = train_phase(1, learning_rate=1e-3, epochs=10)

# Phase 2: unfreeze the top layers and adjust them gently with a 100× smaller learning rate
base.trainable = True
for layer in base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
epochs2, summary2, per_class2, cm2 = train_phase(2, learning_rate=1e-5, epochs=10,
                                                 initial_epoch=len(epochs1))

model.save(RESULTS_DIR / "mobilenet_v2.keras")

epochs_all = pd.concat([epochs1, epochs2])
epochs_all.index.name = "epoch"
epochs_all.to_excel(RESULTS_DIR / "mobilenet_v2_epochs.xlsx")

with pd.ExcelWriter(RESULTS_DIR / "mobilenet_v2_test_metrics.xlsx") as writer:
    pd.DataFrame([summary1, summary2]).to_excel(writer, sheet_name="summary", index=False)
    pd.concat([per_class1, per_class2]).to_excel(writer, sheet_name="per_class", index=False)
    cm1.to_excel(writer, sheet_name="confusion_phase1")
    cm2.to_excel(writer, sheet_name="confusion_phase2")
