import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

train_data = pd.read_excel("../DAta/Train.xlsx")
val_data = pd.read_excel("../DAta/Validation.xlsx")
test_data = pd.read_excel("../DAta/Test.xlsx")

X_train = train_data.drop(columns=["strength"]).values
y_train = train_data[["strength"]].values
X_val = val_data.drop(columns=["strength"]).values
y_val = val_data[["strength"]].values
X_test = test_data.drop(columns=["strength"]).values
y_test = test_data[["strength"]].values

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(16, activation="elu"),
    keras.layers.Dense(1, activation="linear"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=[early_stop],
)

os.makedirs("../Results", exist_ok=True)

results = pd.DataFrame(history.history)
results.index.name = "epoch"
results.to_excel("../Results/improved_model_epochs.xlsx")
