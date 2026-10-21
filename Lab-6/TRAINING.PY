import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Configuration and reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)


# ============================================================
# 1. Load the dataset
# ============================================================

# This script expects the dataset in ../data relative to the
# current working directory. If it is missing, the CSV is downloaded.
DATA_DIR = "../data"
DATA_PATH = os.path.join(DATA_DIR, "airline-passengers.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/airline-passengers.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Dataset not found locally. Downloading...")
    pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    print("Dataset downloaded to:", DATA_PATH)
else:
    print("Using local dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()


# ============================================================
# 2. Create results directory and visualize the time series
# ============================================================

os.makedirs("./results", exist_ok=True)
print("Results directory ensured at ./results")

plt.figure(figsize=(12, 5))
plt.plot(df["Month"], df["Passengers"])
plt.title("Monthly Airline Passenger Counts")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./results/passenger_time_series.png", dpi=300)
plt.show()


# ============================================================
# 3. Chronological train-test split
# ============================================================

values = df[["Passengers"]].values.astype("float32")

train_size = int(len(values) * 0.80)

train_data = values[:train_size]
test_data = values[train_size:]

print("Total observations:", len(values))
print("Training observations:", len(train_data))
print("Testing observations:", len(test_data))


# ============================================================
# 4. Normalize the data
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on training data only to avoid data leakage.
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

print(
    "Training scaled range:",
    train_scaled.min(),
    "to",
    train_scaled.max()
)


# ============================================================
# 5. Create time-series sequences
# ============================================================

def create_sequences(data, look_back):
    """Create sequences of historical values and next-step targets."""
    X, y = [], []

    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def prepare_train_test_sequences(train_scaled, test_scaled, look_back):
    """Prepare training and test sequences for a look-back window."""
    X_train, y_train = create_sequences(train_scaled, look_back)

    # Include the final training observations so the first test
    # prediction has the required historical context.
    test_input = np.concatenate(
        [train_scaled[-look_back:], test_scaled],
        axis=0
    )

    X_test, y_test = create_sequences(test_input, look_back)

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    return X_train, y_train, X_test, y_test


# ============================================================
# 6. Define the LSTM model
# ============================================================

def build_lstm_model(look_back):
    """Build and compile an LSTM model for next-step forecasting."""
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 7. Train and evaluate the three experiments
# ============================================================

LOOK_BACK_WINDOWS = [6, 12, 24]

results = []
predictions = {}
histories = {}

for look_back in LOOK_BACK_WINDOWS:
    print("=" * 60)
    print(f"Training LSTM with {look_back}-month look-back")
    print("=" * 60)

    X_train, y_train, X_test, y_test = prepare_train_test_sequences(
        train_scaled,
        test_scaled,
        look_back
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    model = build_lstm_model(look_back)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.10,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.perf_counter() - start_time

    pred_scaled = model.predict(X_test, verbose=0)

    # Convert targets and predictions back to passenger counts.
    y_test_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    pred_actual = scaler.inverse_transform(
        pred_scaled
    ).flatten()

    mae = mean_absolute_error(y_test_actual, pred_actual)
    rmse = np.sqrt(
        mean_squared_error(y_test_actual, pred_actual)
    )

    results.append({
        "Look-back (months)": look_back,
        "MAE": mae,
        "RMSE": rmse,
        "Epochs": len(history.history["loss"]),
        "Training Time (seconds)": training_time
    })

    predictions[look_back] = {
        "actual": y_test_actual,
        "predicted": pred_actual
    }

    histories[look_back] = history

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Epochs: {len(history.history['loss'])}")
    print(f"Training time: {training_time:.2f} seconds")
    print()


# ============================================================
# 8. Compare experimental results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Look-back (months)"
).reset_index(drop=True)

print("\nModel comparison:")
print(results_df.round(2).to_string(index=False))


# ============================================================
# 9. Identify the best model by test RMSE
# ============================================================

best_row = results_df.loc[results_df["RMSE"].idxmin()]

print(
    "\nBest Look-back Window:",
    int(best_row["Look-back (months)"]),
    "months"
)
print(f"Best MAE: {best_row['MAE']:.2f}")
print(f"Best RMSE: {best_row['RMSE']:.2f}")


# ============================================================
# 10. Plot MAE and RMSE comparison
# ============================================================

x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(
    x - width / 2,
    results_df["MAE"],
    width,
    label="MAE"
)
plt.bar(
    x + width / 2,
    results_df["RMSE"],
    width,
    label="RMSE"
)

plt.xticks(
    x,
    results_df["Look-back (months)"].astype(str)
)
plt.xlabel("Look-back Window (Months)")
plt.ylabel("Error")
plt.title("MAE and RMSE Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("./results/window_metric_comparison.png", dpi=300)
plt.show()


# ============================================================
# 11. Plot actual vs. predicted values for all windows
# ============================================================

test_dates = df["Month"].iloc[train_size:].reset_index(drop=True)

plt.figure(figsize=(13, 6))

plt.plot(
    test_dates,
    predictions[6]["actual"],
    label="Actual",
    linewidth=2
)

for look_back in LOOK_BACK_WINDOWS:
    plt.plot(
        test_dates,
        predictions[look_back]["predicted"],
        label=f"LSTM ({look_back} months)"
    )

plt.title("Actual vs Predicted Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./results/all_windows_actual_vs_predicted.png", dpi=300)
plt.show()


# ============================================================
# 12. Generate individual prediction plots
# ============================================================

for look_back in LOOK_BACK_WINDOWS:
    plt.figure(figsize=(12, 5))

    plt.plot(
        test_dates,
        predictions[look_back]["actual"],
        label="Actual"
    )

    plt.plot(
        test_dates,
        predictions[look_back]["predicted"],
        label=f"Predicted ({look_back}-month window)"
    )

    plt.title(
        f"Actual vs Predicted — {look_back}-Month Look-back"
    )
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"./results/prediction_{look_back}_month_window.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 13. Plot training and validation loss
# ============================================================

for look_back in LOOK_BACK_WINDOWS:
    history = histories[look_back]

    plt.figure(figsize=(10, 5))
    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        f"Training and Validation Loss — {look_back}-Month Window"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"./results/loss_{look_back}_month_window.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 14. Save evaluation results
# ============================================================

results_df.to_csv("./results/model_comparison.csv", index=False)

print("\nEvaluation results saved to ./results/model_comparison.csv")
print("All plots saved to the ./results directory.")
