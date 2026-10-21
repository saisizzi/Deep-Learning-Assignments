# ============================================================
# DEEP LEARNING ASSIGNMENT - 2
# MULTILAYER PERCEPTRON (MLP) FOR IRIS CLASSIFICATION
# ============================================================
# Objective:
# Design and implement a Multilayer Perceptron (MLP) for
# classification of the Iris dataset and evaluate its
# performance using Accuracy and a Confusion Matrix.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ------------------------------------------------------------
# 2. DISPLAY TENSORFLOW VERSION
# ------------------------------------------------------------

print("=" * 60)
print("TENSORFLOW / KERAS CONFIGURATION")
print("=" * 60)

print("TensorFlow Version:", tf.__version__)
print("Keras Version     :", keras.__version__)

print()


# ------------------------------------------------------------
# 3. LOAD IRIS DATASET
# ------------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names
feature_names = iris.feature_names


# ------------------------------------------------------------
# 4. DISPLAY DATASET INFORMATION
# ------------------------------------------------------------

print("=" * 60)
print("IRIS DATASET INFORMATION")
print("=" * 60)

print("Number of Samples :", X.shape[0])
print("Number of Features:", X.shape[1])
print("Number of Classes :", len(np.unique(y)))

print("\nFeatures:")
for feature in feature_names:
    print("-", feature)

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

print("\nFeature Shape:", X.shape)
print("Target Shape :", y.shape)

print()


# ------------------------------------------------------------
# 5. DISPLAY FIRST FEW SAMPLES
# ------------------------------------------------------------

print("=" * 60)
print("FIRST 5 SAMPLES")
print("=" * 60)

print(X[:5])

print("\nCorresponding Labels:")
print(y[:5])

print()


# ------------------------------------------------------------
# 6. VISUALIZE IRIS DATASET
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Iris Dataset Visualization")

plt.colorbar(
    scatter,
    ticks=[0, 1, 2],
    label="Class"
)

plt.show()


# ------------------------------------------------------------
# 7. TRAIN-TEST SPLIT
# ------------------------------------------------------------
# 80% Training Data
# 20% Testing Data
#
# stratify=y keeps class proportions similar in both sets.
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 8. DISPLAY TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

print()


# ------------------------------------------------------------
# 9. FEATURE STANDARDIZATION
# ------------------------------------------------------------
# Standardization transforms features approximately to:
#
# Mean = 0
# Standard Deviation = 1
#
# IMPORTANT:
# The scaler is fitted only on training data.
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# 10. DISPLAY STANDARDIZED DATA
# ------------------------------------------------------------

print("=" * 60)
print("STANDARDIZED DATA")
print("=" * 60)

print("First 5 standardized training samples:\n")
print(X_train[:5])

print()


# ------------------------------------------------------------
# 11. BUILD MULTILAYER PERCEPTRON (MLP)
# ------------------------------------------------------------
#
# Architecture:
#
# Input Layer
#     ↓
# Dense Layer - 16 Neurons - ReLU
#     ↓
# Dense Layer - 8 Neurons - ReLU
#     ↓
# Output Layer - 3 Neurons - Softmax
#
# Iris has 3 classes:
# Setosa, Versicolor, Virginica
# ------------------------------------------------------------

model = keras.Sequential([

    keras.layers.Input(
        shape=(4,)
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        8,
        activation="relu"
    ),

    keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# ------------------------------------------------------------
# 12. DISPLAY MODEL ARCHITECTURE
# ------------------------------------------------------------

print("=" * 60)
print("MLP MODEL ARCHITECTURE")
print("=" * 60)

model.summary()

print()


# ------------------------------------------------------------
# 13. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 14. TRAIN THE MLP
# ------------------------------------------------------------

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=8,
    verbose=1
)


# ------------------------------------------------------------
# 15. EVALUATE MODEL ON TEST DATA
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print()


# ------------------------------------------------------------
# 16. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

# Convert probability outputs into class labels
y_pred = np.argmax(
    predictions,
    axis=1
)


# ------------------------------------------------------------
# 17. CALCULATE ACCURACY USING SKLEARN
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 60)
print("ACCURACY SCORE")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()


# ------------------------------------------------------------
# 18. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print()


# ------------------------------------------------------------
# 19. VISUALIZE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix - Iris MLP")
plt.show()


# ------------------------------------------------------------
# 20. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)


# ------------------------------------------------------------
# 21. PLOT TRAINING AND VALIDATION ACCURACY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 22. PLOT TRAINING AND VALIDATION LOSS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 23. DISPLAY ACTUAL VS PREDICTED LABELS
# ------------------------------------------------------------

print("=" * 60)
print("ACTUAL VS PREDICTED LABELS")
print("=" * 60)

for i in range(len(y_test)):

    actual = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]

    print(
        f"Sample {i + 1:2d} | "
        f"Actual: {actual:10s} | "
        f"Predicted: {predicted}"
    )


# ------------------------------------------------------------
# 24. VISUALIZE PREDICTIONS
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

sample_count = min(15, len(X_test))

for i in range(sample_count):

    plt.subplot(3, 5, i + 1)

    plt.bar(
        range(3),
        predictions[i]
    )

    plt.xticks(
        range(3),
        ["Setosa", "Versicolor", "Virginica"],
        rotation=45
    )

    plt.ylim(0, 1)

    plt.title(
        f"Actual: {class_names[y_test[i]]}\n"
        f"Pred: {class_names[y_pred[i]]}"
    )

    plt.ylabel("Probability")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 25. FINAL RESULT
# ------------------------------------------------------------

print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Number of Classes: {len(class_names)}")
print(f"MLP Test Accuracy: {accuracy * 100:.2f}%")

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

print("\nAssignment 2 Completed Successfully!")
print("=" * 60)