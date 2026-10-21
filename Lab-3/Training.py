# ============================================================
# Deep Learning Lab 3
# Forward Propagation & Backpropagation
# Effect of Learning Rate and Number of Epochs
# Dataset: MNIST
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)


# -----------------------------
# 2. Load MNIST Dataset
# -----------------------------
print("\nLoading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)


# -----------------------------
# 3. Data Preprocessing
# -----------------------------

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nAfter normalization:")
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())


# -----------------------------
# 4. Visualize Sample Images
# -----------------------------
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle("MNIST Sample Images")
plt.tight_layout()
plt.show()


# -----------------------------
# 5. Create MLP Model
# -----------------------------
def create_model(learning_rate):

    model = keras.Sequential([
        layers.Input(shape=(28, 28)),

        # Flatten 28x28 image into 784 values
        layers.Flatten(),

        # Hidden Layer 1
        layers.Dense(128, activation="relu"),

        # Hidden Layer 2
        layers.Dense(64, activation="relu"),

        # Output Layer
        layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 6. Demonstrate Forward
#    Propagation
# -----------------------------
print("\n" + "=" * 60)
print("FORWARD PROPAGATION")
print("=" * 60)

model = create_model(0.001)

# Take one sample
sample = X_train[:1]

# Forward propagation
output = model(sample, training=False)

print("Input shape :", sample.shape)
print("Output shape:", output.shape)
print("\nOutput probabilities:")
print(output.numpy())

print("\nPredicted digit:", np.argmax(output.numpy()))
print("Actual digit   :", y_train[0])


# -----------------------------
# 7. Demonstrate Backpropagation
#    Using TensorFlow GradientTape
# -----------------------------
print("\n" + "=" * 60)
print("BACKPROPAGATION")
print("=" * 60)

# Create a fresh model
backprop_model = create_model(0.001)

# Select a small batch
x_batch = X_train[:32]
y_batch = y_train[:32]

loss_function = keras.losses.SparseCategoricalCrossentropy()

# Record operations for automatic differentiation
with tf.GradientTape() as tape:

    # Forward propagation
    predictions = backprop_model(x_batch, training=True)

    # Calculate loss
    loss = loss_function(y_batch, predictions)

# Backward propagation
gradients = tape.gradient(
    loss,
    backprop_model.trainable_variables
)

print("Loss before weight update:", float(loss))

print("\nGradient information:")

for variable, gradient in zip(
    backprop_model.trainable_variables,
    gradients
):
    print(
        f"{variable.name:30s} "
        f"Shape: {gradient.shape} "
        f"Mean Gradient: {tf.reduce_mean(gradient).numpy():.6f}"
    )

# Update weights using optimizer
optimizer = backprop_model.optimizer

optimizer.apply_gradients(
    zip(
        gradients,
        backprop_model.trainable_variables
    )
)

print("\nBackpropagation completed.")
print("Weights updated using calculated gradients.")


# ============================================================
# 8. Experiment 1
#    Effect of Different Learning Rates
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 1: EFFECT OF LEARNING RATE")
print("=" * 60)

learning_rates = [0.0001, 0.001, 0.01]

lr_results = []

lr_histories = {}

for lr in learning_rates:

    print(f"\nTraining model with learning rate = {lr}")

    model = create_model(lr)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    lr_results.append({
        "Learning Rate": lr,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss
    })

    lr_histories[lr] = history


# -----------------------------
# 9. Learning Rate Results
# -----------------------------
lr_results_df = pd.DataFrame(lr_results)

print("\nLearning Rate Experiment Results:")
print(lr_results_df.to_string(index=False))


# -----------------------------
# 10. Plot Learning Rate
#     Comparison
# -----------------------------
plt.figure(figsize=(9, 5))

for lr, history in lr_histories.items():

    plt.plot(
        history.history["val_accuracy"],
        marker="o",
        label=f"LR = {lr}"
    )

plt.title("Effect of Learning Rate on Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 11. Experiment 2
#     Effect of Number of Epochs
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2: EFFECT OF NUMBER OF EPOCHS")
print("=" * 60)

epoch_values = [2, 5, 10]

epoch_results = []

epoch_histories = {}

fixed_learning_rate = 0.001

for epochs in epoch_values:

    print(
        f"\nTraining model with "
        f"learning rate = {fixed_learning_rate}, "
        f"epochs = {epochs}"
    )

    model = create_model(fixed_learning_rate)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=128,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    epoch_results.append({
        "Epochs": epochs,
        "Learning Rate": fixed_learning_rate,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss
    })

    epoch_histories[epochs] = history


# -----------------------------
# 12. Epoch Results
# -----------------------------
epoch_results_df = pd.DataFrame(epoch_results)

print("\nEpoch Experiment Results:")
print(epoch_results_df.to_string(index=False))


# -----------------------------
# 13. Plot Epoch Comparison
# -----------------------------
plt.figure(figsize=(9, 5))

for epochs, history in epoch_histories.items():

    plt.plot(
        range(1, len(history.history["val_accuracy"]) + 1),
        history.history["val_accuracy"],
        marker="o",
        label=f"{epochs} Epochs"
    )

plt.title("Validation Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 14. Training Accuracy and Loss
#     for Best Configuration
# ============================================================

print("\n" + "=" * 60)
print("BEST MODEL TRAINING")
print("=" * 60)

best_learning_rate = 0.001
best_epochs = 10

best_model = create_model(best_learning_rate)

best_history = best_model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=best_epochs,
    batch_size=128,
    verbose=1
)


# -----------------------------
# 15. Final Evaluation
# -----------------------------
final_loss, final_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Model Performance")
print("-" * 40)
print("Learning Rate :", best_learning_rate)
print("Epochs        :", best_epochs)
print(f"Test Loss     : {final_loss:.4f}")
print(f"Test Accuracy : {final_accuracy * 100:.2f}%")


# -----------------------------
# 16. Plot Training Accuracy
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    best_history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    best_history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 17. Plot Training Loss
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    best_history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    best_history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 18. Make Predictions
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)

predictions = best_model.predict(
    X_test[:10],
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

print("\nPrediction Results:")
print("-" * 40)

for i in range(10):

    print(
        f"Image {i + 1}: "
        f"Actual = {y_test[i]}, "
        f"Predicted = {predicted_labels[i]}"
    )


# -----------------------------
# 19. Visualize Predictions
# -----------------------------
plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i],
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Pred: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.suptitle("MNIST Predictions")
plt.tight_layout()
plt.show()


# ============================================================
# 20. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print("\nLearning Rate Comparison:")
print(lr_results_df.to_string(index=False))

print("\nEpoch Comparison:")
print(epoch_results_df.to_string(index=False))

print("\nFinal Model:")
print(f"Learning Rate : {best_learning_rate}")
print(f"Epochs        : {best_epochs}")
print(f"Test Accuracy : {final_accuracy * 100:.2f}%")
print(f"Test Loss     : {final_loss:.4f}")

print("\n" + "=" * 60)
print("LAB 3 COMPLETED SUCCESSFULLY")
print("=" * 60)