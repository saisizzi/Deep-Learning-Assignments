# ============================================================
# DEEP LEARNING ASSIGNMENT
# MNIST DATASET - TENSORFLOW / KERAS
# ============================================================
# Tasks:
# 1. Install and configure TensorFlow/Keras
# 2. Download MNIST dataset using KaggleHub
# 3. Load the IDX dataset files
# 4. Perform data preprocessing
# 5. Normalize pixel values
# 6. Perform train-validation-test splitting
# 7. Visualize the dataset
# 8. Build and train a basic Keras model
# 9. Evaluate the model
# ============================================================


# ------------------------------------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------------------------------------

!pip install -q tensorflow kagglehub scikit-learn


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import kagglehub

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 3. CHECK TENSORFLOW / KERAS CONFIGURATION
# ------------------------------------------------------------

print("=" * 60)
print("TENSORFLOW / KERAS CONFIGURATION")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)
print("Keras Version      :", keras.__version__)

print("GPU Available      :", tf.config.list_physical_devices("GPU"))

print()


# ------------------------------------------------------------
# 4. DOWNLOAD MNIST DATASET FROM KAGGLE
# ------------------------------------------------------------

print("=" * 60)
print("DOWNLOADING MNIST DATASET")
print("=" * 60)

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Dataset Path:", path)

print("\nFiles available in dataset:")

for file in os.listdir(path):
    print("  -", file)

print()


# ------------------------------------------------------------
# 5. FUNCTIONS TO LOAD IDX FILES
# ------------------------------------------------------------

def load_images(file_path):
    """
    Load MNIST images stored in IDX binary format.
    """

    with open(file_path, "rb") as f:

        # Read IDX header
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        num_columns = int.from_bytes(f.read(4), byteorder="big")

        # Read image data
        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    # Convert 1D data into image format
    images = data.reshape(
        num_images,
        num_rows,
        num_columns
    )

    return images


def load_labels(file_path):
    """
    Load MNIST labels stored in IDX binary format.
    """

    with open(file_path, "rb") as f:

        # Read IDX header
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        # Read label data
        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    return labels


# ------------------------------------------------------------
# 6. DEFINE DATASET FILE PATHS
# ------------------------------------------------------------

train_images_path = os.path.join(
    path,
    "train-images-idx3-ubyte"
)

train_labels_path = os.path.join(
    path,
    "train-labels-idx1-ubyte"
)

test_images_path = os.path.join(
    path,
    "t10k-images-idx3-ubyte"
)

test_labels_path = os.path.join(
    path,
    "t10k-labels-idx1-ubyte"
)


# ------------------------------------------------------------
# 7. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

X_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)

X_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

print("Original Training Images :", X_train.shape)
print("Original Training Labels :", y_train.shape)
print("Original Testing Images  :", X_test.shape)
print("Original Testing Labels  :", y_test.shape)

print()


# ------------------------------------------------------------
# 8. DATASET INFORMATION
# ------------------------------------------------------------

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Number of training images :", len(X_train))
print("Number of testing images  :", len(X_test))
print("Image dimensions          :", X_train.shape[1:])
print("Number of classes         :", len(np.unique(y_train)))
print("Classes                   :", np.unique(y_train))

print()


# ------------------------------------------------------------
# 9. VISUALIZE ORIGINAL DATA
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_train[i],
        cmap="gray"
    )

    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle(
    "Sample MNIST Images",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. CHECK PIXEL VALUES BEFORE NORMALIZATION
# ------------------------------------------------------------

print("=" * 60)
print("PIXEL VALUES BEFORE NORMALIZATION")
print("=" * 60)

print("Minimum pixel value :", X_train.min())
print("Maximum pixel value :", X_train.max())

print()


# ------------------------------------------------------------
# 11. DATA PREPROCESSING
# ------------------------------------------------------------

# Convert uint8 to float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


# ------------------------------------------------------------
# 12. NORMALIZATION
# ------------------------------------------------------------

# Convert pixel values from [0, 255] to [0, 1]

X_train = X_train / 255.0
X_test = X_test / 255.0


# ------------------------------------------------------------
# 13. CHECK PIXEL VALUES AFTER NORMALIZATION
# ------------------------------------------------------------

print("=" * 60)
print("PIXEL VALUES AFTER NORMALIZATION")
print("=" * 60)

print("Minimum pixel value :", X_train.min())
print("Maximum pixel value :", X_train.max())

print()


# ------------------------------------------------------------
# 14. TRAIN-VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# ------------------------------------------------------------
# 15. DISPLAY DATASET SPLIT
# ------------------------------------------------------------

print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print("Training data   :", X_train.shape)
print("Validation data :", X_val.shape)
print("Testing data    :", X_test.shape)

print("Training labels   :", y_train.shape)
print("Validation labels :", y_val.shape)
print("Testing labels    :", y_test.shape)

print()


# ------------------------------------------------------------
# 16. VISUALIZE NORMALIZED DATA
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_train[i],
        cmap="gray"
    )

    plt.title(
        f"Digit: {y_train[i]}"
    )

    plt.axis("off")

plt.suptitle(
    "Normalized MNIST Images",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17. CLASS DISTRIBUTION
# ------------------------------------------------------------

unique, counts = np.unique(
    y_train,
    return_counts=True
)

plt.figure(figsize=(9, 5))

plt.bar(
    unique,
    counts
)

plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.title("MNIST Class Distribution")

plt.xticks(unique)

plt.show()


# ------------------------------------------------------------
# 18. RESHAPE DATA FOR FULLY CONNECTED NEURAL NETWORK
# ------------------------------------------------------------

# Each image:
# 28 x 28 = 784 pixels

X_train_flat = X_train.reshape(
    X_train.shape[0],
    784
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    784
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    784
)


# ------------------------------------------------------------
# 19. DISPLAY FINAL DATA SHAPES
# ------------------------------------------------------------

print("=" * 60)
print("FINAL DATA SHAPES")
print("=" * 60)

print("Training data   :", X_train_flat.shape)
print("Validation data :", X_val_flat.shape)
print("Testing data    :", X_test_flat.shape)

print()


# ------------------------------------------------------------
# 20. BUILD TENSORFLOW / KERAS MODEL
# ------------------------------------------------------------

model = keras.Sequential([

    keras.layers.Input(
        shape=(784,)
    ),

    keras.layers.Dense(
        128,
        activation="relu"
    ),

    keras.layers.Dense(
        64,
        activation="relu"
    ),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ------------------------------------------------------------
# 21. DISPLAY MODEL ARCHITECTURE
# ------------------------------------------------------------

print("=" * 60)
print("KERAS MODEL")
print("=" * 60)

model.summary()


# ------------------------------------------------------------
# 22. COMPILE MODEL
# ------------------------------------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 23. TRAIN MODEL
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

history = model.fit(

    X_train_flat,
    y_train,

    validation_data=(
        X_val_flat,
        y_val
    ),

    epochs=5,

    batch_size=32,

    verbose=1
)


# ------------------------------------------------------------
# 24. EVALUATE MODEL ON TEST DATA
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test_flat,
    y_test,
    verbose=0
)

print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)


# ------------------------------------------------------------
# 25. PLOT TRAINING AND VALIDATION ACCURACY
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
# 26. PLOT TRAINING AND VALIDATION LOSS
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
# 27. MAKE SAMPLE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(
    X_test_flat[:10],
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ------------------------------------------------------------
# 28. DISPLAY PREDICTIONS
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i],
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.suptitle(
    "MNIST Predictions",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 29. FINAL SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("TensorFlow Version :", tf.__version__)
print("Keras Version      :", keras.__version__)

print("\nDataset:")
print("  Training samples   :", X_train_flat.shape[0])
print("  Validation samples :", X_val_flat.shape[0])
print("  Testing samples    :", X_test_flat.shape[0])

print("\nImage:")
print("  Original size      : 28 x 28")
print("  Input features     : 784")
print("  Normalized range   : 0 to 1")

print("\nModel:")
print("  Hidden Layer 1     : 128 neurons - ReLU")
print("  Hidden Layer 2     : 64 neurons - ReLU")
print("  Output Layer       : 10 neurons - Softmax")
print("  Optimizer          : Adam")

print("\nPerformance:")
print("  Test Accuracy      :", round(test_accuracy * 100, 2), "%")

print("\nAssignment completed successfully!")