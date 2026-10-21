# 🧠 Deep Learning Assignment — MNIST Classification

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-MNIST-6C63FF?style=for-the-badge" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

<p align="center">
  <b>MNIST Data Preprocessing, Visualization & Neural Network Classification</b>
</p>

---

## 📌 Objective

The objective of this assignment is to **install and configure TensorFlow/Keras in Google Colab** and perform data preprocessing, normalization, train-test splitting, and visualization using the **MNIST handwritten digit dataset**.

### The assignment covers:

- ⚙️ TensorFlow/Keras installation and configuration
- 📥 Dataset downloading using KaggleHub
- 🔍 Dataset inspection and exploration
- 🖼️ Data visualization
- 🧹 Data preprocessing
- 📏 Pixel-value normalization
- ✂️ Train-validation-test splitting
- 📊 Class distribution analysis
- 🧠 Building a neural network using Keras
- 🏋️ Model training
- 🧪 Model evaluation
- 📈 Training and validation performance visualization
- 🔮 Prediction on unseen images

---

## 📊 Dataset

The **MNIST dataset** contains grayscale images of handwritten digits ranging from **0 to 9**.

| Property | Value |
|---|---:|
| Training Images | 60,000 |
| Test Images | 10,000 |
| Image Size | 28 × 28 pixels |
| Number of Classes | 10 |
| Original Pixel Range | 0–255 |
| Normalized Pixel Range | 0–1 |
| Image Type | Grayscale |

### 📥 Dataset Source

The dataset is downloaded directly from Kaggle using **KaggleHub**:

```python
import kagglehub

path = kagglehub.dataset_download("hojjatk/mnist-dataset")
```

---

## 🛠️ Technologies & Libraries

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,jupyter,git,github" />
</p>

### Technologies

- Python
- TensorFlow
- Keras
- Google Colab
- KaggleHub

### Libraries

- NumPy
- Matplotlib
- Scikit-learn

---

# 🔄 Project Workflow

```text
                    ┌─────────────────────┐
                    │    MNIST Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Download via       │
                    │     KaggleHub       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Load IDX        │
                    │      Files          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Exploration   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Visualization      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Normalization     │
                    │      0 → 1          │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       ┌──────────────┐                  ┌──────────────┐
       │   Training   │                  │    Testing   │
       │   60,000     │                  │    10,000    │
       └──────┬───────┘                  └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ Train / Val  │
       │ 48,000/12,000│
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Reshape      │
       │ 28×28 → 784  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Keras Neural │
       │    Network   │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   Training   │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Evaluation  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Predictions  │
       └──────────────┘
```

---

# 📥 Dataset Loading

The Kaggle MNIST dataset stores images and labels in **IDX binary format**.

Custom functions are used to load:

- Training images
- Training labels
- Testing images
- Testing labels

After loading:

```text
Training Images : (60000, 28, 28)
Training Labels : (60000,)

Testing Images  : (10000, 28, 28)
Testing Labels  : (10000,)
```

Each image contains:

```text
28 × 28 = 784 pixels
```

---

# 🖼️ Data Visualization

Sample MNIST images are visualized using **Matplotlib**.

Visualization is used to verify:

- Images are loaded correctly
- Labels correspond to the correct images
- Images are grayscale
- Digits belong to the expected classes `0–9`

---

# ⚙️ Data Preprocessing

## 1. Convert Data Type

The original MNIST images are stored as unsigned 8-bit integers.

They are converted to `float32` before normalization:

```python
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
```

---

## 2. Normalize Pixel Values

The original pixel values range from:

```text
0 → 255
```

They are normalized to:

```text
0.0 → 1.0
```

using:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### Why Normalize?

Normalization puts input values into a smaller numerical range, which generally helps neural networks train more efficiently and provides better numerical stability during optimization.

```text
Original Pixel       Normalized Pixel

     0       ───────►      0.0
    128      ───────►      0.5
    255      ───────►      1.0
```

---

# ✂️ Train-Validation-Test Split

The original MNIST dataset contains:

```text
60,000 Training Images
10,000 Testing Images
```

The 60,000 training images are further divided into:

```text
Training      → 48,000 images
Validation    → 12,000 images
Testing       → 10,000 images
```

The split is performed using:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)
```

### Why Validation Data?

The validation set is used during training to monitor how well the model performs on data that is not used to update its weights.

---

# 🔢 Input Reshaping

Each MNIST image has dimensions:

```text
28 × 28
```

For a fully connected neural network, the image is flattened:

```text
28 × 28 = 784
```

Therefore:

```text
        28 × 28 Image
              │
              ▼
        Flatten / Reshape
              │
              ▼
       784-Dimensional Vector
```

Final shapes:

```text
Training   : (48000, 784)
Validation : (12000, 784)
Testing    : (10000, 784)
```

---

# 📊 Class Distribution

The MNIST dataset contains ten classes:

```text
0  1  2  3  4  5  6  7  8  9
```

A bar chart is generated to visualize the number of samples belonging to each digit.

This provides a basic understanding of the dataset's class distribution.

---

# 🧠 Neural Network Architecture

A basic fully connected neural network is implemented using **TensorFlow/Keras**.

```text
                  ┌─────────────────┐
                  │   Input Layer   │
                  │   784 Neurons   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Dense Layer    │
                  │  128 Neurons    │
                  │     ReLU        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Dense Layer    │
                  │   64 Neurons    │
                  │     ReLU        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Output Layer   │
                  │   10 Neurons    │
                  │    Softmax      │
                  └────────┬────────┘
                           │
                           ▼
                    Digit Prediction
                       0 ───── 9
```

### Model Architecture

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 784 | — |
| Dense | 128 | ReLU |
| Dense | 64 | ReLU |
| Output | 10 | Softmax |

---

# ⚡ Model Compilation

The model uses the **Adam optimizer**, **Sparse Categorical Crossentropy**, and **Accuracy** metric.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Components

#### Adam Optimizer

An adaptive gradient-based optimization algorithm used to update neural-network parameters during training.

#### Sparse Categorical Crossentropy

Used for multi-class classification when target labels are represented as integer class values.

#### Accuracy

Measures the percentage of correctly classified samples.

---

# 🏋️ Model Training

The model is trained using:

```python
history = model.fit(
    X_train_flat,
    y_train,
    validation_data=(X_val_flat, y_val),
    epochs=5,
    batch_size=32
)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Epochs | 5 |
| Batch Size | 32 |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |

---

# 📈 Performance Visualization

Training performance is visualized using:

### Accuracy Curve

The graph compares:

- Training Accuracy
- Validation Accuracy

### Loss Curve

The graph compares:

- Training Loss
- Validation Loss

These graphs help analyze:

- Model convergence
- Training progress
- Generalization
- Possible overfitting

---

# 🧪 Model Evaluation

After training, the model is evaluated on the separate test dataset:

```python
test_loss, test_accuracy = model.evaluate(
    X_test_flat,
    y_test
)
```

The evaluation returns:

```text
Test Loss
Test Accuracy
```

The test dataset remains separate from training and validation data so that the final evaluation measures performance on unseen samples.

---

# 🔮 Prediction

The trained model is used to classify unseen MNIST images.

```python
predictions = model.predict(X_test_flat[:10])

predicted_labels = np.argmax(
    predictions,
    axis=1
)
```

The predicted labels are compared with the actual labels.

```text
┌──────────────────┐
│   MNIST Image    │
│        7         │
└────────┬─────────┘
         │
         ▼
   Neural Network
         │
         ▼
   Prediction: 7

Actual:    7
Predicted: 7
```

---

# 📁 Repository Structure

```text
Deep-Learning-Assignments/
│
├── Assignment-01/
│   └── assignment_01.ipynb
│
├── README.md
│
└── ...
```

The assignment notebook contains the complete implementation, preprocessing steps, visualizations, model training, evaluation, and prediction results.

---

# 🚀 How to Run

### 1. Open Google Colab

Open `assignment_01.ipynb` in Google Colab.

### 2. Install Dependencies

The notebook installs the required libraries:

```bash
pip install tensorflow kagglehub scikit-learn
```

### 3. Download the Dataset

The dataset is automatically downloaded using:

```python
path = kagglehub.dataset_download(
    "hojjatk/mnist-dataset"
)
```

### 4. Run the Notebook

Execute the notebook cells sequentially.

The complete workflow is:

```text
Dataset Download
       ↓
Data Loading
       ↓
Data Exploration
       ↓
Visualization
       ↓
Preprocessing
       ↓
Normalization
       ↓
Train / Validation / Test Split
       ↓
Reshaping
       ↓
Model Creation
       ↓
Model Training
       ↓
Evaluation
       ↓
Prediction
```

---

# 🎯 Learning Outcomes

After completing this assignment, the following concepts are demonstrated:

- TensorFlow/Keras installation and configuration
- Dataset acquisition using KaggleHub
- Loading binary IDX dataset files
- Understanding image data representation
- Data preprocessing
- Pixel normalization
- Train-validation-test splitting
- Data visualization
- Class distribution analysis
- Image reshaping and flattening
- Building a fully connected neural network
- ReLU activation
- Softmax activation
- Adam optimization
- Sparse categorical crossentropy
- Model training
- Model evaluation
- Training/validation performance analysis
- Prediction on unseen data

---

# 📌 Conclusion

This assignment demonstrates a complete **Deep Learning pipeline using the MNIST dataset**, starting from dataset acquisition and preprocessing and progressing through neural-network training, evaluation, and prediction.

```text
┌────────────────────┐
│   MNIST Dataset    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Data Preprocessing│
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   Normalization    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Data Splitting     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Neural Network     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│     Training       │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│    Evaluation      │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│    Prediction      │
└────────────────────┘
```

> **From raw handwritten digits to a trained neural network — a complete introduction to the fundamental Deep Learning workflow.**

---

<p align="center">
  <b>🧠 Deep Learning • TensorFlow • Keras • MNIST</b>
</p>

<p align="center">
  ⭐ <i>Learn • Implement • Experiment • Analyze</i> ⭐
</p>