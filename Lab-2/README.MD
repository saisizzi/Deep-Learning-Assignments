# 🧠 Deep Learning Lab 2 — Multilayer Perceptron (MLP)

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-MLP-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-red?style=for-the-badge&logo=keras">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Evaluation-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=for-the-badge&logo=googlecolab">
</p>

<p align="center">
  <b>Multilayer Perceptron for Iris Dataset Classification</b>
</p>

---

## 📌 Objective

Design and implement a **Multilayer Perceptron (MLP)** for classification of the **Iris dataset** and evaluate its performance using:

- Accuracy
- Confusion Matrix
- Classification Report
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

---

## 🌸 Dataset — Iris

The **Iris dataset** is a well-known multiclass classification dataset containing measurements of iris flowers from three different species.

### Features

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal |
| Sepal Width | Width of the sepal |
| Petal Length | Length of the petal |
| Petal Width | Width of the petal |

### Target Classes

| Label | Class |
|---:|---|
| 0 | Iris Setosa |
| 1 | Iris Versicolor |
| 2 | Iris Virginica |

### Dataset Information

| Property | Value |
|---|---:|
| Total Samples | 150 |
| Features | 4 |
| Classes | 3 |
| Samples per Class | 50 |
| Training Data | 80% |
| Testing Data | 20% |

---

## 🧠 What is an MLP?

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

The network used in this assignment has the following architecture:

```text
                 INPUT LAYER
                  4 Features
                      │
                      ▼
             ┌─────────────────┐
             │   Dense Layer   │
             │   16 Neurons    │
             │      ReLU       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   Dense Layer   │
             │    8 Neurons    │
             │      ReLU       │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   Output Layer  │
             │    3 Neurons    │
             │     Softmax     │
             └────────┬────────┘
                      │
                      ▼
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Setosa     Versicolor   Virginica
```

### Architecture Summary

| Layer | Neurons | Activation |
|---|---:|---|
| Input Layer | 4 | — |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 3 | Softmax |

---

## 🔄 Complete Workflow

```text
                 Iris Dataset
                      │
                      ▼
               Data Exploration
                      │
                      ▼
               Train-Test Split
                      │
                      ▼
             Feature Standardization
                      │
                      ▼
                  MLP Model
               4 → 16 → 8 → 3
                      │
                      ▼
                 Model Training
                      │
                      ▼
                  Prediction
                      │
             ┌────────┴────────┐
             ▼                 ▼
          Accuracy       Confusion Matrix
             │                 │
             └────────┬────────┘
                      ▼
              Performance Analysis
```

---

## ⚙️ Data Preprocessing

### 1. Train-Test Split

The Iris dataset contains 150 samples.

It is divided into:

```text
                  150 Samples
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       80% Training          20% Testing
        120 Samples            30 Samples
```

The dataset is split using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The `stratify=y` parameter ensures that the class distribution remains approximately the same in both training and testing sets.

---

## 📏 Feature Standardization

Since the four features can have different scales, **StandardScaler** is used.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization transforms the features approximately to:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

The scaler is fitted only on the training data and then applied to the testing data.

This prevents information from the test dataset from influencing the preprocessing stage.

---

## 🔥 Activation Functions

### ReLU

The hidden layers use the **ReLU (Rectified Linear Unit)** activation function.

```text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity into the network and helps the model learn complex relationships between the input features.

### Softmax

The output layer uses the **Softmax** activation function.

Softmax converts the output of the network into class probabilities.

For example:

```text
Setosa       → 0.02
Versicolor   → 0.95
Virginica    → 0.03
```

The class with the highest probability is selected as the final prediction.

---

## ⚡ Model Compilation

The MLP is compiled using the following configuration:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Components

| Component | Description |
|---|---|
| Adam | Optimization algorithm used to update model weights |
| Sparse Categorical Crossentropy | Loss function for multiclass classification with integer labels |
| Accuracy | Measures the percentage of correct predictions |

---

## 🏋️ Model Training

The model is trained using:

```python
history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=8
)
```

### Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Epochs | 50 |
| Batch Size | 8 |
| Validation Split | 20% |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

---

## 📊 Model Evaluation

After training, the model is evaluated on the unseen test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
```

The test accuracy is displayed as:

```text
Test Accuracy: XX.XX%
```

The exact accuracy can vary slightly depending on the execution environment and training process.

---

## 🎯 Accuracy

Accuracy represents the proportion of correctly classified samples.

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)
```

### Formula

```text
                 Correct Predictions
Accuracy = ─────────────────────────────
                 Total Predictions
```

For example, if the model correctly classifies 29 out of 30 test samples:

```text
Accuracy = 29 / 30 × 100
         = 96.67%
```

---

## 🔲 Confusion Matrix

A **confusion matrix** provides a detailed view of the model's classification performance.

For the Iris dataset, the confusion matrix contains three classes:

```text
                         Predicted
                  ┌────────┬────────────┬───────────┐
                  │ Setosa │ Versicolor │ Virginica │
──────────────────┼────────┼────────────┼───────────┤
 Actual Setosa    │   TP   │     FP     │    FP     │
──────────────────┼────────┼────────────┼───────────┤
 Actual Versicolor│   FP   │     TP     │    FP     │
──────────────────┼────────┼────────────┼───────────┤
 Actual Virginica │   FP   │     FP     │    TP     │
──────────────────┴────────┴────────────┴───────────┘
```

The confusion matrix is calculated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

It is visualized using:

```python
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
```

### Interpretation

- **Diagonal values** → Correct predictions
- **Off-diagonal values** → Incorrect predictions
- Higher diagonal values indicate better classification performance.

---

## 📋 Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

```python
classification_report(
    y_test,
    y_pred,
    target_names=class_names
)
```

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| Precision | Proportion of correct predictions among all predictions for a class |
| Recall | Proportion of correctly identified samples among all actual samples of a class |
| F1-score | Harmonic mean of precision and recall |
| Support | Number of actual samples belonging to each class |

---

## 📈 Training vs Validation Accuracy

The model stores its training history in the `history` object.

Training and validation accuracy are plotted using:

```python
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
```

The graph helps determine whether the model is learning effectively.

```text
Accuracy
   │
1.0│                  ─────────
   │              ────
   │          ────
   │      ────
   │  ────
0.0└──────────────────────────────►
                 Epochs
```

---

## 📉 Training vs Validation Loss

Training and validation loss are plotted using:

```python
plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
```

The loss curve helps analyze model convergence and identify potential overfitting.

```text
Loss
 │╲
 │ ╲
 │  ╲
 │   ╲____
 │
 └──────────────────────────────►
                Epochs
```

---

## 🔮 Prediction Process

The trained model generates probabilities for each class:

```python
predictions = model.predict(X_test)
```

The predicted class is obtained using:

```python
y_pred = np.argmax(
    predictions,
    axis=1
)
```

### Prediction Pipeline

```text
Input Features
      │
      ▼
Standardization
      │
      ▼
   MLP Model
      │
      ▼
Softmax Probabilities
      │
      ▼
Highest Probability
      │
      ▼
Predicted Class
```

---

## 🧪 Example Prediction

Suppose the model produces:

```text
Setosa       → 0.01
Versicolor   → 0.97
Virginica    → 0.02
```

The highest probability is for **Versicolor**.

Therefore:

```text
Predicted Class → Iris Versicolor
```

---

## 📊 Visualizations Generated

The program generates several visualizations:

### 1. Iris Dataset Visualization

Displays the distribution of Iris samples based on selected features.

### 2. Confusion Matrix

Shows correct and incorrect predictions for each class.

### 3. Training vs Validation Accuracy

Shows the change in model accuracy over epochs.

### 4. Training vs Validation Loss

Shows the change in model loss over epochs.

### 5. Prediction Probabilities

Displays the probability assigned to each Iris class for selected test samples.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 🧠 TensorFlow | Deep Learning Framework |
| 🔥 Keras | Neural Network API |
| 📊 Scikit-learn | Dataset, preprocessing and evaluation |
| 📈 Matplotlib | Data visualization |
| ☁️ Google Colab | Development Environment |

---

## 📦 Required Libraries

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

The main libraries used are:

```python
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
```

---

## 🚀 How to Run

### Using Google Colab

1. Open the Python program in Google Colab.
2. Install the required dependencies if required.
3. Run the complete program.
4. Observe the Iris dataset visualization.
5. Observe the MLP architecture.
6. Monitor the training process.
7. Check the test accuracy.
8. Analyze the confusion matrix.
9. Review the classification report.
10. Analyze the training and validation curves.

---

## 📁 Repository Structure

```text
Deep-Learning-Assignments/
│
├── Lab-1/
│   ├── Lab1.pdf
│   ├── README.md
│   └── Training.py
│
└── Lab-2/
    ├── Lab2.pdf
    ├── README.md
    └── Training.py
```

---

## 🎯 Learning Outcomes

After completing this assignment, the following concepts are demonstrated:

- Understanding multiclass classification
- Understanding the Iris dataset
- Loading datasets using Scikit-learn
- Performing train-test splitting
- Using stratified sampling
- Feature standardization
- Understanding MLP architecture
- Designing dense neural-network layers
- Using ReLU activation
- Using Softmax activation
- Understanding the Adam optimizer
- Using sparse categorical crossentropy
- Training a neural network using Keras
- Using validation data
- Making predictions using a trained model
- Calculating classification accuracy
- Generating a confusion matrix
- Understanding precision, recall and F1-score
- Visualizing training and validation performance
- Evaluating a neural-network classifier

---

## 📝 Conclusion

In this assignment, a **Multilayer Perceptron (MLP)** was designed and implemented for classifying the Iris dataset.

The model uses the architecture:

```text
4 Input Features
       ↓
16 Neurons + ReLU
       ↓
8 Neurons + ReLU
       ↓
3 Neurons + Softmax
       ↓
Iris Class Prediction
```

The complete machine-learning workflow includes data preprocessing, feature standardization, neural-network training, prediction, accuracy evaluation, confusion-matrix analysis, and visualization of training performance.

The experiment demonstrates how an MLP can be used effectively for a multiclass classification problem.

---

<p align="center">
  <b>🌸 Iris Classification</b>
  <br>
  <b>🧠 Multilayer Perceptron • TensorFlow • Keras</b>
</p>

<p align="center">
  ⭐ <i>Learn • Implement • Train • Evaluate</i> ⭐
</p>