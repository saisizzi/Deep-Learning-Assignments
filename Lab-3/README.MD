# 🧠 Deep Learning Lab 3 — Forward & Backpropagation

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Backpropagation-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-red?style=for-the-badge&logo=keras">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/MNIST-Dataset-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=for-the-badge&logo=googlecolab">
</p>

<p align="center">
  <b>Implementation of Forward Propagation and Backpropagation using TensorFlow/Keras</b>
</p>

---

## 📌 Objective

Implement **forward propagation** and **backpropagation** using **TensorFlow/Keras** and analyze the effect of:

- Different learning rates
- Different numbers of epochs
- Model training performance
- Validation accuracy
- Test accuracy
- Training and validation loss

The **MNIST handwritten digit dataset** is used for multiclass image classification.

---

## 🌟 Overview

This experiment demonstrates the complete training process of a neural network:

```text
                    MNIST Dataset
                         │
                         ▼
                  Data Preprocessing
                         │
                         ▼
                  Forward Propagation
                         │
                         ▼
                    Prediction
                         │
                         ▼
                      Loss
                         │
                         ▼
                  Backpropagation
                         │
                         ▼
                    Gradients
                         │
                         ▼
                  Weight Updates
                         │
                         ▼
                 Repeat for Epochs
                         │
                         ▼
                   Final Model
```

---

## 📊 Dataset — MNIST

The **MNIST dataset** contains handwritten digits from **0 to 9**.

Each image is a grayscale image of size:

```text
28 × 28 pixels
```

### Dataset Information

| Property | Value |
|---|---:|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 |
| Number of Classes | 10 |
| Classes | 0–9 |
| Pixel Range | 0–255 |
| Normalized Range | 0–1 |

---

## 🖼️ Sample MNIST Images

The program displays sample handwritten digits before training.

```text
┌───────┐  ┌───────┐  ┌───────┐
│   0   │  │   1   │  │   2   │
│       │  │       │  │       │
└───────┘  └───────┘  └───────┘

┌───────┐  ┌───────┐  ┌───────┐
│   3   │  │   4   │  │   5   │
│       │  │       │  │       │
└───────┘  └───────┘  └───────┘
```

---

# 🧹 Data Preprocessing

The MNIST images contain pixel values between:

```text
0 → 255
```

These values are normalized to:

```text
0 → 1
```

using:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

### Why Normalize?

Normalization makes the input values smaller and helps the neural network train more efficiently.

---

# 🧠 Neural Network Architecture

The MLP used in this experiment has the following architecture:

```text
              MNIST IMAGE
                28 × 28
                   │
                   ▼
             ┌───────────┐
             │  Flatten  │
             │   784     │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │   Dense   │
             │ 128 Nodes │
             │   ReLU    │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │   Dense   │
             │  64 Nodes │
             │   ReLU    │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │   Dense   │
             │  10 Nodes │
             │  Softmax  │
             └─────┬─────┘
                   │
                   ▼
          Digit: 0 1 2 ... 9
```

### Architecture Summary

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 784 | — |
| Hidden Layer 1 | 128 | ReLU |
| Hidden Layer 2 | 64 | ReLU |
| Output Layer | 10 | Softmax |

---

# ➡️ Forward Propagation

Forward propagation is the process of passing the input through the neural network to obtain a prediction.

The process is:

```text
Input
  │
  ▼
Weighted Sum
  │
  ▼
Activation Function
  │
  ▼
Next Layer
  │
  ▼
Output Probabilities
```

For a neuron, the basic operation is:

```text
z = Wx + b
```

where:

- `W` = weights
- `x` = input
- `b` = bias
- `z` = weighted sum

The activation function is then applied:

```text
a = activation(z)
```

---

## 🔥 ReLU Activation

The hidden layers use the ReLU activation function:

```text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity and allows the network to learn complex patterns.

---

## 🎯 Softmax Activation

The output layer contains 10 neurons because MNIST has 10 classes.

Softmax converts the output values into probabilities.

Example:

```text
Digit 0 → 0.01
Digit 1 → 0.02
Digit 2 → 0.01
Digit 3 → 0.03
Digit 4 → 0.01
Digit 5 → 0.04
Digit 6 → 0.02
Digit 7 → 0.03
Digit 8 → 0.01
Digit 9 → 0.82
```

The predicted digit is:

```text
9
```

because it has the highest probability.

---

# ❌ Loss Calculation

The model uses:

```python
sparse_categorical_crossentropy
```

as the loss function.

The loss measures how different the model's prediction is from the actual class.

```text
Correct Prediction
       ↓
Lower Loss

Incorrect Prediction
       ↓
Higher Loss
```

The objective of training is to minimize the loss.

---

# 🔄 Backpropagation

Backpropagation is the process of calculating how much each weight contributed to the error.

The process is:

```text
              Prediction
                   │
                   ▼
                  Loss
                   │
                   ▼
            Backpropagation
                   │
                   ▼
              Gradients
                   │
                   ▼
             Weight Update
```

TensorFlow's `GradientTape` is used to explicitly demonstrate this process.

```python
with tf.GradientTape() as tape:

    predictions = model(x_batch, training=True)

    loss = loss_function(
        y_batch,
        predictions
    )

gradients = tape.gradient(
    loss,
    model.trainable_variables
)
```

The gradients are then used to update the weights:

```python
optimizer.apply_gradients(
    zip(
        gradients,
        model.trainable_variables
    )
)
```

---

# 📐 Gradient Descent

The basic weight update rule is:

```text
New Weight = Old Weight − Learning Rate × Gradient
```

or:

```text
Wnew = Wold − η × ∂L/∂W
```

where:

- `W` = weight
- `η` = learning rate
- `L` = loss
- `∂L/∂W` = gradient

---

# ⚡ Learning Rate

The **learning rate** determines how large each weight update is.

```text
Small Learning Rate
        │
        ▼
Small Weight Updates
        │
        ▼
Slower Learning

Large Learning Rate
        │
        ▼
Large Weight Updates
        │
        ▼
Faster but potentially unstable Learning
```

The experiment compares:

```text
0.0001
0.001
0.01
```

---

# 🧪 Experiment 1 — Effect of Learning Rate

Three different learning rates are tested:

| Learning Rate | Expected Behaviour |
|---:|---|
| 0.0001 | Slow learning |
| 0.001 | Stable and effective learning |
| 0.01 | Faster learning but may become unstable |

Each model is trained for:

```text
5 Epochs
```

The test accuracy and loss are recorded for each learning rate.

### Comparison

The program generates a graph showing:

```text
Validation Accuracy
        │
        │       ╭────────
        │    ╭──╯
        │ ╭──╯
        │╭╯
        └──────────────────►
               Epochs

      Different Learning Rates
```

---

# 🔁 Experiment 2 — Effect of Number of Epochs

The number of epochs determines how many times the model sees the training dataset.

The experiment compares:

```text
2 Epochs
5 Epochs
10 Epochs
```

The learning rate is fixed at:

```text
0.001
```

### Expected Behaviour

```text
More Epochs
     │
     ▼
More Training
     │
     ▼
Usually Better Learning
     │
     ▼
Higher Accuracy
```

However, too many epochs can eventually lead to **overfitting**.

---

# 📈 Training and Validation Accuracy

The training history is used to plot accuracy:

```python
plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)
```

The graph helps determine whether the model is learning properly.

### Interpretation

```text
Training Accuracy ↑
Validation Accuracy ↑
        │
        ▼
Good Learning

Training Accuracy ↑
Validation Accuracy ↓
        │
        ▼
Possible Overfitting
```

---

# 📉 Training and Validation Loss

Training and validation loss are also plotted.

```python
plt.plot(
    history.history["loss"]
)

plt.plot(
    history.history["val_loss"]
)
```

A decreasing loss generally indicates that the model is improving.

---

# 📊 Final Model

After the experiments, a final model is trained using:

| Parameter | Value |
|---|---:|
| Learning Rate | 0.001 |
| Epochs | 10 |
| Batch Size | 128 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |

The model is then evaluated on the **10,000 unseen MNIST test images**.

---

# 🔮 Prediction

The trained model predicts the digit using:

```python
predictions = model.predict(X_test)
```

The class with the highest probability is selected:

```python
predicted_labels = np.argmax(
    predictions,
    axis=1
)
```

Example:

```text
Actual Digit    : 7
Predicted Digit : 7
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 🧠 TensorFlow | Deep Learning Framework |
| 🔥 Keras | Neural Network API |
| 📊 NumPy | Numerical Computation |
| 📈 Matplotlib | Visualization |
| 📋 Pandas | Experiment Results |
| ☁️ Google Colab | Development Environment |

---

# 📦 Installation

Install the required libraries using:

```bash
pip install tensorflow numpy pandas matplotlib
```

Google Colab generally provides TensorFlow and many required packages by default.

---

# 🚀 How to Run

### Google Colab

1. Open `Training.py` in Google Colab.
2. Run the complete program.
3. The MNIST dataset will be downloaded automatically.
4. Sample images will be displayed.
5. Forward propagation will be demonstrated.
6. Backpropagation will be demonstrated using `GradientTape`.
7. Different learning rates will be tested.
8. Different numbers of epochs will be tested.
9. Accuracy and loss graphs will be generated.
10. The final model will make predictions on test images.

---

# 📁 Repository Structure

```text
Deep-Learning-Assignments/
│
├── Lab-1/
│   ├── Lab1.pdf
│   ├── README.md
│   └── Training.py
│
├── Lab-2/
│   ├── Lab2.pdf
│   ├── README.md
│   └── Training.py
│
└── Lab-3/
    ├── Lab3.pdf
    ├── README.md
    └── Training.py
```

---

# 🎯 Learning Outcomes

After completing this assignment, the following concepts are demonstrated:

- Understanding forward propagation
- Understanding backpropagation
- Understanding neural-network weights and biases
- Understanding activation functions
- Understanding ReLU
- Understanding Softmax
- Understanding loss functions
- Understanding gradients
- Understanding gradient descent
- Using TensorFlow `GradientTape`
- Updating neural-network weights
- Understanding learning rate
- Understanding epochs
- Comparing different learning rates
- Comparing different numbers of epochs
- Evaluating model performance
- Visualizing training accuracy
- Visualizing validation accuracy
- Visualizing training loss
- Visualizing validation loss
- Making predictions using a trained neural network

---

# 📝 Conclusion

This experiment demonstrates the complete learning mechanism of a neural network using **TensorFlow/Keras**.

The training process can be summarized as:

```text
              Input Data
                  │
                  ▼
        Forward Propagation
                  │
                  ▼
             Prediction
                  │
                  ▼
             Loss/Error
                  │
                  ▼
         Backpropagation
                  │
                  ▼
              Gradients
                  │
                  ▼
          Weight Updates
                  │
                  ▼
            Next Epoch
                  │
                  ▼
          Trained Network
```

The experiments demonstrate that the **learning rate** controls the size of weight updates, while the **number of epochs** controls how many times the model learns from the training data.

A suitable learning rate and sufficient number of epochs are important for achieving good model performance without causing unstable training or overfitting.

---

<p align="center">
  <b>🧠 Forward Propagation • 🔄 Backpropagation • ⚡ Learning Rate • 📈 Epochs</b>
</p>

<p align="center">
  ⭐ <i>Learn • Implement • Experiment • Evaluate</i> ⭐
</p>