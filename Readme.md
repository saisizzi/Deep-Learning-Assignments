# 🧠 Deep Learning Assignments

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Assignments-6C63FF?style=for-the-badge&logo=pytorch&logoColor=white" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-FF6F00?style=for-the-badge" alt="Machine Learning">
</p>

<p align="center">
  <b>A collection of Deep Learning assignments, implementations, experiments, and practical explorations.</b>
</p>

---

## 📌 About This Repository

This repository contains my **Deep Learning assignments and practical implementations**, covering fundamental concepts as well as neural-network-based approaches to solving machine learning problems.

The primary goal of this repository is to build a strong understanding of:

* 🧠 Artificial Neural Networks
* 🔢 Forward & Backpropagation
* ⚡ Activation Functions
* 📉 Loss Functions & Optimization
* 🏋️ Model Training
* 🧮 Gradient Descent
* 🖼️ Computer Vision
* 🔄 Convolutional Neural Networks
* 🔗 Sequence Models
* 📊 Model Evaluation
* 🛠️ Practical Deep Learning workflows

> **Learning by implementation:** Each assignment focuses on understanding the concept, implementing it, experimenting with it, and analyzing the results.

---

## 🗂️ Repository Overview

```text
Deep-Learning/
│
├── 📁 Assignment-01/
│   ├── README.md
│   └── assignment_01.ipynb
│
├── 📁 Assignment-02/
│   ├── README.md
│   └── assignment_02.ipynb
│
├── 📁 Assignment-03/
│   ├── README.md
│   └── assignment_03.ipynb
│
├── 📁 Assignment-04/
│   ├── README.md
│   └── assignment_04.ipynb
│
├── 📁 Assignment-05/
│   ├── README.md
│   └── assignment_05.ipynb
│
├── 📁 datasets/
│
├── 📁 assets/
│   └── deep-learning-banner.png
│
└── 📄 README.md
```

---

# 🧭 Assignments

|  #  | Assignment                    | Main Concepts                               | Status |
| :-: | :---------------------------- | :------------------------------------------ | :----: |
|  01 | Neural Network Fundamentals   | Neurons, Weights, Bias, Activation          |    ✅   |
|  02 | Forward & Backpropagation     | Gradients, Chain Rule, Backpropagation      |    ✅   |
|  03 | Optimization                  | Gradient Descent, Learning Rate, Optimizers |    ✅   |
|  04 | Convolutional Neural Networks | Convolution, Pooling, Feature Maps          |   🚧   |
|  05 | Model Evaluation              | Metrics, Validation, Overfitting            |   🚧   |

> **Note:** The table can be updated as new assignments are added.

---

# 🧠 Core Concepts Covered

### 1. Neural Networks

Understanding the basic building blocks of a neural network:

```text
Input Layer
    │
    ▼
┌───────────────┐
│ Hidden Layer  │
│ ○  ○  ○  ○    │
└───────────────┘
    │
    ▼
┌───────────────┐
│ Hidden Layer  │
│ ○  ○  ○       │
└───────────────┘
    │
    ▼
┌───────────────┐
│ Output Layer  │
│      ○        │
└───────────────┘
```

Topics include:

* Neurons
* Weights
* Bias
* Linear transformations
* Activation functions
* Forward propagation
* Loss calculation

---

### 2. Activation Functions

Different activation functions are explored and compared based on their mathematical properties and practical applications.

| Function | Typical Use                      |
| -------- | -------------------------------- |
| ReLU     | Hidden layers                    |
| Sigmoid  | Binary classification            |
| Tanh     | Hidden layers / centered outputs |
| Softmax  | Multi-class classification       |

---

### 3. Backpropagation

The repository explores how neural networks learn from their errors.

```text
Prediction
    │
    ▼
Loss
    │
    ▼
Gradient Calculation
    │
    ▼
Backpropagation
    │
    ▼
Weight Update
    │
    └──────────────► Repeat
```

The key idea is to calculate how much each parameter contributed to the error and update the parameters accordingly.

---

### 4. Optimization

Model optimization experiments include concepts such as:

* Gradient Descent
* Learning Rate
* Batch Training
* Mini-batch Training
* Momentum
* Adaptive Optimization

The effect of different hyperparameters on convergence and model performance is also explored.

---

### 5. Convolutional Neural Networks

CNN-based experiments cover the fundamentals of image-based deep learning:

```text
Image
  │
  ▼
Convolution
  │
  ▼
Feature Maps
  │
  ▼
Pooling
  │
  ▼
Flatten
  │
  ▼
Fully Connected Layer
  │
  ▼
Prediction
```

Topics include:

* Kernels / Filters
* Convolution
* Stride
* Padding
* Feature maps
* Pooling
* Flattening
* Classification

---

# 🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,jupyter,pytorch,tensorflow,git,github" alt="Tech Stack">
</p>

### Languages & Tools

* **Python**
* **Jupyter Notebook**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**

### Deep Learning Frameworks

* **PyTorch**
* **TensorFlow / Keras**

### Development & Version Control

* Git
* GitHub
* Jupyter Notebook

---

# 📊 Typical Workflow

Each assignment follows a structured workflow:

```text
┌──────────────────────┐
│  Problem Definition  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Understanding   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Preprocessing   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Model Development    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Training             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Evaluation           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Results & Analysis   │
└──────────────────────┘
```

---

# 📈 What Each Assignment Contains

Whenever applicable, assignments are organized around the following components:

### 🔹 Objective

A clear description of the problem being solved.

### 🔹 Theory

Important concepts and mathematical foundations required to understand the implementation.

### 🔹 Implementation

Python/Jupyter Notebook implementation of the discussed concepts.

### 🔹 Experiments

Experiments with different:

* Hyperparameters
* Architectures
* Optimizers
* Activation functions
* Training configurations

### 🔹 Results

Performance metrics, graphs, predictions, and observations.

### 🔹 Analysis

Interpretation of the results and important conclusions from the experiment.

---

# 📚 Learning Outcomes

After completing these assignments, the key learning outcomes include:

* Understand the fundamental architecture of neural networks.
* Explain how forward propagation works.
* Understand the mathematical intuition behind backpropagation.
* Implement gradient-based optimization.
* Understand the role of activation and loss functions.
* Train and evaluate deep learning models.
* Understand CNN architecture and image feature extraction.
* Analyze model performance using appropriate evaluation metrics.
* Identify common problems such as overfitting and underfitting.
* Experiment with hyperparameters and model architectures.
* Develop practical experience with modern deep learning frameworks.

---

# 🔬 Experiments & Analysis

A major focus of this repository is not only **getting a model to work**, but understanding *why* it works.

Experiments may include comparisons such as:

```text
                 Model Performance
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Learning Rate   Optimizer    Architecture
          │             │             │
          ▼             ▼             ▼
       Accuracy      Loss Curve    Validation
```

Examples of analysis:

* Training vs validation accuracy
* Training vs validation loss
* Effect of learning rate
* Effect of batch size
* Optimizer comparison
* Model architecture comparison
* Detection of overfitting
* Convergence behavior

---

# 📉 Model Evaluation

Depending on the assignment, models may be evaluated using:

| Metric           | Purpose                              |
| ---------------- | ------------------------------------ |
| Accuracy         | Overall classification performance   |
| Precision        | Correctness of positive predictions  |
| Recall           | Ability to identify positive samples |
| F1 Score         | Balance between precision and recall |
| Loss             | Optimization objective               |
| Confusion Matrix | Class-wise prediction analysis       |

Visualizations are used wherever useful to make the model's behavior easier to understand.

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

## 2️⃣ Navigate to the Project

```bash
cd <your-repository>
```

## 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

Install the required deep learning framework depending on the assignment:

```bash
pip install torch torchvision
```

or

```bash
pip install tensorflow
```

## 5️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

Open the required assignment notebook and run the cells sequentially.

---

# 📁 Assignment Structure

Each assignment can follow the following structure:

```text
Assignment-X/
│
├── 📓 assignment_X.ipynb
├── 📄 README.md
├── 📁 data/
├── 📁 results/
└── 📁 models/
```

Where:

* `assignment_X.ipynb` → Complete implementation
* `README.md` → Assignment-specific explanation
* `data/` → Dataset or dataset information
* `results/` → Generated graphs/results
* `models/` → Saved trained models, when applicable

---

# 🧪 Reproducibility

To make experiments reproducible, important configurations such as:

```python
learning_rate = 0.001
batch_size = 32
epochs = 50
```

should be documented inside the corresponding notebook.

Where applicable, random seeds should also be fixed.

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```

---

# 📌 Repository Goals

This repository is intended to serve as:

> **A structured record of my practical journey through Deep Learning — from fundamental neural networks to more advanced architectures and experiments.**

The focus is on combining:

**Theory → Implementation → Experimentation → Evaluation → Understanding**

---

# 🌱 Progress

```text
Deep Learning Journey

Fundamentals        ████████████████████ 100%
Neural Networks     ████████████████████ 100%
Optimization        ███████████████░░░░░  75%
CNNs                ██████████░░░░░░░░░░  50%
Advanced Models     █████░░░░░░░░░░░░░░░  25%
```

> Progress will be updated as new assignments and experiments are completed.

---

# ⭐ Highlights

* 📓 Practical Jupyter Notebook implementations
* 🧠 Strong focus on conceptual understanding
* 📊 Visual analysis of experiments
* 🔬 Hands-on model experimentation
* 📈 Performance evaluation
* 🛠️ Python-based implementations
* 📚 Structured academic documentation

---

# 📖 References & Learning Resources

The implementations and explanations are based on concepts from:

* Deep Learning coursework
* Lecture material
* Official framework documentation
* Research papers and technical references
* Practical experimentation

---

# 👨‍💻 Author

**Sarthak Anil Patkar**

Computer Science / AI & ML Student

<p>
  <img src="https://img.shields.io/badge/Focus-Deep%20Learning-blueviolet?style=flat-square" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Field-Artificial%20Intelligence-blue?style=flat-square" alt="Artificial Intelligence">
</p>

---

<p align="center">
  <i>“The goal is not just to train a model, but to understand how and why it learns.”</i>
</p>

<p align="center">
  ⭐ <b>If you find this repository useful, consider giving it a star!</b> ⭐
</p>
