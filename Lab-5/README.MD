# 🍅 Tomato Leaf Disease Classification

### Deep Learning-Based Plant Disease Detection Using Convolutional Neural Networks

<p align="center">
  <b>An end-to-end deep learning project for classifying tomato leaf diseases from images.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-red?style=for-the-badge" alt="CNN">
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-purple?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## 📌 Overview

**Tomato Leaf Disease Classification** is a computer vision project that uses a **Convolutional Neural Network (CNN)** to identify diseases in tomato leaves from images.

The system learns visual patterns such as:

* Leaf discoloration
* Spots and lesions
* Texture variations
* Fungal or bacterial symptoms
* Structural abnormalities

The trained model classifies an input image into one of the disease categories present in the dataset.

> **Project Goal:** Demonstrate how deep learning and image classification can be applied to agricultural disease detection.

---

## 🎯 Problem Statement

Tomato crops are vulnerable to several diseases that can significantly affect crop yield and quality. Manual disease identification requires agricultural expertise and can be time-consuming.

This project explores an automated image-based approach:

> **Given an image of a tomato leaf, predict its disease category using a trained CNN model.**

### Input

A tomato leaf image.

### Output

* Predicted disease class
* Class probability / confidence score

---

## ✨ Key Features

* 🖼️ Image-based tomato leaf classification
* 🧠 Custom CNN architecture built using TensorFlow/Keras
* 🔄 Data augmentation for improved generalization
* 📏 Image resizing and pixel normalization
* 📊 Training and validation accuracy tracking
* 📉 Training and validation loss visualization
* 📋 Classification report with precision, recall, and F1-score
* 🔥 Confusion matrix for class-wise evaluation
* 💾 Trained model saved in `.keras` format
* 🔍 Single-image prediction support

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────────┐
                    │   Tomato Leaf Dataset    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Image Loading &         │
                    │  Preprocessing            │
                    │  Resize: 128 × 128       │
                    │  Normalize: 0–1          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Data Augmentation      │
                    │ Flip / Rotation / Zoom   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      CNN Feature         │
                    │       Extraction         │
                    │                          │
                    │ Conv2D → Pooling         │
                    │ Conv2D → Pooling         │
                    │ Conv2D → Pooling         │
                    │ Conv2D → Pooling         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Fully Connected Layer  │
                    │       Dense (128)         │
                    │       Dropout (0.5)       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Softmax Classification │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Predicted Disease Class  │
                    │ + Confidence             │
                    └──────────────────────────┘
```

---

## 🧠 Model Architecture

The project uses a **custom sequential CNN** designed to progressively learn low-level and high-level visual features.

| Layer        | Configuration      | Purpose                          |
| ------------ | ------------------ | -------------------------------- |
| Input        | 128 × 128 × 3      | Accept RGB leaf images           |
| Rescaling    | 1/255              | Normalize pixel values           |
| Conv2D       | 32 filters, 3 × 3  | Learn basic visual features      |
| MaxPooling2D | 2 × 2              | Reduce spatial dimensions        |
| Conv2D       | 64 filters, 3 × 3  | Learn intermediate features      |
| MaxPooling2D | 2 × 2              | Downsample feature maps          |
| Conv2D       | 128 filters, 3 × 3 | Learn complex patterns           |
| MaxPooling2D | 2 × 2              | Reduce dimensionality            |
| Conv2D       | 256 filters, 3 × 3 | Learn high-level representations |
| MaxPooling2D | 2 × 2              | Extract dominant features        |
| Flatten      | —                  | Convert feature maps to a vector |
| Dense        | 128 neurons, ReLU  | Classification representation    |
| Dropout      | 0.5                | Reduce overfitting               |
| Output       | Softmax            | Predict class probabilities      |

### Why CNN?

CNNs are effective for image classification because they automatically learn spatial hierarchies:

```text
Edges → Textures → Shapes → Leaf Patterns → Disease Features
```

Unlike traditional machine learning approaches that require manually engineered image features, CNNs learn useful representations directly from training images.

---

## 📂 Dataset

The project uses the **Tomato Leaf Disease dataset** available through Kaggle.

**Dataset source:** [Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)

### Dataset Structure

```text
tomato/
├── train/
│   ├── Class_1/
│   ├── Class_2/
│   ├── Class_3/
│   └── ...
│
└── val/
    ├── Class_1/
    ├── Class_2/
    ├── Class_3/
    └── ...
```

> The exact class names are automatically extracted from the dataset directory structure during execution.

### Dataset Loading

The dataset is loaded using:

```python
tf.keras.utils.image_dataset_from_directory()
```

This automatically:

* Reads images from class-specific folders
* Assigns integer labels
* Resizes images
* Creates TensorFlow datasets

---

## 🔄 Data Preprocessing

Before training, the images undergo the following preprocessing steps:

### 1. Image Resizing

All images are resized to:

```text
128 × 128 × 3
```

This ensures consistent input dimensions.

### 2. Pixel Normalization

Pixel values are scaled:

```text
Original range: 0–255
Normalized range: 0–1
```

Formula:

$$
x_{\text{normalized}} = \frac{x}{255}
$$

### 3. Data Augmentation

The model applies random transformations to training images:

* Horizontal flipping
* Random rotation
* Random zoom

Data augmentation helps the model learn robust patterns and reduces dependence on the exact orientation or position of training images.

---

## ⚙️ Training Configuration

| Parameter               | Value                           |
| ----------------------- | ------------------------------- |
| Framework               | TensorFlow / Keras              |
| Model                   | Custom CNN                      |
| Image Size              | 128 × 128                       |
| Batch Size              | 32                              |
| Training Epochs         | 12                              |
| Optimizer               | Adam                            |
| Initial Learning Rate   | 0.001                           |
| Loss Function           | Sparse Categorical Crossentropy |
| Evaluation Metric       | Accuracy                        |
| Dropout                 | 0.5                             |
| Learning Rate Scheduler | ReduceLROnPlateau               |

### Optimizer

The **Adam optimizer** is used to update model weights efficiently during training.

### Loss Function

Since the model performs multi-class classification with integer-encoded labels, it uses:

```python
sparse_categorical_crossentropy
```

### Learning Rate Scheduling

`ReduceLROnPlateau` reduces the learning rate when validation loss stops improving, helping the model continue learning more effectively.

---

## 📈 Evaluation Metrics

The model is evaluated using multiple metrics rather than accuracy alone.

### Accuracy

Measures the proportion of correctly classified images.

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

### Precision

Measures how many predicted instances of a class are actually correct.

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

Measures how many actual instances of a class are correctly identified.

$$
Recall = \frac{TP}{TP + FN}
$$

### F1-Score

The harmonic mean of precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### Confusion Matrix

A confusion matrix shows how predictions are distributed across actual and predicted classes, helping identify which diseases the model confuses.

---

## 📊 Results and Visualizations

The notebook generates the following evaluation outputs:

### 1. Training vs Validation Accuracy

Used to analyze:

* Learning progress
* Generalization
* Potential overfitting

### 2. Training vs Validation Loss

Used to understand:

* Optimization progress
* Convergence
* Divergence between training and validation performance

### 3. Classification Report

Provides class-wise:

* Precision
* Recall
* F1-score
* Support

### 4. Confusion Matrix

Visualizes correct and incorrect predictions across all disease classes.

> **Note:** Add your final validation accuracy, classification report, and confusion matrix screenshots here after running the notebook.

Example:

```markdown
## 📸 Results

![Training Accuracy](assets/training_accuracy.png)

![Training Loss](assets/training_loss.png)

![Confusion Matrix](assets/confusion_matrix.png)
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/tomato-leaf-disease-classification.git
```

### 2. Navigate to the Project

```bash
cd tomato-leaf-disease-classification
```

### 3. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```text
tensorflow
numpy
matplotlib
seaborn
scikit-learn
kagglehub
pillow
```

> TensorFlow installation may vary depending on your operating system and whether you use CPU or GPU acceleration.

---

## ▶️ Usage

### Option 1: Run the Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

### Option 2: Run as a Python Script

If the notebook has been converted into a Python script:

```bash
python train.py
```

The pipeline will:

1. Download/load the dataset
2. Preprocess images
3. Build the CNN
4. Train the model
5. Evaluate performance
6. Save the trained model

---

## 🔍 Single-Image Prediction

After training, the saved model can be used to classify a new tomato leaf image.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model(
    "tomato_disease_cnn.keras"
)

# Path to input image
image_path = "sample_leaf.jpg"

# Load and resize image
img = load_img(
    image_path,
    target_size=(128, 128)
)

# Convert image to array
img_array = img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(
    img_array,
    axis=0
)

# Predict
predictions = model.predict(img_array)

# Get predicted class
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

# Get confidence
confidence = float(
    np.max(predictions[0]) * 100
)

print("Predicted Disease:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
```

> **Important:** The `class_names` list must match the class-index order used during training. Save or reconstruct it consistently when performing inference.

---

## 📁 Recommended Project Structure

```text
tomato-leaf-disease-classification/
│
├── 📓 Tomato_Leaf_Disease_Classification.ipynb
│
├── 📁 assets/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── 📁 models/
│   └── tomato_disease_cnn.keras
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🧪 Project Workflow

```text
Dataset Acquisition
       ↓
Image Preprocessing
       ↓
Data Augmentation
       ↓
CNN Model Construction
       ↓
Model Compilation
       ↓
Model Training
       ↓
Validation
       ↓
Performance Evaluation
       ↓
Model Saving
       ↓
New Image Prediction
```

---

## ⚠️ Limitations

* Performance depends on the quality and diversity of the training dataset.
* Images captured under different lighting or backgrounds may affect predictions.
* A custom CNN may not generalize as well as a well-tuned transfer-learning model.
* The model is intended as an educational and research prototype, not a substitute for professional agricultural diagnosis.
* Confidence scores represent model probabilities and should not automatically be interpreted as certainty.

---

## 🔮 Future Improvements

* [ ] Add **transfer learning** using MobileNetV2, EfficientNet, or ResNet50
* [ ] Compare custom CNN performance with pretrained architectures
* [ ] Add Grad-CAM visualizations for model interpretability
* [ ] Build a Streamlit web application
* [ ] Add real-time camera-based leaf detection
* [ ] Include treatment and prevention recommendations from verified agricultural sources
* [ ] Deploy the model using an API backend
* [ ] Add model versioning and experiment tracking
* [ ] Evaluate performance on real-world field images
* [ ] Add confidence-based “uncertain prediction” handling

---

## 🛠️ Technologies Used

| Technology           | Purpose                         |
| -------------------- | ------------------------------- |
| **Python**           | Programming language            |
| **TensorFlow**       | Deep learning framework         |
| **Keras**            | CNN model development           |
| **NumPy**            | Numerical computation           |
| **Matplotlib**       | Training visualization          |
| **Seaborn**          | Confusion matrix visualization  |
| **Scikit-learn**     | Model evaluation                |
| **KaggleHub**        | Dataset download                |
| **Jupyter Notebook** | Experimentation and development |

---

## 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Fundamentals of convolutional neural networks
* Image preprocessing and normalization
* Data augmentation
* Multi-class image classification
* Model training and validation
* Overfitting control using dropout
* Learning rate scheduling
* Classification metrics
* Confusion matrix analysis
* Model serialization and inference

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/improvement
```

3. Commit your changes.

```bash
git commit -m "Add model improvement"
```

4. Push the branch.

```bash
git push origin feature/improvement
```

5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

You may use, modify, and distribute the code according to the terms of the license.

---

## 👨‍💻 Author

**Sarthak Anil Patkar**

---

<p align="center">
  <b>🍃 Exploring the intersection of Deep Learning and Smart Agriculture.</b>
</p>
