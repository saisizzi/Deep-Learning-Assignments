# 🐱🐶 Cats vs Dogs Image Classification

### A Comparative Study of AlexNet, VGG16, ResNet50, and EfficientNetB0 Using Transfer Learning

<p align="center">
  <b>Comparing classical and modern CNN architectures for binary image classification.</b>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Sanskar-cpu-sudo/Deep-Learning/blob/main/lab7.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Transfer%20Learning-Enabled-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-green?style=for-the-badge">
</p>

---

## 📌 Overview

This project explores **binary image classification** using deep learning by training and comparing four convolutional neural network architectures:

* **AlexNet**
* **VGG16**
* **ResNet50**
* **EfficientNetB0**

The objective is to classify images into two categories:

> **Cat 🐱 or Dog 🐶**

The project demonstrates how different CNN architectures perform on the same dataset and compares them using **test accuracy, test loss, and training time**.

It also introduces the practical use of **transfer learning**, where pretrained convolutional models are adapted to a new classification task.

---

## 🎯 Objectives

* Build an image classification pipeline using TensorFlow/Keras.
* Prepare and load a Cats vs Dogs dataset.
* Apply image resizing and data augmentation.
* Implement transfer learning using pretrained CNN architectures.
* Construct an AlexNet-inspired CNN from scratch.
* Train each model under a consistent experimental setup.
* Evaluate model performance on a separate test dataset.
* Compare accuracy, loss, and training time.
* Visualize the comparative performance of the models.

---

## 🧠 What Is Transfer Learning?

**Transfer learning** is a deep learning technique in which a model trained on a large dataset is reused for a different but related task.

Instead of learning every visual feature from the beginning, a pretrained model can provide useful representations such as:

```text
Edges → Textures → Shapes → Object Features → Classification
```

In this project:

1. Pretrained CNN backbones are loaded.
2. Their learned visual representations are reused.
3. A new classification head is added.
4. The model is trained for cat-versus-dog classification.

### Models Used

| Model              | Approach                                                       |
| ------------------ | -------------------------------------------------------------- |
| **VGG16**          | Pretrained CNN backbone + custom classification head           |
| **ResNet50**       | Pretrained residual CNN backbone + custom classification head  |
| **EfficientNetB0** | Pretrained efficient CNN backbone + custom classification head |
| **AlexNet**        | Custom CNN architecture implemented using Keras                |

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────────┐
                    │   Cats vs Dogs Dataset   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Dataset Preparation    │
                    │                          │
                    │ Resize: 224 × 224        │
                    │ Binary Labels            │
                    │ Train / Validation / Test│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Data Augmentation      │
                    │ Flip / Rotation / Zoom   │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
                 ▼               ▼                ▼
          ┌────────────┐  ┌────────────┐  ┌──────────────┐
          │   VGG16    │  │  ResNet50  │  │ EfficientNet │
          │ Pretrained │  │ Pretrained │  │     B0       │
          └──────┬─────┘  └──────┬─────┘  └──────┬───────┘
                 │               │                │
                 └───────────────┼────────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
                 ▼                                ▼
          ┌────────────┐                 ┌──────────────┐
          │  AlexNet   │                 │ Custom CNN   │
          │ From Scratch│                │ Architecture │
          └──────┬─────┘                 └──────┬───────┘
                 │                                │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Global Average Pooling│
                    │   / Flatten              │
                    │   Dense Classification  │
                    │   Sigmoid Output        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Cat or Dog Prediction   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Accuracy / Loss / Time   │
                    │ Model Comparison         │
                    └──────────────────────────┘
```

---

## 📂 Dataset

The project uses the **Cat and Dog dataset** downloaded from Kaggle.

**Dataset:** [Cat and Dog Dataset — Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

### Dataset Download

The notebook downloads the dataset using the Kaggle CLI:

```bash
kaggle datasets download -d tongpython/cat-and-dog
```

The downloaded ZIP file is extracted into the `cats_dogs` directory.

### Dataset Structure

```text
cats_dogs/
│
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
│
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/
```

### Dataset Split

The notebook creates:

| Split          | Source                    | Purpose                |
| -------------- | ------------------------- | ---------------------- |
| Training Set   | 80% of training directory | Model training         |
| Validation Set | 20% of training directory | Monitor generalization |
| Test Set       | Separate test directory   | Final evaluation       |

The training-validation split uses:

```python
validation_split=0.2
```

---

## 🔄 Data Preprocessing

### Image Resizing

All images are resized to:

```text
224 × 224 × 3
```

This standardizes the input dimensions for all four models.

### Label Encoding

The task is binary classification:

```text
Cat → 0
Dog → 1
```

The notebook uses:

```python
label_mode="binary"
```

### Data Augmentation

The models use random image transformations during training:

* Horizontal flipping
* Random rotation
* Random zoom

Example:

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.1)
layers.RandomZoom(0.1)
```

Data augmentation helps expose the model to different image orientations and scales.

---

## 🧠 Model Architectures

### 1. VGG16

**VGG16** is a deep CNN architecture that uses a sequence of small convolutional filters.

In this project, the pretrained VGG16 backbone is followed by:

```text
VGG16 Backbone
      ↓
Global Average Pooling
      ↓
Dropout (0.3)
      ↓
Dense (1, Sigmoid)
```

The model uses VGG16 preprocessing before feature extraction.

---

### 2. ResNet50

**ResNet50** uses residual connections to improve the training of deep neural networks.

Residual connections help information and gradients flow through the network more effectively.

Architecture used:

```text
ResNet50 Backbone
       ↓
Global Average Pooling
       ↓
Dropout (0.3)
       ↓
Dense (1, Sigmoid)
```

---

### 3. EfficientNetB0

**EfficientNetB0** is designed to achieve a balance between model accuracy and computational efficiency.

Architecture used:

```text
EfficientNetB0 Backbone
         ↓
Global Average Pooling
         ↓
Dropout (0.3)
         ↓
Dense (1, Sigmoid)
```

EfficientNetB0 is included to compare an efficient modern architecture against VGG16, ResNet50, and AlexNet.

---

### 4. AlexNet

Unlike the transfer-learning models, AlexNet is implemented as a custom CNN.

#### Architecture

```text
Input: 224 × 224 × 3
        ↓
Conv2D: 96 filters, 11 × 11, stride 4
        ↓
MaxPooling
        ↓
Conv2D: 256 filters, 5 × 5
        ↓
MaxPooling
        ↓
Conv2D: 384 filters, 3 × 3
        ↓
Conv2D: 384 filters, 3 × 3
        ↓
Conv2D: 256 filters, 3 × 3
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense: 4096
        ↓
Dropout: 0.5
        ↓
Dense: 4096
        ↓
Dropout: 0.5
        ↓
Dense: 1, Sigmoid
```

### AlexNet Configuration

| Component              | Configuration                 |
| ---------------------- | ----------------------------- |
| Input Size             | 224 × 224 × 3                 |
| First Convolution      | 96 filters, 11 × 11, stride 4 |
| Second Convolution     | 256 filters, 5 × 5            |
| Middle Convolutions    | 384, 384, 256 filters         |
| Fully Connected Layers | 4096, 4096                    |
| Dropout                | 0.5                           |
| Output Activation      | Sigmoid                       |

---

## ⚙️ Training Configuration

Each model is trained using the same training and validation datasets.

| Parameter     | VGG16               | ResNet50            | EfficientNetB0      | AlexNet             |
| ------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Input Size    | 224 × 224           | 224 × 224           | 224 × 224           | 224 × 224           |
| Batch Size    | 32                  | 32                  | 32                  | 32                  |
| Epochs        | 3                   | 3                   | 3                   | 3                   |
| Optimizer     | Adam                | Adam                | Adam                | Adam                |
| Loss          | Binary Crossentropy | Binary Crossentropy | Binary Crossentropy | Binary Crossentropy |
| Output        | Sigmoid             | Sigmoid             | Sigmoid             | Sigmoid             |
| Learning Rate | Default Adam        | Default Adam        | Default Adam        | 0.0001              |

### Compilation

The transfer-learning models use:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

AlexNet uses an Adam optimizer with a learning rate of `0.0001`.

---

## 📊 Evaluation Methodology

Each model is evaluated on the separate test dataset.

The following metrics are recorded:

### Accuracy

Measures the proportion of correctly classified images.

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

### Binary Crossentropy Loss

Measures the difference between predicted probabilities and actual binary labels.

### Training Time

The notebook records the time required to train each model:

```python
start = time.time()

model.fit(...)

training_time = time.time() - start
```

This enables comparison of computational cost alongside predictive performance.

---

## 📈 Model Comparison

The notebook stores the evaluation results in a Pandas DataFrame.

```python
results = pd.DataFrame({
    "Model": [
        "AlexNet",
        "VGG16",
        "ResNet50",
        "EfficientNetB0"
    ],
    "Accuracy": [
        alexnet_accuracy,
        vgg_accuracy,
        resnet_accuracy,
        efficientnet_accuracy
    ],
    "Loss": [
        alexnet_loss,
        vgg_loss,
        resnet_loss,
        efficientnet_loss
    ],
    "Training Time (seconds)": [
        alexnet_time,
        vgg_time,
        resnet_time,
        efficientnet_time
    ]
})
```

### Comparison Table

| Model          |             Test Accuracy |                 Test Loss |             Training Time |
| -------------- | ------------------------: | ------------------------: | ------------------------: |
| AlexNet        | Generated after execution | Generated after execution | Generated after execution |
| VGG16          | Generated after execution | Generated after execution | Generated after execution |
| ResNet50       | Generated after execution | Generated after execution | Generated after execution |
| EfficientNetB0 | Generated after execution | Generated after execution | Generated after execution |

> **Note:** The uploaded notebook contains the code for generating these results, but the available notebook source does not include executed output values. The numerical results should be filled in after running the notebook.

---

## 📊 Visualization

The project generates a bar chart comparing test accuracy across all models.

```python
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Accuracy Comparison of CNN Models")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)

plt.show()
```

### Suggested Results Section

After running the notebook, add the generated chart here:

```markdown
## 📸 Results

![Model Accuracy Comparison](assets/model_accuracy_comparison.png)
```

---

## 🛠️ Technologies Used

| Technology                 | Purpose                           |
| -------------------------- | --------------------------------- |
| **Python**                 | Programming language              |
| **TensorFlow**             | Deep learning framework           |
| **Keras**                  | Model development                 |
| **VGG16**                  | Transfer-learning architecture    |
| **ResNet50**               | Residual CNN architecture         |
| **EfficientNetB0**         | Efficient pretrained architecture |
| **NumPy**                  | Numerical operations              |
| **Pandas**                 | Results analysis                  |
| **Matplotlib**             | Visualization                     |
| **Kaggle CLI**             | Dataset download                  |
| **Jupyter / Google Colab** | Development environment           |

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SarthakPatkar2006/Deep-Learning.git
```

### 2. Navigate to the Project

```bash
cd Deep-Learning
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib kaggle
```

### 4. Configure Kaggle API

The dataset is downloaded using the Kaggle CLI.

Configure your Kaggle API credentials according to the official Kaggle documentation:

[Kaggle API Documentation](https://www.kaggle.com/docs/api)

> **Security:** Never commit API tokens, credentials, or secret keys to a public repository. Use environment variables or Kaggle's supported authentication methods.

---





### Execution Workflow

```text
Install Dependencies
        ↓
Configure Kaggle API
        ↓
Download Dataset
        ↓
Extract Dataset
        ↓
Load Training / Validation / Test Data
        ↓
Apply Preprocessing
        ↓
Build VGG16 Model
        ↓
Train and Evaluate VGG16
        ↓
Build ResNet50 Model
        ↓
Train and Evaluate ResNet50
        ↓
Build EfficientNetB0 Model
        ↓
Train and Evaluate EfficientNetB0
        ↓
Build AlexNet Model
        ↓
Train and Evaluate AlexNet
        ↓
Compare Accuracy, Loss, and Training Time
        ↓
Generate Visualization
```

---

## 📁 Recommended Repository Structure

```text
Deep-Learning/
│
├── 📓 Training.ipynb
├── 📄 README.md

```

---

## ⚠️ Limitations

* The models are trained for only **3 epochs**, so the results may not represent their full potential.
* The project compares architectures under a limited experimental configuration.
* The pretrained backbones are used as feature extractors in the notebook; extensive fine-tuning is not implemented.
* Accuracy alone does not provide a complete view of classification performance.
* Dataset performance may not directly represent real-world images captured under different conditions.
* Training time depends on the available hardware and execution environment.

---

## 🔮 Future Improvements

* [ ] Increase the number of training epochs.
* [ ] Add model fine-tuning by unfreezing selected pretrained layers.
* [ ] Compare precision, recall, and F1-score.
* [ ] Generate confusion matrices for each model.
* [ ] Add ROC curves and AUC comparison.
* [ ] Use callbacks such as EarlyStopping and ReduceLROnPlateau.
* [ ] Save trained models for inference.
* [ ] Build a Streamlit web application.
* [ ] Add single-image prediction functionality.
* [ ] Compare parameter count and inference latency.
* [ ] Perform systematic hyperparameter tuning.
* [ ] Add experiment tracking for reproducible comparisons.

---

## 📚 Learning Outcomes

This project demonstrates:

* Binary image classification
* CNN architecture fundamentals
* Transfer learning
* Pretrained model utilization
* Data augmentation
* Image preprocessing
* Model compilation and training
* Validation and test evaluation
* Performance comparison
* Training-time measurement
* Visualization of deep learning results

---

## 🤝 Contributing

Contributions and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/improvement
```

3. Commit your changes.

```bash
git commit -m "Improve model evaluation"
```

4. Push the branch.

```bash
git push origin feature/improvement
```

5. Open a Pull Request.

---

## 📄 License

This project is intended for educational and research purposes.

Add an appropriate license file to the repository if you intend to distribute the project publicly.

---

## 👨‍💻 Author

**Sarthak Anil Patkar**


---

<p align="center">
  <b>🐱🐶 Exploring Deep Learning Through CNN Architecture Comparison.</b>
</p>
