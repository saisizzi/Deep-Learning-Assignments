# 🎬 Sentiment Analysis Using RNN, LSTM & GRU

<p align="center">
  <b>Sequence Classification on the IMDB Movie Reviews Dataset</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Task-Text%20Classification-6A5ACD?style=for-the-badge" alt="Task">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-50K%20IMDB%20Reviews-2E8B57?style=flat-square" alt="Dataset">
  <img src="https://img.shields.io/badge/Models-RNN%20%7C%20LSTM%20%7C%20GRU-orange?style=flat-square" alt="Models">
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-3498DB?style=flat-square" alt="Visualization">
  <img src="https://img.shields.io/badge/Notebook-Jupyter%20%7C%20Colab-F37626?style=flat-square" alt="Notebook">
</p>

---

## 📌 Overview

This project implements and compares three recurrent neural network architectures for **binary sentiment classification**:

- 🔹 Simple Recurrent Neural Network — **RNN**
- 🔹 Long Short-Term Memory — **LSTM**
- 🔹 Gated Recurrent Unit — **GRU**

The models are trained on the **IMDB Dataset of 50K Movie Reviews** to classify movie reviews as either **positive** or **negative**.

The project evaluates and compares the models based on classification performance, training behavior, computational complexity, and training efficiency.

---

## ✨ Project Highlights

- 📚 Text preprocessing and cleaning
- 🔤 Text tokenization using Keras
- 🧩 Sequence padding and truncation
- 🧠 Implementation of RNN, LSTM, and GRU models
- ⚖️ Fair comparison using identical experimental settings
- 📊 Accuracy, precision, recall, and F1-score evaluation
- 📈 Training and validation performance curves
- 🔥 Confusion matrix visualization
- 📉 ROC and Precision-Recall curves
- ⏱️ Training-time comparison
- 🧮 Model parameter comparison
- 🏆 Final model ranking

---

## 🧠 Project Workflow

```text
                    IMDB Movie Reviews
                            │
                            ▼
                   Data Preprocessing
                            │
                            ▼
                       Tokenization
                            │
                            ▼
                    Sequence Padding
                            │
                            ▼
                     Embedding Layer
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
         Simple RNN        LSTM            GRU
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                      Dropout Layer
                            │
                            ▼
                  Dense Sigmoid Output
                            │
                            ▼
                  Positive / Negative
```

---

## 📂 Dataset

### IMDB Dataset of 50K Movie Reviews

The dataset contains **50,000 movie reviews** collected from IMDb. Each review is labeled as either positive or negative.

| Sentiment | Encoded Value | Number of Reviews |
|-----------|---------------|-------------------|
| Positive | `1` | 25,000 |
| Negative | `0` | 25,000 |
| **Total** | — | **50,000** |

### Dataset Source

[Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical operations |
| Pandas | Data loading and analysis |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| KaggleHub | Dataset downloading |
| Jupyter Notebook | Development environment |
| Google Colab | Optional cloud execution environment |

---

## ⚙️ Model Configuration

The same major hyperparameters are used for all three models to ensure a fair comparison.

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary Size | `20,000` |
| Maximum Sequence Length | `250` |
| Embedding Dimension | `128` |
| Recurrent Units | `64` |
| Dropout Rate | `0.3` |
| Batch Size | `64` |
| Maximum Epochs | `5` |
| Learning Rate | `0.001` |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Output Activation | Sigmoid |

---

## 🧹 Data Preprocessing

The following preprocessing operations are performed:

1. Load the dataset using Pandas.
2. Check the dataset shape and structure.
3. Check for missing values.
4. Check for duplicate reviews.
5. Convert text to lowercase.
6. Remove HTML tags such as `<br />`.
7. Remove unnecessary special characters.
8. Normalize extra whitespace.
9. Convert sentiment labels into binary values.
10. Tokenize the reviews.
11. Convert reviews into integer sequences.
12. Pad or truncate sequences to a fixed length.

### Dataset Split

| Dataset | Samples | Percentage |
|---------|---------|------------|
| Training | 40,000 | 80% |
| Validation | 5,000 | 10% |
| Testing | 5,000 | 10% |
| **Total** | **50,000** | **100%** |

> The tokenizer is fitted only on the training dataset to avoid data leakage.

---

## 🏗️ Model Architectures

### 1. Simple RNN

A Simple RNN processes sequential input one time step at a time and maintains a hidden state containing information from previous tokens.

```text
Input
  ↓
Embedding
  ↓
SimpleRNN
  ↓
Dropout
  ↓
Dense Sigmoid Output
```

#### Advantages

- Simple architecture
- Lower computational cost
- Easy to implement

#### Limitations

- Can struggle with long-term dependencies
- More susceptible to vanishing-gradient problems

---

### 2. LSTM

Long Short-Term Memory networks use memory cells and gates to retain important information over longer sequences.

```text
Input
  ↓
Embedding
  ↓
LSTM
  ↓
Dropout
  ↓
Dense Sigmoid Output
```

#### Advantages

- Handles long-term dependencies effectively
- Reduces the vanishing-gradient problem
- Suitable for complex sequential patterns

#### Limitations

- More parameters
- Higher computational cost
- May require more training time

---

### 3. GRU

Gated Recurrent Units use update and reset gates to control information flow. GRU is simpler than LSTM while still being capable of learning long-term dependencies.

```text
Input
  ↓
Embedding
  ↓
GRU
  ↓
Dropout
  ↓
Dense Sigmoid Output
```

#### Advantages

- Fewer parameters than LSTM
- Often faster to train
- Captures long-term dependencies
- Good balance between performance and efficiency

#### Limitations

- Performance depends on the dataset and hyperparameters
- May not always outperform LSTM

---

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Percentage of correctly classified reviews |
| Precision | Proportion of predicted positive reviews that are actually positive |
| Recall | Proportion of actual positive reviews correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Ability of the model to distinguish between the two classes |
| PR-AUC | Trade-off between precision and recall |
| Training Time | Time required to train the model |
| Total Parameters | Number of parameters in the model |

---

## 📈 Visualizations

The project generates several visualizations for data analysis, training analysis, and model comparison.

### Dataset Visualizations

- Sentiment distribution countplot
- Review-length histogram
- Review-length KDE plot

### Training Visualizations

- Training accuracy curves
- Validation accuracy curves
- Training loss curves
- Validation loss curves

### Model Comparison Visualizations

- Accuracy comparison
- Precision comparison
- Recall comparison
- F1-score comparison
- ROC-AUC comparison
- PR-AUC comparison
- Training-time comparison
- Parameter-count comparison

### Classification Visualizations

- Confusion matrices
- ROC curve comparison
- Precision-Recall curve comparison
- Final model ranking

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sequence-classification-rnn-lstm-gru.git
```

Navigate to the project directory:

```bash
cd sequence-classification-rnn-lstm-gru
```

---

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub jupyter
```

You can also use the following command inside a Jupyter Notebook or Google Colab cell:

```python
!pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub
```

---

### 3. Download the Dataset

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

print("Path to dataset files:", path)
```

The downloaded dataset contains the file:

```text
IMDB Dataset.csv
```

---

### 4. Run the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
sequence_classification_rnn_lstm_gru.ipynb
```

Run the notebook cells in the following order:

```text
1. Install dependencies
2. Import libraries
3. Download the dataset
4. Explore the dataset
5. Preprocess the reviews
6. Split the dataset
7. Tokenize and pad sequences
8. Build the RNN, LSTM, and GRU models
9. Train the models
10. Evaluate the models
11. Generate visualizations
12. Compare the final results
```

---

## 💻 Running on Google Colab

Google Colab is recommended if you want to use a cloud GPU.

### Steps

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the notebook.
3. Select:

```text
Runtime → Change runtime type → T4 GPU
```

4. Run all cells sequentially.

A GPU can significantly reduce the training time of recurrent neural networks.

---

## 📁 Repository Structure

```text
sequence-classification-rnn-lstm-gru/
│
├── README.md
│
├── sequence_classification_rnn_lstm_gru.ipynb
│
├── requirements.txt
│
└── results/
    ├── model_results.csv
    └── visualizations/
```

---

## 🧪 Example Prediction

After training, the model can be used to predict the sentiment of a new movie review.

```python
sample_review = [
    "The movie was fantastic, engaging, and beautifully acted."
]

sample_sequence = tokenizer.texts_to_sequences(sample_review)

sample_padded = tf.keras.preprocessing.sequence.pad_sequences(
    sample_sequence,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

prediction = model.predict(sample_padded)[0][0]

if prediction >= 0.5:
    print("Positive Review")
else:
    print("Negative Review")
```

---

## 🏆 Results

The final results are generated after training and evaluating all three models.

The results table contains:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Training Time | Parameters |
|-------|----------|-----------|--------|----------|---------|--------|---------------|------------|
| Simple RNN | Generated after execution | — | — | — | — | — | — | — |
| LSTM | Generated after execution | — | — | — | — | — | — | — |
| GRU | Generated after execution | — | — | — | — | — | — | — |

> The actual values depend on the execution environment, random initialization, TensorFlow version, hardware, and training configuration.

---

## 🔍 Key Observations

The project analyzes the following aspects:

- Simple RNN has a relatively simple architecture and lower computational cost.
- LSTM is designed to capture long-term dependencies using memory cells and gates.
- GRU uses a simpler gated architecture than LSTM.
- LSTM and GRU generally handle long-term dependencies better than a basic RNN.
- A model with more parameters does not always provide better performance.
- The best model should be selected by considering both predictive performance and computational efficiency.

---

## 🔮 Future Improvements

Potential improvements include:

- Implementing Bidirectional RNN, LSTM, and GRU models
- Using pretrained word embeddings such as GloVe
- Applying hyperparameter tuning
- Increasing the training epochs
- Adding attention mechanisms
- Comparing with Transformer-based models
- Deploying the best model using Streamlit
- Creating a web interface for real-time sentiment prediction
- Saving and loading trained models for deployment

---

## 📌 Important Notes

- All three models use the same dataset splits.
- The same tokenizer is used for all models.
- The same major hyperparameters are used for a fair comparison.
- The test dataset is used only for final evaluation.
- Early stopping is used to reduce overfitting.
- The tokenizer is fitted only on training data.
- Actual model performance must be obtained by running the notebook.
- The model ranking may vary depending on the experimental setup.

---

## 👨‍💻 Author

**Sarthak Anil Patkar**


---

## 📄 License

This project is created for educational and academic purposes.

---

<p align="center">
  ⭐ If you found this project useful, consider giving the repository a star!
</p>