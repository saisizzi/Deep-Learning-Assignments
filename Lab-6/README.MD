
<div align="center">

# 📈 LSTM Time-Series Forecasting

### Airline Passenger Prediction Using Deep Learning

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Deep%20Learning-Time%20Series-purple" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project Status">
</p>

**A deep learning project to forecast monthly airline passenger traffic using LSTM neural networks and compare different historical look-back windows.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Experimental Setup](#-experimental-setup)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results and Visualizations](#-results-and-visualizations)
- [Technologies Used](#-technologies-used)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

## 🚀 Overview

Time-series forecasting is a machine learning task that uses historical observations to predict future values.

This project implements **Long Short-Term Memory (LSTM)** neural networks to predict monthly airline passenger counts using historical passenger data.

Three models are trained using different look-back windows:

- **6 months:** Captures recent passenger trends.
- **12 months:** Captures one year of historical observations.
- **24 months:** Captures two years of historical observations.

The models are evaluated using MAE and RMSE to determine which look-back window performs best on the held-out test dataset.

## 🎯 Objectives

- Load, preprocess, and visualize time-series data.
- Perform chronological training and testing splits.
- Normalize passenger counts using feature scaling.
- Convert the time series into supervised learning sequences.
- Build and train LSTM-based forecasting models.
- Compare the impact of different look-back windows.
- Evaluate predictions using MAE and RMSE.
- Visualize actual versus predicted passenger counts.
- Analyze model training and validation loss.

## 📊 Dataset

**Dataset:** Airline Passengers

The dataset contains monthly airline passenger counts and is commonly used for time-series forecasting experiments.

| Property | Description |
|---|---|
| Dataset | Airline Passengers |
| Data type | Univariate time series |
| Frequency | Monthly |
| Input feature | Historical passenger count |
| Target | Next month's passenger count |
| File format | CSV |

**Dataset source:** [Airline Passengers CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)

The notebook loads the CSV file from the configured data directory and processes the month and passenger-count columns.

## 🧠 Model Architecture

The LSTM network learns temporal patterns from a sequence of historical passenger counts.

```text
Historical Passenger Counts
             │
             ▼
       Input Sequence
             │
             ▼
        LSTM Layer
       (64 units)
             │
             ▼
      Dropout Layer
       (rate = 0.2)
             │
             ▼
    Dense Layer (32)
      ReLU Activation
             │
             ▼
      Output Layer
       (1 neuron)
             │
             ▼
 Predicted Passenger Count
```

### Model Components

| Component | Purpose |
|---|---|
| Input layer | Receives historical time-series sequences |
| LSTM | Learns temporal dependencies |
| Dropout | Helps reduce overfitting |
| Dense layer | Learns nonlinear representations |
| Output layer | Predicts the next passenger count |

## ⚙️ Experimental Setup

The model architecture and training configuration are kept consistent across experiments. The look-back window is the primary variable being compared.

| Parameter | Configuration |
|---|---|
| Model | LSTM Neural Network |
| Look-back windows | 6, 12, 24 months |
| LSTM units | 64 |
| Dense units | 32 |
| Dropout rate | 0.2 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Training metric | Mean Absolute Error |
| Maximum epochs | 100 |
| Batch size | 16 |
| Validation split | 10% |
| Early stopping patience | 15 epochs |
| Random seed | 42 |
| Train-test split | 80:20, chronological |

### Workflow

1. Load and inspect the dataset.
2. Visualize the original time series.
3. Split the observations chronologically.
4. Fit the scaler on the training data.
5. Generate sequences for each look-back window.
6. Train the LSTM model with early stopping.
7. Generate predictions on the test period.
8. Inverse-transform the predictions to the original scale.
9. Calculate evaluation metrics.
10. Compare the results and visualize the forecasts.

## 📏 Evaluation Metrics

### 1. Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted passenger counts.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

### 2. Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction error, penalizing larger errors more heavily.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

**Interpretation:** Lower MAE and RMSE indicate more accurate predictions. The best look-back window is selected based on test RMSE.

## 📁 Project Structure

```text
LSTM-Time-Series-Forecasting/
│
├── data/
│   └── airline-passengers.csv
│
├── notebooks/
│   └── Assignment_4.ipynb
│
├── results/
│   ├── passenger_time_series.png
│   ├── window_metric_comparison.png
│   ├── all_windows_actual_vs_predicted.png
│   ├── prediction_6_month_window.png
│   ├── prediction_12_month_window.png
│   ├── prediction_24_month_window.png
│   ├── loss_6_month_window.png
│   ├── loss_12_month_window.png
│   └── loss_24_month_window.png
│
├── README.md
└── requirements.txt
```

*Note: The structure above is a suggested repository layout. Keep the filenames and directories consistent with your actual notebook and generated files.*

## 💻 Installation

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- pip package manager

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd LSTM-Time-Series-Forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
jupyter
```

Install the packages:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. Open the project directory.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `Assignment_6.ipynb`.
4. Verify that the dataset path is correct.
5. Execute all cells sequentially.
6. Review the training history, evaluation metrics, and prediction plots.
7. Inspect the generated files in the `results/` directory.

## 📈 Results and Visualizations

The notebook generates visualizations to analyze the forecasting process.

| Visualization | Purpose |
|---|---|
| Passenger time series | Displays historical passenger trends |
| Metric comparison | Compares MAE and RMSE across look-back windows |
| Actual vs. predicted | Evaluates forecast alignment with actual observations |
| Training and validation loss | Tracks convergence and potential overfitting |

### Model Comparison

Use the following table to document the metrics produced by your notebook after execution.

| Look-back Window | MAE | RMSE |
|---|---:|---:|
| 6 months | — | — |
| 12 months | — | — |
| 24 months | — | — |

**Best configuration:** The model with the lowest test RMSE.

*Replace the placeholders with your actual experimental results. The best-performing window should be determined from the completed evaluation.*

## 🛠️ Technologies Used

- **Python** — Core programming language
- **TensorFlow / Keras** — Neural network implementation
- **NumPy** — Numerical computations
- **pandas** — Dataset manipulation
- **Matplotlib** — Data visualization
- **scikit-learn** — Data scaling and evaluation metrics
- **Jupyter Notebook** — Experimentation and analysis

## ⚠️ Limitations

- The experiment uses a small, univariate dataset.
- Forecasting performance may vary with the chosen look-back window.
- The experiment focuses on one-step-ahead prediction.
- No external variables, such as ticket prices or seasonal events, are included.
- The results do not establish that LSTM is superior to statistical forecasting methods.

## 🔮 Future Improvements

- Compare LSTM with GRU, ARIMA, and other forecasting models.
- Add a naive seasonal forecasting baseline.
- Evaluate additional look-back windows and hyperparameters.
- Incorporate external variables to improve prediction accuracy.
- Implement multi-step-ahead forecasting.
- Develop an interactive dashboard for forecast visualization.
- Deploy the trained model through a REST API.

---

<div align="center">

### Built with Python and Deep Learning

**LSTM-Based Time-Series Forecasting**

</div>
