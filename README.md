# 🧠 DeepStock AI: Predicting Stock Prices with LSTM & ARIMA Baselines

This project, **DeepStock AI**, focuses on forecasting stock prices using both traditional and deep learning methods — including **ARIMA**, **Dense (MLP)**, and **Bidirectional LSTM** models. The system evaluates performance based on multiple statistical and deep learning techniques to find the most effective predictor.

---

## 📊 Project Overview

The goal of **DeepStock AI** is to build a reliable forecasting pipeline that:

* Uses **log-returns** instead of raw prices for stationarity.
* Compares **traditional statistical (ARIMA)** and **deep learning (Dense & LSTM)** models.
* Produces a **side-by-side performance table** (RMSE, MAE, R²).
* Visualizes **training losses, predictions, and rolling statistics**.
* Saves all preprocessing steps (scaler, model weights, comparison CSV).

---

## 🧩 Model Architectures

### 🔹 1. ARIMA Model (Statistical Baseline)

* Captures linear temporal dependencies.
* Trained on **log-return** values.
* Provides a benchmark for comparison against deep learning models.

### 🔹 2. Dense (MLP) Model

* Non-sequential neural network baseline.
* Uses **flattened window inputs** from the time series.
* Helps evaluate deep learning benefits over simple neural architectures.

### 🔹 3. Bidirectional LSTM Model

* Sequence-based deep learning model.
* Learns temporal dependencies in both forward and backward directions.
* Captures long-term trends in stock data efficiently.

---

## ⚙️ Installation

To set up and run this project locally:

```bash
# Clone the repository
git clone https://github.com/Rayhan-Dodi/DeepStock-

# Navigate to the project folder
cd DeepStock-AI

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

# Install required dependencies
pip install -r requirements.txt
```

---

## 🧾 Dataset

The project uses a CSV file named **`indexStock_Funal.csv`**, which contains the historical stock price data.

| Column | Description    |
| ------ | -------------- |
| Date   | Trading date   |
| Open   | Opening price  |
| High   | Highest price  |
| Low    | Lowest price   |
| Close  | Closing price  |
| Volume | Trading volume |

---

## 🚀 Usage

### 1️⃣ Open the Notebook

Run the Jupyter Notebook file:

```bash
jupyter notebook DeepStock_AI_Model.ipynb
```

### 2️⃣ Steps inside the notebook

* Load and visualize the dataset.
* Compute log-returns and analyze rolling statistics.
* Train ARIMA, Dense, and LSTM models.
* Generate evaluation metrics and comparison CSV.
* Visualize:

  * Series plots
  * Rolling mean & standard deviation
  * Training loss
  * Actual vs. Predicted (Full + Zoomed view)

---

## 📈 Results & Performance Summary

| Model       | RMSE           | MAE            | R²             |
| ----------- | -------------- | -------------- | -------------- |
| ARIMA       | *8810.57* | *6392.54* | *-0.9746* |
| Dense (MLP) | *7196.58* | *6015.29* | *-0.3174* |
| LSTM        | *6070.28* | *5087.57* | *0.0627* |

📌 *You can update this table after running the final notebook.*

### Example Outputs

* **Training Loss Curve**
* **Predicted vs. Actual Graphs (Full & Zoomed)**
* **Rolling Stats Visualization**

---

## 💾 Saved Artifacts

All key artifacts are saved for reproducibility:

* `scaler.pkl` → Data normalization
* `arima_model.pkl` → Fitted ARIMA model
* `dense_model.h5` → Trained Dense model
* `lstm_model.h5` → Trained Bidirectional LSTM model
* `comparison_results.csv` → Evaluation summary

---

## 🧠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Statsmodels**
* **NumPy, Pandas, Matplotlib, Seaborn**
* **Scikit-learn**

---

## 📚 Future Improvements

* Add GRU and Transformer-based models for deeper comparisons
* Include feature engineering using technical indicators
* Build a Flask dashboard for live stock forecasting

---

## 🧑‍💻 Author

**Rayhan Kabir Dodi**
💼 Research-based AI enthusiast focused on time-series forecasting and deep learning applications.
📫 Feel free to connect on GitHub or LinkedIn.

---

## 🪪 License

This project is open-source under the **MIT License**.
Feel free to use, modify, and share it for research or educational purposes.

