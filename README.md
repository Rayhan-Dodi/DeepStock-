```markdown
# 📊 Deepstock

Deepstock-ML is a machine learning & AI pipeline for **stock market prediction (1990–2022 data)**.  
It integrates technical indicators, preprocessing equations, and multiple AI/ML models including Neural Networks, LSTMs, Autoencoders, RBMs, ELM, and RBF networks.

---

## 🚀 Project Roadmap

### 1. Data Collection
- Historical stock data (1990–2022)
- Features: `Open, High, Low, Close, Volume`
- Optional: Sentiment/Fundamental data

---

### 2. Data Preprocessing
- **Normalization (Min–Max Scaling):**  
  \[
  a^{scaled}_m = \frac{a^i_m - a_{min}}{a_{max} - a_{min}}
  \]

- **Anti-normalization (inverse scaling):**  
  \[
  \hat{y_t} = y^{scaled}_t \cdot (y_{max} - y_{min}) + y_{min}
  \]

- **Feature Selection (Correlation):**  
  \[
  Corr(i) = \frac{cov(a_i, b)}{\sqrt{var(a_i) \cdot var(b)}}
  \]

---

### 3. Feature Engineering (Technical Indicators)
- **SMA:**  
  \[
  SMA(t,N) = \frac{1}{N}\sum_{k=1}^N CP(t-k)
  \]

- **EMA:**  
  \[
  EMA(t,\Delta) = (CP(t) - EMA(t-1))\cdot\Gamma + EMA(t-1)
  \]

- **MACD:**  
  \[
  MACD = EMA(t,k) - EMA(t,d)
  \]

- **OBV:**  
  \[
  OBV = OBV_{pr} \pm Volume
  \]

- **RSI:**  
  \[
  RSI = 100 \cdot \frac{1}{1+RS(t)}
  \]

---

### 4. Model Building (ML + AI)
- **Neural Network (Perceptron):**  
  \[
  Z = W^T X + b
  \]

- **RNN / LSTM:**  
  \[
  M_t = \tanh(W[STM_{t-1},E_t] + b)
  \]

- **Autoencoder:**  
  \[
  E(X,X') = ||X - X'||^2
  \]

- **RBM (Restricted Boltzmann Machine):**  
  \[
  G(X,Y) = -\alpha^T X - \beta^T Y - X^T W Y
  \]

- **Extreme Learning Machine (ELM):**  
  \[
  Y_j = \sum_{i=1}^d \eta_i f(W_i X_j + e_i)
  \]

- **Radial Basis Function Network (RBF):**  
  \[
  y(x) = \sum_{i=1}^N \mu_i \nu(\|x-x_i\|)
  \]

---

### 5. Training
- Input: Technical Indicators + Normalized Features  
- Output: Predicted Stock Price / Trend  
- Optimization via Backpropagation + Gradient Descent

---

### 6. Evaluation
- **MSE:**  
  \[
  MSE = \frac{1}{n}\sum_{k=1}^n(y_k - \hat{y}_k)^2
  \]

- **RMSE:**  
  \[
  RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^N(\hat{y}_t-y_t)^2}
  \]

- **MAPE:**  
  \[
  MAPE = \frac{1}{N}\sum_{t=1}^N \frac{|y_t-\hat{y}_t|}{y_t}
  \]

- **Directional Accuracy (DA):**  
  \[
  DA = \frac{1}{N}\sum_{t=1}^N D_t
  \]

---

## 📂 Project Structure

```

Deepstock-ML/
│── data/
│   ├── raw/            # raw stock datasets
│   ├── processed/      # cleaned/normalized data
│   └── external/       # extra data (sentiment, fundamentals)
│
│── notebooks/
│   ├── EDA.ipynb       # exploratory data analysis
│   ├── Indicators.ipynb# technical indicators
│   └── Models.ipynb    # experiments with ML/AI models
│
│── src/
│   ├── preprocessing/  # normalization, feature engineering, selection
│   ├── models/         # NN, LSTM, Autoencoder, RBM, ELM, RBF
│   ├── training/       # training pipeline & optimizers
│   ├── evaluation/     # metrics & visualization
│   └── utils/          # helpers, config
│
│── results/
│   ├── models/         # saved trained models
│   ├── predictions/    # prediction outputs
│   └── logs/           # training logs
│
│── requirements.txt
│── README.md
│── main.py

````

---

## ⚙️ Usage

1. Place your dataset in `data/raw/` (CSV format).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

3. Run the training pipeline:

   ```bash
   python -m src.training.train_pipeline
   ```
4. Results (predictions, models, plots) will be saved in `results/`.

---

## 📊 Evaluation Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Directional Accuracy (DA)

---

## 🔮 Future Improvements

* Add sentiment analysis from news/Twitter
* Implement advanced feature selection (PCA, mutual info)
* Ensemble multiple models (LSTM + ELM + RBF)
* Deploy via Flask/FastAPI for live predictions

---

## ✍️ Author

Developed by **Rayhan Kabir Dodi** for academic and research purposes.

```

