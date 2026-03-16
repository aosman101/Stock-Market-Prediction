<h1 align="center">Deep Stock Insights</h1>
<h3 align="center">Predicting TSLA Prices with LSTM Networks + KNN Baseline</h3>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow 2.x"></a>
  <a href="#notebook-workflow"><img src="https://img.shields.io/badge/Format-Jupyter%20Notebook-f37726.svg" alt="Jupyter"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</p>

<p align="center">
  A deep LSTM model forecasting daily TSLA closing prices with ARIMA and KNN baselines, covering the full pipeline from data preprocessing to evaluation and 5-day rolling forecasts.
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Model Results](#model-results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/<your-username>/Stock-Market-Prediction.git
cd Stock-Market-Prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow statsmodels jupyter
```

### 2. Run the notebook

```bash
jupyter notebook "Project Code/LSTM Networks.ipynb"
```

Loads Tesla data, trains LSTM/ARIMA models, and generates evaluation plots.

### 3. Test the KNN baseline

```bash
python "Project Code/knn_model.py"
```

Quick 5-day rolling forecast and MSE check.

---

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LSTM** | 0.0217 | 0.0283 | **0.9586** |
| ARIMA (baseline) | 285.08 | 285.08 | — |
| KNN (fast check) | — | — | — |

The LSTM captures ~95.86% of price variance, significantly outperforming the linear ARIMA baseline.

---

## Repository Structure

```
Stock-Market-Prediction/
├── Project Code/
│   ├── LSTM Networks.ipynb    # End-to-end training & evaluation
│   ├── knn_model.py           # KNN baseline & rolling forecast
│   └── tesla.csv              # Sample Tesla OHLCV data
├── README.md
└── LICENSE
```

---

## Requirements

**Python 3.8+** with these packages:
- `numpy`, `pandas` — Data handling
- `scikit-learn` — KNN, scaling, metrics  
- `tensorflow` 2.x — LSTM model
- `statsmodels` — ARIMA baseline
- `matplotlib`, `seaborn` — Visualization
- `jupyter` — Notebook runtime

Install all: `pip install numpy pandas scikit-learn matplotlib seaborn tensorflow statsmodels jupyter`

---

## Data & Customization

The notebook uses Yahoo Finance OHLCV data. Default: `Project Code/tesla.csv`

**Using a different ticker:**
1. Export CSV from Yahoo Finance
2. Update notebook: `df = pd.read_csv('your_ticker.csv')`
3. Ensure columns: Date, Open, High, Low, Close, Volume

---


## Architecture

```
OHLCV Data → MinMax Scale → Train/Val/Test Split
                                     ↓
                          ┌──────────────────────┐
                          │  LSTM Network        │
                          │  (2x LSTM layers)    │
                          │  + Dropout           │
                          └──────────────────────┘
                                     ↓
                      Inverse Scale → Forecast
                                     ↓
                          MAE, RMSE, R² Metrics
```

A 2-layer LSTM with dropout and early stopping trains on normalized price data, then inverse-scales predictions to actual prices.

---

## Key Tuning Parameters

- **Lookback window:** Number of past days the LSTM sees (tunable in notebook)
- **Forecast horizon:** Default 1-day ahead; KNN extends to rolling 5-day
- **MinMax scaling:** Normalizes features to [0,1] for stable gradient flow
- **Early stopping:** Prevents overfitting on noisy price regimes

---

## Contributing

We welcome issues, ideas, and PRs! Potential improvements:

- Alternative tickers or multi-ticker support
- Additional baselines (XGBoost, Prophet, Transformer)
- Interactive visualizations (Plotly, walk-forward analysis)
- Hyperparameter tuning (Optuna, Keras Tuner)

Please open an issue first to discuss larger changes.

---

## License

[MIT License](LICENSE)
