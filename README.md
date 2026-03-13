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
  A volatility-aware LSTM that forecasts daily TSLA closing prices, benchmarked against ARIMA and a fast KNN baseline — covering preprocessing, training, evaluation, and a rolling 5-day forward look.
</p>

---

## Table of Contents

- [Overview & Results](#overview--results)
- [Model Comparison](#model-comparison)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Notebook Workflow](#notebook-workflow)
  - [KNN Baseline Script](#knn-baseline-script)
- [Data](#data)
- [Architecture](#architecture)
- [Modeling Notes](#modeling-notes)
- [Contributing](#contributing)
- [License](#license)

---

## Overview & Results

This project walks through the full ML pipeline for stock price forecasting:

1. **Ingest & clean** — TSLA OHLCV data from Yahoo Finance
2. **Scale & split** — MinMax normalization + train / validation / test splits
3. **Train** — LSTM with dropout and early stopping; ARIMA and KNN as baselines
4. **Evaluate** — MAE, RMSE, R² with residual and trend-fit plots
5. **Forecast** — rolling 5-day horizon from the KNN script in seconds

---

## Model Comparison

| Model | MSE | MAE | RMSE | R² |
|---|---|---|---|---|
| **LSTM** (deep model) | `0.00080` | `0.02165` | `0.0283` | **0.9586** |
| KNN (baseline) | — | — | — | quick check |
| ARIMA (linear baseline) | `81 273.42` | `285.08` | `285.08` | — |

> **LSTM explains ~95.86% of price variance** on the held-out test set, while ARIMA's error magnitude highlights the advantage of sequence modeling on this dataset.

---

## Repository Layout

```
Stock-Market-Prediction/
├── Project Code/
│   ├── LSTM Networks.ipynb   # End-to-end: load → scale → train LSTM/ARIMA → evaluate/plot
│   ├── knn_model.py          # Fast KNN baseline + rolling 5-day forecast
│   └── tesla.csv             # Sample TSLA OHLCV data (Yahoo Finance export)
└── README.md
```

---

## Requirements

- Python 3.8+
- The following packages:

| Package | Purpose |
|---|---|
| `numpy` / `pandas` | Data wrangling |
| `scikit-learn` | KNN, MinMax scaling, metrics |
| `tensorflow` 2.x | LSTM model |
| `statsmodels` | ARIMA baseline |
| `matplotlib` / `seaborn` | Visualizations |
| `jupyter` | Notebook runtime |

---

## Quick Start

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/Stock-Market-Prediction.git
cd Stock-Market-Prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow statsmodels jupyter
```

### Notebook Workflow

```bash
jupyter notebook "Project Code/LSTM Networks.ipynb"
```

Run cells top-to-bottom. The notebook will:

- Load `tesla.csv`, scale prices, and build train/val/test splits
- Train the LSTM (with early stopping) and the ARIMA baseline
- Print MAE, RMSE, and R² for each model
- Plot price-fit and residual charts side by side

### KNN Baseline Script

```bash
python "Project Code/knn_model.py"
```

Outputs:
- Test-set MSE
- A rounded **5-day rolling forecast** for a quick plausibility check

> Run this before and after any LSTM changes as a fast regression test.

---

## Data

| Field | Detail |
|---|---|
| Default file | `Project Code/tesla.csv` |
| Columns | `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits` |
| Source | Yahoo Finance export |

**Swapping tickers:**

1. Replace `tesla.csv` with your own Yahoo Finance CSV export.
2. Update the path in the notebook:
   ```python
   df = pd.read_csv(r'Project Code/your_ticker.csv')
   ```
3. Ensure column names match the list above — preprocessing runs without changes.

---

## Architecture

```
Raw OHLCV CSV
      │
      ▼
MinMax Scaling  ──────────────────────────────────────────────┐
      │                                                        │
      ▼                                                        ▼
Train / Val / Test Split                               ARIMA Baseline
      │
      ▼
  ┌─────────────────────────────────┐
  │  LSTM Network                   │
  │  ┌──────────┐  ┌──────────┐    │
  │  │ LSTM     │→ │ LSTM     │→ Dense(1)
  │  │ + Dropout│  │ + Dropout│    │
  │  └──────────┘  └──────────┘    │
  └─────────────────────────────────┘
      │
      ▼
Inverse Scale → Predicted Close Price
      │
      ▼
MAE · RMSE · R² + Trend/Residual Plots
```

---

## Modeling Notes

<details>
<summary><strong>Lookback window & forecast horizon</strong></summary>

The lookback window (number of past days the LSTM sees) and forecast horizon are easily tunable in the notebook. Defaults are set for daily TSLA data — shorter windows react faster to regime changes; longer windows smooth out noise.

</details>

<details>
<summary><strong>Scaling</strong></summary>

MinMax scaling keeps all inputs in `[0, 1]`, which stabilizes LSTM gradient flow. If you add engineered features (e.g., RSI, volume delta), ensure each feature is scaled independently before stacking.

</details>

<details>
<summary><strong>Early stopping</strong></summary>

Patience is tuned for noisy price regimes. For smoother assets or longer histories, increase patience to allow the model more time to converge before halting.

</details>

<details>
<summary><strong>Baselines first workflow</strong></summary>

Run `knn_model.py` before and after any LSTM architecture change. A sudden drop in KNN performance is a strong signal that a data preprocessing step broke upstream.

</details>

---

## Contributing

Issues, ideas, and pull requests are welcome. Areas of particular interest:

- Alternative tickers or multi-ticker support
- Additional baselines (XGBoost, Prophet, Transformer)
- Enhanced visualizations (interactive Plotly charts, walk-forward validation plots)
- Hyperparameter search (Optuna, Keras Tuner)

Please open an issue to discuss larger changes before submitting a PR.

---

## License

Released under the [MIT License](LICENSE).
