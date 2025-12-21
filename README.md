# Deep Stock Insights: Predicting Stocks with LSTM Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#requirements)
[![TensorFlow 2](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#requirements)
[![Made with Jupyter](https://img.shields.io/badge/Format-Jupyter%20Notebook-f37726.svg)](#notebook-workflow)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

Volatility-aware LSTM forecasting on daily TSLA prices, plus a **lightweight KNN baseline** for quick reality checks. The notebook walks through preprocessing → training → evaluation with MAE / RMSE / R², while the standalone script prints a rolling 5-day forecast in seconds.

- **Topics**: Time Series, ARIMA, RNN, LSTM, Finance, TensorFlow, NumPy, Scikit-learn, Pandas, Statsmodels, Baselines (KNN), Stock Forecasting, Backtesting, Deep Learning.

---

## Overview & Results

- **Workflow**: Cleans and scales TSLA OHLCV data, builds train/validation/test splits, and fits an LSTM alongside ARIMA and KNN baselines. The KNN script also prints a rolling 5-day forecast in seconds.
- **LSTM performance** (test set): MSE `0.00080`, MAE `0.02165`, RMSE `0.03`, R² `0.9586` (~95.86% variance explained), showing tight tracking of price moves.
- **ARIMA baseline** (test set): MSE `81273.41845`, MAE `285.08490`, underscoring the gap between the deep model and a linear alternative on this dataset.

---

## Quick Links

- Notebook workflow: `Project Code/LSTM Networks.ipynb`
- Baseline script: `Project Code/knn_model.py`
- Sample data: `Project Code/tesla.csv`

---

## Highlights

- 🧠 **Volatility-sensitive LSTM**: tuned lookback window, dropout, and early stopping to handle choppy price regimes.
- 📊 **Clear evaluation**: MAE, RMSE, and R² with plots for trend fit and error behavior.
- ⚖️ **Instant baseline**: KNN regressor that trains fast and prints a rolling 5-day horizon for sanity checks.
- 🔁 **Reproducible flow**: fixed splits, deterministic seeds, and a single notebook that runs end-to-end.

---

## Repository Layout

```
.
├── Project Code/
│   ├── LSTM Networks.ipynb   # End-to-end workflow: load → scale → train LSTM/ARIMA → evaluate/plot.
│   ├── knn_model.py          # Quick KNN baseline + 5-day rolling forecast.
│   └── tesla.csv             # Sample TSLA OHLCV data (Yahoo Finance export).
└── README.md
```

---

## Requirements

- Python 3.8+
- Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow` (2.x), `statsmodels`, `jupyter`

Install them in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow statsmodels jupyter
```

---

## Run It Yourself

### Notebook workflow

1) Activate the environment and launch Jupyter:
```bash
jupyter notebook "Project Code/LSTM Networks.ipynb"
```
2) Run the cells top-to-bottom. The notebook:
   - Loads `Project Code/tesla.csv`, scales prices, and builds train/validation/test splits.
   - Trains an LSTM (and ARIMA for comparison), using early stopping to avoid overfitting.
   - Reports MAE, RMSE, and R², plus plots for price fit and residuals.

### Lightweight KNN baseline

```bash
python "Project Code/knn_model.py"
```
- Prints test-set MSE.
- Outputs a rounded 5-day rolling forecast for quick plausibility checks.

---

## Data

- Default sample: `Project Code/tesla.csv` with columns `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`.
- Swap in another ticker by replacing the CSV and updating the path in the notebook (the default read is `df = pd.read_csv(r'Project Code/tesla.csv')`).
- Make sure your CSV matches the column names above to keep preprocessing identical.

---

## Modeling Notes

- **Lookback window** and **forecast horizon** are easy levers to tweak in the notebook; the defaults target daily TSLA behavior.
- **Scaling**: MinMax scaling keeps LSTM training stable; adjust if you add engineered features.
- **Early stopping**: patience is tuned for noisy regimes—lengthen it if you add smoother assets or longer histories.
- **Baselines first**: run the KNN script before/after LSTM changes to catch regressions quickly.

---

## Contributing

Issues, ideas, and PRs are welcome—especially for alternative tickers, additional baselines, or visualization improvements.

---

## License

Released under the [MIT License](LICENSE).
