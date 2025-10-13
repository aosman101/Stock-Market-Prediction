# Deep Stock Insights: Predicting Stocks with LSTM Networks

LSTM forecasting with **volatility-aware early stopping**, systematic hyperparameter tuning, and risk-adjusted evaluation. Includes a **lightweight KNN baseline** for quick comparisons.

Topics - Time-Series, ARIMA, RNN, LSTM, Finance, Tensorflow, Numpy, Scikit-learn, Pandas, Statsmodels, Random Forest (sklearn implementation), PyTorch, Stock-Forecasting, Backtesting, AI, Volatility, KNN, Machine-Learning, Neural-Networks and Deep-Learning.


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#requirements)
[![TensorFlow 2](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#requirements)

---

## Overview

Classical time-series models often struggle with non-linear dynamics and regime shifts in equities. This repo implements an **LSTM forecaster** for daily stock prices (demonstrated on TSLA), adds **volatility-sensitive early stopping** to curb overfitting during turbulent periods, and evaluates performance with **risk-adjusted metrics**. A tiny **KNN** script serves as a fast, non-parametric baseline and 5-day forecaster.

**What you’ll find here**

- 🧠 **Tuned LSTM**: lookback, units, dropout, learning rate schedule, and early-stopping patience geared to volatile data.
- 📈 **Trading-aware evaluation**: MAE / RMSE / R² alongside **Sharpe**, **Sortino**, and **max drawdown**.
- ⚖️ **Baseline sanity check**: a compact **KNN regressor** that trains in seconds and prints a rolling 5-day forecast.
- 🔁 **Reproducible workflow**: clear splits, fixed seeds, and a single notebook that runs end-to-end.

---

## Repository Structure

├── Project Code/

├── LSTM Networks.ipynb # End-to-end workflow: data → train → eval → backtest.

├── tesla.csv # Obtained from Yahoo API.

│ └── knn_model.py # Simple KNN baseline on TSLA closes (5-day forecast).


└── README.md.
