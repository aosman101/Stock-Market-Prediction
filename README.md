# Deep Stock Insights: Predicting Stocks with LSTM Networks

LSTM forecasting with **volatility-aware early stopping**, systematic hyperparameter tuning, and risk-adjusted evaluation. Includes a **lightweight KNN baseline** for quick comparisons.

Topics: Time Series, ARIMA, RNN, LSTM, Finance, TensorFlow, NumPy, Scikit-learn, Pandas, Statsmodels, Random Forest (sklearn implementation), PyTorch, Stock Forecasting, Backtesting, AI, Volatility, KNN, Machine Learning, Neural Networks, and Deep Learning.


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#requirements)
[![TensorFlow 2](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#requirements)

---

## Overview

Classical time-series models frequently struggle to capture non-linear dynamics and regime shifts in equity markets. This repository features an **LSTM forecaster** designed for daily stock prices, with a demonstration using TSLA. It incorporates **volatility-sensitive early stopping** to mitigate overfitting during turbulent market conditions and evaluates performance using **risk-adjusted metrics**. Additionally, a concise **KNN** script is included as a quick, non-parametric baseline and a 5-day forecaster.

**What you’ll find here**

- 🧠 **Tuned LSTM**: lookback, units, dropout, learning rate schedule, and early-stopping patience geared to volatile data.
- 📈 **Trading-aware evaluation**: MAE / RMSE / R² alongside **Sharpe**, **Sortino**, and **max drawdown**.
- ⚖️ **Baseline sanity check**: a compact **KNN regressor** that trains in seconds and prints a rolling 5-day forecast.
- 🔁 **Reproducible workflow**: clear splits, fixed seeds, and a single notebook that runs end-to-end.

---

## Repository Structure

├── Project Code/

├── LSTM Networks.ipynb # End-to-end workflow: data collection → training → evaluation → backtesting.

├── tesla.csv # Acquired from the Yahoo API.

│ └── knn_model.py # Basic K-Nearest Neighbours (KNN) model for predicting Tesla (TSLA) closing prices with a 5-day forecast.


└── README.md.
