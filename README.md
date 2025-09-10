# Deep Stock Insights: Predicting Stocks Using LSTM Networks

Adil Osman - 13199246.

Stock market prediction using Long Short-Term Memory (LSTM) hyperparameter tuning and volatility-sensitive early stopping enhances predictive accuracy.

## Alternative Time Series Model

In addition to the LSTM notebook, this repository provides a simple k-nearest neighbors (KNN) regressor example for stock price forecasting. The script `Project Code/knn_model.py` trains a non-linear KNN model on Tesla closing prices and generates a five-day forecast.

### Running the KNN example

```bash
python "Project Code/knn_model.py"
```

The script reports mean squared error on a small test split and prints the next five-day forecast values.
