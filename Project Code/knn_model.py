# k-NN for Time Series Forecasting
import csv
import math
import os
import pandas as pd
from typing import List, Tuple

# Load closing prices from CSV file.
def load_close_prices(filename: str) -> List[float]:
    prices: List[float] = []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prices.append(float(row["Close"]))
            except (KeyError, ValueError):
                continue
    return prices

# Create lagged dataset.
def create_lagged_dataset(series: List[float], n_lags: int) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []
    y: List[float] = []
    for i in range(len(series) - n_lags):
        X.append(series[i : i + n_lags])
        y.append(series[i + n_lags])
    return X, y

# Euclidean distance between two points.
def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# k-NN prediction.
def knn_predict(train_X: List[List[float]], train_y: List[float], query: List[float], k: int = 3) -> float:
    distances = [(euclidean(train_X[i], query), train_y[i]) for i in range(len(train_X))]
    distances.sort(key=lambda t: t[0])
    neighbors = distances[:k]
    return sum(val for _, val in neighbors) / k

# Mean Squared Error.
def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Main execution
def main() -> None:
    base_dir = os.path.dirname(__file__)
    series = load_close_prices(os.path.join(base_dir, "tesla.csv"))
    n_lags = 5
    X, y = create_lagged_dataset(series, n_lags)
    split = int(0.8 * len(X))
    train_X, train_y = X[:split], y[:split]
    test_X, test_y = X[split:], y[split:]

    # Evaluate on test set.
    preds = [knn_predict(train_X, train_y, row) for row in test_X]
    mse = mean_squared_error(test_y, preds)
    print(f"Test MSE: {mse:.4f}")

    # Forecast next 5 days
    window = series[-n_lags:]
    forecast: List[float] = []
    for _ in range(5):
        next_val = knn_predict(train_X, train_y, window)
        forecast.append(next_val)
        window = window[1:] + [next_val]
    
    # Round forecast values for better readability.
    rounded = [round(f, 2) for f in forecast]
    print("5-day forecast:", rounded)

# Run the main function.
if __name__ == "__main__":
    main()