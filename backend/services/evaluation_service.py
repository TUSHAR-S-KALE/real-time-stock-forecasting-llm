import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from backend.services.model_service import prepare_data, load_model


def evaluate_model(ticker: str):
    model = load_model()
    if model is None:
        return {"error": "Train model first."}

    X, y = prepare_data(ticker)

    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, predictions).tolist(), 
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"]).to_dict(orient="index")
    }

    return metrics


def backtest_strategy(ticker: str):
    model = load_model()
    if model is None:
        return {"error": "Train model first."}

    X, y = prepare_data(ticker)

    split_index = int(len(X) * 0.8)
    X_test = X[split_index:]

    predictions = model.predict(X_test)

    df = X_test.copy()
    df["Prediction"] = predictions

    # Strategy Return: Buy only when prediction = 1
    df["Market_Return"] = y[split_index:].values
    df["Strategy_Return"] = df["Prediction"] * df["Market_Return"]

    strategy_accuracy = accuracy_score(y[split_index:], predictions)

    total_trades = int(df["Prediction"].sum())

    return {
        "strategy_accuracy": round(strategy_accuracy, 4),
        "total_trades": total_trades,
        "buy_signals": int(df["Prediction"].sum()),
        "sell_signals": int((df["Prediction"] == 0).sum())
    }


def feature_importance():
    model = load_model()
    if model is None:
        return {"error": "Train model first."}

    feature_names = [
        "SMA_20",
        "EMA_20",
        "SMA_50",
        #"SMA_200",
        "RSI_14",
        "MACD",
        "MACD_signal",
        "BB_upper",
        "BB_lower"
    ]

    importances = model.feature_importances_

    importance_dict = dict(
        sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return importance_dict