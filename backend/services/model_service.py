import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from backend.services.feature_service import generate_features

MODEL_PATH = "backend/models/stock_model.pkl"


def prepare_data(ticker: str, period: str = "6mo"):
    df = generate_features(ticker, period)

    # Create Target: 1 if next day close > today close else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()

    feature_columns = [
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

    X = df[feature_columns]
    y = df["Target"]

    return X, y


def train_model(ticker: str):
    X, y = prepare_data(ticker)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Model trained successfully",
        "accuracy": round(accuracy, 4)
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_next_day(ticker: str):
    model = load_model()
    if model is None:
        return {"error": "Model not trained yet"}

    X, _ = prepare_data(ticker)

    latest_data = X.tail(1)

    prediction = model.predict(latest_data)[0]
    probability = model.predict_proba(latest_data)[0]

    return {
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": round(max(probability), 4)
    }
