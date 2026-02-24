from fastapi import FastAPI, HTTPException
from backend.services.feature_service import generate_features
from backend.services.model_service import train_model, predict_next_day
from backend.services.data_service import (
    get_live_price,
    get_historical_data
)
from backend.services.evaluation_service import (
    evaluate_model,
    backtest_strategy,
    feature_importance
)

app = FastAPI(
    title="Stock AI API",
    description="Live stock data ingestion using Yahoo Finance",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Stock AI API is running"}


@app.get("/price/{ticker}")
def price(ticker):
    data = get_live_price(ticker)

    if data is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    return data


@app.get("/history/{ticker}")
def history(ticker, period = "6mo"):
    data = get_historical_data(ticker, period)

    if data is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    return {
        "ticker": ticker.upper(),
        "period": period,
        "data": data
    }

@app.get("/features/{ticker}")
def features(ticker, period = "6mo"):
    df = generate_features(ticker, period)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Could not generate features")

    return {
        "ticker": ticker.upper(),
        "period": period,
        "rows": len(df),
        "columns": list(df.columns),
        "data_sample": df.tail(5).to_dict(orient="records")
    }

@app.get("/train/{ticker}")
def train(ticker: str, period = "6mo"):
    return train_model(ticker, period)

@app.get("/predict/{ticker}")
def predict(ticker: str, period = "6mo"):
    return predict_next_day(ticker, period)

@app.get("/evaluate/{ticker}")
def evaluate(ticker: str):
    return evaluate_model(ticker)

@app.get("/backtest/{ticker}")
def backtest(ticker: str):
    return backtest_strategy(ticker)

@app.get("/feature-importance")
def importance():
    return feature_importance()