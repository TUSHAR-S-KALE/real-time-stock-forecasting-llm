from fastapi import FastAPI, HTTPException
from backend.services.data_service import (
    get_live_price,
    get_historical_data
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
def price(ticker: str):
    data = get_live_price(ticker)

    if data is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    return data


@app.get("/history/{ticker}")
def history(ticker: str, period: str = "1mo"):
    data = get_historical_data(ticker, period)

    if data is None:
        raise HTTPException(status_code=404, detail="Ticker not found")

    return {
        "ticker": ticker.upper(),
        "period": period,
        "data": data
    }
