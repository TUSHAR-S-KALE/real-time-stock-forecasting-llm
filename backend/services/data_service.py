import yfinance as yf
import pandas as pd
from datetime import datetime

def get_live_price(ticker: str):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")

    if data.empty:
        return None

    latest_price = data["Close"].iloc[-1]

    return {
        "ticker": ticker.upper(),
        "price": round(float(latest_price), 2),
        "timestamp": datetime.now().isoformat()
    }

def get_historical_data(ticker: str, period: str = "1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        return None

    data.reset_index(inplace=True)

    result = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    return result.to_dict(orient="records")

