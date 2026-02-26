import yfinance as yf
import pandas as pd
from cachetools import TTLCache

#Cache for 5 minutes
price_cache = TTLCache(maxsize=100, ttl=300)
history_cache = TTLCache(maxsize=100, ttl=300)


def get_live_price(ticker: str):
    if ticker in price_cache:
        return price_cache[ticker]

    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")

    if data.empty:
        return None

    price = float(data["Close"].iloc[-1])

    result = {
        "ticker": ticker,
        "price": price
    }

    price_cache[ticker] = result
    return result

def get_historical_data(ticker: str, period: str = "6mo"):
    cache_key = f"{ticker}_{period}"

    if cache_key in history_cache:
        return history_cache[cache_key]

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        return None

    df.reset_index(inplace=True)
    df["Date"] = df["Date"].astype(str)

    result = df.to_dict(orient="records")

    history_cache[cache_key] = result
    return result
