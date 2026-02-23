import pandas as pd
import numpy as np
import pandas_ta as ta
from backend.services.data_service import get_historical_data


def create_lag_features(df, lags = 5):
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    return df


def add_technical_indicators(df):

    # SMA
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    #df["SMA_200"] = ta.sma(df["Close"], length=200)

    # EMA
    df["EMA_20"] = ta.ema(df["Close"], length=20)

    # RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    if macd is not None:
        df["MACD"] = macd.iloc[:, 0]
        df["MACD_signal"] = macd.iloc[:, 1]

    # Bollinger Bands (SAFE VERSION)
    bbands = ta.bbands(df["Close"], length=20)

    if bbands is not None:
        df["BB_lower"] = bbands.iloc[:, 0]
        df["BB_middle"] = bbands.iloc[:, 1]
        df["BB_upper"] = bbands.iloc[:, 2]

    return df


def add_volatility(df):
    df["returns"] = df["Close"].pct_change()    #calculates fractional change/relative or per unit change
    df["volatility_20"] = df["returns"].rolling(window=20).std()
    return df

def generate_features(ticker, period= "6mo"):
    raw_data = get_historical_data(ticker, period)

    if raw_data is None:
        return None

    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    #Adding features
    df = add_technical_indicators(df)
    df = create_lag_features(df, lags=10)
    df = add_volatility(df)

    #Dropping NaN rows caused by indicators
    df.dropna(inplace=True)

    return df

