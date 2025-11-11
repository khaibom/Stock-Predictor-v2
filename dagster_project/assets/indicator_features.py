import pandas as pd
import os

from dagster import asset, Output
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

@asset(
    name="indicator_features",
    group_name="add_features",
    kinds={"python"}
)
def indicator_features(context, lag_features):
    def add_trend_indicators(df):
        # Trend indicators help the model identify **general price direction** (uptrend, downtrend, sideways)
        # over various timeframes. They’re useful for detecting persistent market behavior.

        # EMA (20): Short-term trend line. Reacts faster to price changes.
        ema20 = EMAIndicator(close=df["Adj_Close"], window=20)
        df["EMA_20"] = ema20.ema_indicator()

        # EMA (50): Medium-term trend line. Smoother than EMA 20.
        ema50 = EMAIndicator(close=df["Adj_Close"], window=50)
        df["EMA_50"] = ema50.ema_indicator()

        # MACD: Difference between 12-EMA and 26-EMA; detects trend reversals and momentum shifts.
        macd = MACD(close=df["Adj_Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        return df

    def add_momentum_indicators(df):
        # Momentum indicators measure the **speed and strength** of price movement.
        # They help the model recognize overbought/oversold conditions and potential reversals.

        # RSI (14): Measures recent gains vs. losses; values >70 = overbought, <30 = oversold.
        rsi = RSIIndicator(close=df["Adj_Close"], window=14)
        df["RSI_14"] = rsi.rsi()

        # Stochastic Oscillator: Shows where the current close sits within recent high/low range.
        stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
        df["Stoch_k"] = stoch.stoch()
        df["Stoch_d"] = stoch.stoch_signal()

        # ROC (Rate of Change): Percentage change over time; detects acceleration in price.
        roc = ROCIndicator(close=df["Adj_Close"], window=10)
        df["ROC_10"] = roc.roc()

        return df

    def add_volatility_indicators(df):
        # Volatility indicators measure how much price fluctuates. Volatility gives context to trend strength
        # and helps the model recognize market uncertainty or breakout potential.

        # Bollinger Bands: Creates dynamic upper/lower bands around a moving average.
        bbands = BollingerBands(close=df["Adj_Close"], window=20, window_dev=2)
        df["BB_high"] = bbands.bollinger_hband()
        df["BB_low"] = bbands.bollinger_lband()
        df["BB_mid"] = bbands.bollinger_mavg()

        # Band Width: Simple measure of volatility from BB.
        df["BB_width"] = df["BB_high"] - df["BB_low"]

        # ATR (Average True Range): Measures daily price range variation.
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["ATR_14"] = atr.average_true_range()

        return df

    def add_volume_indicators(df):
        # Volume indicators reflect **trading activity** and help detect the conviction behind price moves.
        # They are important for identifying fakeouts vs. real moves.

        # OBV (On-Balance Volume): Accumulates volume based on price direction; rising OBV = buying pressure.
        obv = OnBalanceVolumeIndicator(close=df["Adj_Close"], volume=df["Volume"])
        df["OBV"] = obv.on_balance_volume()

        # Volume SMA (20): Helps detect unusual volume surges.
        df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()

        # MFI (Money Flow Index): Like RSI but volume-weighted; good for spotting divergence.
        mfi = MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14)
        df["MFI_14"] = mfi.money_flow_index()

        return df

    def add_custom_features(df):
        # How far current price is from EMA 20 (relative).
        df["Close_to_EMA20"] = df["Adj_Close"] / df["EMA_20"] - 1

        # Intraday spread relative to closing price.
        df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]
        return df

    def save_features(df, ticker="NVDA"):
        path = f"data/processed/{ticker.lower()}_features.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved indicator feature data to: {path}")

    ticker = "NVDA"
    path = f"data/processed/{ticker.lower()}_daily_lagged.csv"
    df = lag_features

    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_custom_features(df)

    # Drop initial rows with NaN ()
    df = df.dropna().reset_index(drop=True)

    save_features(df, ticker)
    context.log.info(df.info())
    context.log.info(df.head())
    context.log.info(df.tail())
    return Output(df,
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            })