import os

from dagster import asset, Output
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

from .methods.save_data import save_data
from .methods.logging import log_df

@asset(
    name="asset_features_full",
    group_name="add_features",
    kinds={"python"}
)
def asset_features_full(context, asset_features_lagged):
    def add_trend_indicators(df):
        # Trend indicators help the model identify **general price direction** (uptrend, downtrend, sideways)
        # over various timeframes. They’re useful for detecting persistent market behavior.

        # EMA (20): Short-term trend line. Reacts faster to price changes.
        ema20 = EMAIndicator(close=df["adj_close"], window=20)
        df["EMA_20"] = ema20.ema_indicator()

        # EMA (50): Medium-term trend line. Smoother than EMA 20.
        ema50 = EMAIndicator(close=df["adj_close"], window=50)
        df["EMA_50"] = ema50.ema_indicator()

        # MACD: Difference between 12-EMA and 26-EMA; detects trend reversals and momentum shifts.
        macd = MACD(close=df["adj_close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        return df

    def add_momentum_indicators(df):
        # Momentum indicators measure the **speed and strength** of price movement.
        # They help the model recognize overbought/oversold conditions and potential reversals.

        # RSI (14): Measures recent gains vs. losses; values >70 = overbought, <30 = oversold.
        rsi = RSIIndicator(close=df["adj_close"], window=14)
        df["RSI_14"] = rsi.rsi()

        # Stochastic Oscillator: Shows where the current close sits within recent high/low range.
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        df["Stoch_k"] = stoch.stoch()
        df["Stoch_d"] = stoch.stoch_signal()

        # ROC (Rate of Change): Percentage change over time; detects acceleration in price.
        roc = ROCIndicator(close=df["adj_close"], window=10)
        df["ROC_10"] = roc.roc()

        return df

    def add_volatility_indicators(df):
        # Volatility indicators measure how much price fluctuates. Volatility gives context to trend strength
        # and helps the model recognize market uncertainty or breakout potential.

        # Bollinger Bands: Creates dynamic upper/lower bands around a moving average.
        bbands = BollingerBands(close=df["adj_close"], window=20, window_dev=2)
        df["BB_high"] = bbands.bollinger_hband()
        df["BB_low"] = bbands.bollinger_lband()
        df["BB_mid"] = bbands.bollinger_mavg()

        # Band Width: Simple measure of volatility from BB.
        df["BB_width"] = df["BB_high"] - df["BB_low"]

        # ATR (Average True Range): Measures daily price range variation.
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["ATR_14"] = atr.average_true_range()

        return df

    def add_volume_indicators(df):
        # Volume indicators reflect **trading activity** and help detect the conviction behind price moves.
        # They are important for identifying fakeouts vs. real moves.

        # OBV (On-Balance Volume): Accumulates volume based on price direction; rising OBV = buying pressure.
        obv = OnBalanceVolumeIndicator(close=df["adj_close"], volume=df["volume"])
        df["OBV"] = obv.on_balance_volume()

        # Volume SMA (20): Helps detect unusual volume surges.
        df["Volume_SMA_20"] = df["volume"].rolling(window=20).mean()

        # MFI (Money Flow Index): Like RSI but volume-weighted; good for spotting divergence.
        mfi = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14)
        df["MFI_14"] = mfi.money_flow_index()

        return df

    def add_custom_features(df):
        # How far current price is from EMA 20 (relative).
        df["Close_to_EMA20"] = df["adj_close"] / df["EMA_20"] - 1

        # Intraday spread relative to closing price.
        df["High_Low_Spread"] = (df["high"] - df["low"]) / df["close"]
        return df

    df, ticker = asset_features_lagged

    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_custom_features(df)

    # Drop initial rows with NaN ()
    df = df.dropna().reset_index(drop=True)

    log_df(df, context, 'asset_features_full')
    save_data(df=df,
              filename=f"{ticker}_features.csv",
              dir="data/processed",
              context=context,
              asset="asset_features_full"
              )
    return Output((df, ticker),
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            "ticker": ticker,
                            })