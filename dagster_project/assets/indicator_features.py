import os
import numpy as np

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
    def get_indicator_windows(interval: str):
        """
        Get interval-aware window sizes for technical indicators.
        Returns appropriate window sizes based on the data interval.
        """
        if interval.endswith('m'):
            # Minutes: 1m, 2m, 5m, 15m, 30m, 60m, 90m
            minutes = int(interval[:-1])
            bars_per_hour = 60 // minutes
            bars_per_day = 390 // minutes  # 390 minutes in trading day
            # Technical indicators adjusted for intraday
            return {
                'ema_short': bars_per_day,  # ~1 day
                'ema_long': bars_per_day * 3,  # ~3 days
                'rsi': bars_per_day,  # ~1 day
                'stoch': bars_per_day,  # ~1 day
                'roc': bars_per_hour * 5,  # ~5 hours
                'bb': bars_per_day,  # ~1 day
                'atr': bars_per_day,  # ~1 day
                'volume_sma': bars_per_day,  # ~1 day
                'mfi': bars_per_day,  # ~1 day
                'macd_fast': bars_per_hour * 6,  # ~6 hours
                'macd_slow': bars_per_day,  # ~1 day
                'macd_signal': bars_per_hour * 4,  # ~4 hours
            }
        elif interval in ['1h', '60m', '90m']:
            # Hourly data
            return {
                'ema_short': 13,  # ~2 days
                'ema_long': 39,  # ~6 days
                'rsi': 13,
                'stoch': 13,
                'roc': 7,
                'bb': 13,
                'atr': 13,
                'volume_sma': 13,
                'mfi': 13,
                'macd_fast': 8,
                'macd_slow': 17,
                'macd_signal': 6,
            }
        elif interval == '1d':
            # Daily data (original behavior)
            return {
                'ema_short': 20,
                'ema_long': 50,
                'rsi': 14,
                'stoch': 14,
                'roc': 10,
                'bb': 20,
                'atr': 14,
                'volume_sma': 20,
                'mfi': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
            }
        elif interval in ['5d', '1wk']:
            # Weekly data
            return {
                'ema_short': 4,  # ~1 month
                'ema_long': 10,  # ~2.5 months
                'rsi': 3,
                'stoch': 3,
                'roc': 2,
                'bb': 4,
                'atr': 3,
                'volume_sma': 4,
                'mfi': 3,
                'macd_fast': 3,
                'macd_slow': 6,
                'macd_signal': 2,
            }
        elif interval in ['1mo', '3mo']:
            # Monthly data
            return {
                'ema_short': 3,  # ~3 months
                'ema_long': 6,  # ~6 months
                'rsi': 3,
                'stoch': 3,
                'roc': 2,
                'bb': 3,
                'atr': 3,
                'volume_sma': 3,
                'mfi': 3,
                'macd_fast': 2,
                'macd_slow': 4,
                'macd_signal': 2,
            }
        else:
            # Default to daily parameters
            return {
                'ema_short': 20,
                'ema_long': 50,
                'rsi': 14,
                'stoch': 14,
                'roc': 10,
                'bb': 20,
                'atr': 14,
                'volume_sma': 20,
                'mfi': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
            }
    
    def add_trend_indicators(df, windows):
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

    def add_momentum_indicators(df, windows):
        # Momentum indicators measure the **speed and strength** of price movement.
        # They help the model recognize overbought/oversold conditions and potential reversals.

        # RSI: Measures recent gains vs. losses; values >70 = overbought, <30 = oversold.
        rsi = RSIIndicator(close=df["adj_close"], window=windows['rsi'])
        df["RSI_14"] = rsi.rsi()

        # Stochastic Oscillator: Shows where the current close sits within recent high/low range.
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], 
                                      window=windows['stoch'], smooth_window=3)
        df["Stoch_k"] = stoch.stoch()
        df["Stoch_d"] = stoch.stoch_signal()

        # ROC (Rate of Change): Percentage change over time; detects acceleration in price.
        roc = ROCIndicator(close=df["adj_close"], window=windows['roc'])
        df["ROC_10"] = roc.roc()

        return df

    def add_volatility_indicators(df, windows):
        # Volatility indicators measure how much price fluctuates. Volatility gives context to trend strength
        # and helps the model recognize market uncertainty or breakout potential.

        # Bollinger Bands: Creates dynamic upper/lower bands around a moving average.
        bbands = BollingerBands(close=df["adj_close"], window=windows['bb'], window_dev=2)
        df["BB_high"] = bbands.bollinger_hband()
        df["BB_low"] = bbands.bollinger_lband()
        df["BB_mid"] = bbands.bollinger_mavg()

        # Band Width: Simple measure of volatility from BB.
        df["BB_width"] = df["BB_high"] - df["BB_low"]

        # ATR (Average True Range): Measures price range variation (interval-aware).
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=windows['atr'])
        df["ATR_14"] = atr.average_true_range()

        return df

    def add_volume_indicators(df, windows):
        # Volume indicators reflect **trading activity** and help detect the conviction behind price moves.
        # They are important for identifying fakeouts vs. real moves.

        # OBV (On-Balance Volume): Accumulates volume based on price direction; rising OBV = buying pressure.
        obv = OnBalanceVolumeIndicator(close=df["adj_close"], volume=df["volume"])
        df["OBV"] = obv.on_balance_volume()

        # Volume SMA: Helps detect unusual volume surges.
        df["Volume_SMA_20"] = df["volume"].rolling(window=windows['volume_sma']).mean()

        # MFI (Money Flow Index): Like RSI but volume-weighted; good for spotting divergence.
        mfi = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], 
                          volume=df["volume"], window=windows['mfi'])
        df["MFI_14"] = mfi.money_flow_index()

        return df

    def add_custom_features(df):
        """
        Add custom engineered features and interactions.
        These often provide the most predictive power.
        """
        # Price relative to EMAs - ADD EPSILON TO PREVENT DIV BY ZERO
        df["Close_to_EMA20"] = df["adj_close"] / (df["EMA_20"] + 1e-8) - 1
        df["Close_to_EMA50"] = df["adj_close"] / (df["EMA_50"] + 1e-8) - 1
        df["EMA20_to_EMA50"] = df["EMA_20"] / (df["EMA_50"] + 1e-8) - 1  # Golden/Death cross signal
        
        # Intraday spread relative to closing price - ADD EPSILON
        df["High_Low_Spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        
        # Bollinger Band position (where price is within bands)
        df["BB_position"] = (df["adj_close"] - df["BB_low"]) / (df["BB_high"] - df["BB_low"] + 1e-8)
        df["BB_width_pct"] = df["BB_width"] / (df["BB_mid"] + 1e-8)  # Normalized BB width
        
        # Price momentum (ROC) relative to volatility (ATR)
        # Strong moves with low volatility are more significant
        df["ROC_to_ATR"] = df["ROC_10"] / (df["ATR_14"] / (df["adj_close"] + 1e-8) + 1e-8)
        
        # Volume-Price divergence
        # When volume increases but price doesn't = potential reversal
        df["volume_price_trend"] = df["OBV"] / (df["adj_close"] + 1e-8)
        
        # RSI momentum (change in RSI)
        df["RSI_change"] = df["RSI_14"].diff()
        df["RSI_slope_5d"] = df["RSI_14"].diff(5)
        
        # MACD momentum
        df["MACD_acceleration"] = df["MACD_diff"].diff()  # Second derivative
        df["MACD_to_signal_ratio"] = df["MACD"] / (df["MACD_signal"] + 1e-8)
        
        # Stochastic momentum
        df["Stoch_K_D_diff"] = df["Stoch_k"] - df["Stoch_d"]  # Crossover signal
        df["Stoch_change"] = df["Stoch_k"].diff()
        
        # Price efficiency ratio (trending vs ranging)
        # |net price change| / sum(abs daily changes)
        window = 10
        net_change = df["adj_close"].diff(window).abs()
        sum_changes = df["adj_close"].diff().abs().rolling(window=window).sum()
        df["efficiency_ratio_10d"] = net_change / (sum_changes + 1e-8)
        
        return df
    
    def add_advanced_indicators(df, windows):
        """
        Add more advanced technical indicators that are often predictive.
        """
        # Average Directional Index (ADX) - measures trend strength
        from ta.trend import ADXIndicator
        adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=windows['atr'])
        df["ADX_14"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()
        
        # Commodity Channel Index (CCI) - identifies cyclical trends
        from ta.trend import CCIIndicator
        cci = CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=windows['bb'])
        df["CCI_20"] = cci.cci()
        
        # Williams %R - momentum indicator
        from ta.momentum import WilliamsRIndicator
        willr = WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=windows['rsi'])
        df["WillR_14"] = willr.williams_r()
        
        # Keltner Channels - volatility bands (alternative to Bollinger)
        from ta.volatility import KeltnerChannel
        kc = KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=windows['bb'])
        df["KC_high"] = kc.keltner_channel_hband()
        df["KC_low"] = kc.keltner_channel_lband()
        df["KC_mid"] = kc.keltner_channel_mband()
        df["KC_width"] = df["KC_high"] - df["KC_low"]
        df["KC_position"] = (df["adj_close"] - df["KC_low"]) / (df["KC_high"] - df["KC_low"] + 1e-8)
        
        # Donchian Channels - breakout indicator
        from ta.volatility import DonchianChannel
        dc = DonchianChannel(high=df["high"], low=df["low"], close=df["close"], window=windows['bb'])
        df["DC_high"] = dc.donchian_channel_hband()
        df["DC_low"] = dc.donchian_channel_lband()
        df["DC_mid"] = dc.donchian_channel_mband()
        
        # Ultimate Oscillator - combines multiple timeframes
        from ta.momentum import UltimateOscillator
        uo = UltimateOscillator(high=df["high"], low=df["low"], close=df["close"])
        df["UO"] = uo.ultimate_oscillator()
        
        # Chaikin Money Flow - volume-weighted price momentum
        from ta.volume import ChaikinMoneyFlowIndicator
        cmf = ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], 
                                       volume=df["volume"], window=windows['bb'])
        df["CMF_20"] = cmf.chaikin_money_flow()
        
        return df

    df, ticker, interval = asset_features_lagged
    
    context.log.info(f"Adding technical indicators for interval: {interval}")
    context.log.info(f"Input dataframe shape: {df.shape}")

    # Check if we have enough data
    if len(df) < 100:
        context.log.warning(f"WARNING: Only {len(df)} rows available after lag features!")
        context.log.warning("This may cause issues with technical indicators")
        context.log.warning("Consider:")
        context.log.warning("  1. Using a shorter time range")
        context.log.warning("  2. Using a longer interval (e.g., 4h instead of 1h)")
        context.log.warning("  3. Reducing lookback windows")
    
    if len(df) == 0:
        raise ValueError(f"No data remaining after lag features for {ticker} at {interval} interval. "
                        f"The rolling windows are too large for the available data.")

    # Get interval-aware indicator windows
    windows = get_indicator_windows(interval)
    context.log.info(f"Using indicator windows: {windows}")
    
    # Check if windows are too large for available data
    max_window = max(windows.values())
    if len(df) < max_window * 2:
        context.log.error(f"Not enough data! Have {len(df)} rows but need at least {max_window * 2}")
        context.log.error(f"Largest window: {max_window}")
        raise ValueError(
            f"Insufficient data for technical indicators: {len(df)} rows available, "
            f"but need at least {max_window * 2} for largest window ({max_window}). "
            f"Try reducing the date range or using a longer interval."
        )

    df = add_trend_indicators(df, windows)
    df = add_momentum_indicators(df, windows)
    df = add_volatility_indicators(df, windows)
    df = add_volume_indicators(df, windows)
    df = add_advanced_indicators(df, windows)
    df = add_custom_features(df)

    # Clean up any infinity or extreme values that might have been created
    # Replace inf/-inf with NaN, then drop them
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop initial rows with NaN ()
    df = df.dropna().reset_index(drop=True)

    print(f"\n[asset_features_full] Shape: {df.shape} | Columns & Types:\n{df.dtypes}\n")

    log_df(df, context, 'asset_features_full')
    save_data(df=df,
              filename=f"{ticker}_{interval}_features.csv",
              dir=f"data/processed/{ticker}",
              context=context,
              asset="asset_features_full"
              )
    return Output((df, ticker, interval),
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            "ticker": ticker,
                            "interval": interval,
                            })