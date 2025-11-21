import numpy as np

from dagster import asset, Output
from .methods.save_data import save_data
from .methods.logging import log_df

@asset(
    name="asset_features_lagged",
    group_name="add_features",
    kinds={"python"}
)
def asset_features_lagged(context, asset_market_raw):
    def add_lag_features(df, lags=[1, 2, 3, 5, 7, 10], cols=["adj_close", "daily_return", "volume", "high", "low"]):
        """
        Add lagged versions of selected columns.
        EXPANDED: More lags (up to 10 days) and more columns.
        
        Parameters:
            df: DataFrame with stock data
            lags: list of lag values (e.g. [1, 2, 3, 5, 7, 10])
            cols: list of columns to lag
        Returns:
            df: DataFrame with new lagged features
        """
        for col in cols:
            for lag in lags:
                lag_col_name = f"{col}_lag{lag}"
                df[lag_col_name] = df[col].shift(lag)
        return df

    def add_daily_return(df):
        df["daily_return"] = df["adj_close"].pct_change()
        return df
    
    def add_multi_period_returns(df):
        """
        Add returns over multiple periods: 2d, 3d, 5d, 10d, 20d.
        Critical for capturing different timeframe momentum.
        """
        for period in [2, 3, 5, 10, 20]:
            df[f"return_{period}d"] = df["adj_close"].pct_change(periods=period)
        return df
    
    def add_rolling_statistics(df):
        """
        Add rolling statistics: mean, std, min, max over different windows.
        These capture local trends and volatility that LSTM needs.
        """
        for window in [5, 10, 20]:
            # Price statistics
            df[f"close_mean_{window}d"] = df["adj_close"].rolling(window=window).mean()
            df[f"close_std_{window}d"] = df["adj_close"].rolling(window=window).std()
            df[f"close_min_{window}d"] = df["adj_close"].rolling(window=window).min()
            df[f"close_max_{window}d"] = df["adj_close"].rolling(window=window).max()
            
            # Return statistics
            df[f"return_mean_{window}d"] = df["daily_return"].rolling(window=window).mean()
            df[f"return_std_{window}d"] = df["daily_return"].rolling(window=window).std()
            
            # Volume statistics
            df[f"volume_mean_{window}d"] = df["volume"].rolling(window=window).mean()
            df[f"volume_std_{window}d"] = df["volume"].rolling(window=window).std()
        
        return df
    
    def add_price_position_features(df):
        """
        Add features showing where current price sits relative to recent ranges.
        Critical for identifying breakouts and reversals.
        """
        for window in [5, 10, 20]:
            # Position within recent range (0 = at low, 1 = at high)
            close_min = df["adj_close"].rolling(window=window).min()
            close_max = df["adj_close"].rolling(window=window).max()
            df[f"price_position_{window}d"] = (df["adj_close"] - close_min) / (close_max - close_min + 1e-8)
            
            # Distance from recent high/low (as percentage) - ADD EPSILON TO PREVENT DIV BY ZERO
            df[f"pct_from_high_{window}d"] = (df["adj_close"] - close_max) / (close_max + 1e-8)
            df[f"pct_from_low_{window}d"] = (df["adj_close"] - close_min) / (close_min + 1e-8)
        
        return df
    
    def add_volume_features(df):
        """
        Add volume-based features.
        Volume often leads price changes.
        """
        df["volume_change"] = df["volume"].pct_change()
        df["volume_change_5d"] = df["volume"].pct_change(periods=5)
        
        # Volume ratio to recent average
        for window in [5, 10, 20]:
            vol_avg = df["volume"].rolling(window=window).mean()
            df[f"volume_ratio_{window}d"] = df["volume"] / (vol_avg + 1e-8)
        
        return df
    
    def add_volatility_features(df):
        """
        Add volatility measures beyond ATR.
        """
        for window in [5, 10, 20]:
            # High-low range as % of close
            df[f"hl_pct_{window}d"] = (
                (df["high"] - df["low"]).rolling(window=window).mean() / df["adj_close"]
            )
        
        # Realized volatility (std of returns)
        df["realized_vol_10d"] = df["daily_return"].rolling(window=10).std() * np.sqrt(252)
        df["realized_vol_20d"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)
        
        return df

    # Load cleaned data
    df, ticker = asset_market_raw

    # Add all features
    df = add_daily_return(df)
    df = add_multi_period_returns(df)
    df = add_rolling_statistics(df)
    df = add_price_position_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_lag_features(df, lags=[1, 2, 3, 5, 7, 10], cols=["adj_close", "daily_return", "volume", "high", "low"])

    # Clean up any infinity or extreme values that might have been created
    # Replace inf/-inf with NaN, then drop them
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop initial rows with NaN (due to lagging/rolling/inf values)
    df = df.dropna().reset_index(drop=True)

    print(f"\n[asset_features_lagged] Shape: {df.shape} | Columns & Types:\n{df.dtypes}\n")

    log_df(df, context, 'asset_features_lagged')
    save_data(df=df,
              filename=f"{ticker}_daily_lagged.csv",
              dir=f"data/processed/{ticker}",
              context=context,
              asset="asset_features_lagged"
              )
    return Output((df, ticker),
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            "ticker": ticker,
                            })