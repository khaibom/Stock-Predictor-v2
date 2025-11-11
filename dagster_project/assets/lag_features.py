# lag_features.py

import pandas as pd
import os

def add_lag_features(df, lags=[1, 2, 3], cols=["Adj_Close", "Daily_Return"]):
    """
    Add lagged versions of selected columns.
    For example, for lag=1 and col="Adj_Close", adds new column "Adj_Close_Lag1".

    Parameters:
        df: DataFrame with stock data
        lags: list of lag values (e.g. [1, 2, 3])
        cols: list of columns to lag
    Returns:
        df: DataFrame with new lagged features
    """
    for col in cols:
        for lag in lags:
            lag_col_name = f"{col}_Lag{lag}"
            df[lag_col_name] = df[col].shift(lag)
    return df

def add_daily_return(df):
    df["Daily_Return"] = df["Adj_Close"].pct_change()
    return df

def save_lagged_data(df, ticker="NVDA"):
    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{ticker.lower()}_daily_lagged.csv"
    df.to_csv(path, index=False)
    print(f"Saved lagged feature data to {path}")

if __name__ == "__main__":
    # Load cleaned data
    ticker = "NVDA"
    path = f"data/raw/{ticker.lower()}_daily.csv"
    df = pd.read_csv(path, parse_dates=["Date"])

    df = df.rename(columns={"Adj Close": "Adj_Close"})

    # Add features
    df = add_daily_return(df)
    df = add_lag_features(df, lags=[1, 2, 3], cols=["Adj_Close", "Daily_Return"])

    # Drop initial rows with NaN (due to lagging)
    df = df.dropna().reset_index(drop=True)

    # Save
    save_lagged_data(df, ticker)
