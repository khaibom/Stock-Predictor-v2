# Script to download and save raw stock data

import yfinance as yf
import pandas as pd
import os

def download_daily_data(ticker="NVDA", start="2010-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)
    
    # Flatten column names if grouped by ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return df

def clean_data(df):
    print("Initial columns:", df.columns.tolist())

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                print(f"Converting column: {col}")
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                print(f"Error converting column {col}: {e}")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    print("Missing values per column:\n", df.isnull().sum())

    df_cleaned = df.dropna()
    if "Close" in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned["Close"] > 1]

    print(f"Remaining rows after cleaning: {len(df_cleaned)}")
    return df_cleaned

def save_data(df, ticker="NVDA"):
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{ticker.lower()}_daily.csv"
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")

if __name__ == "__main__":
    ticker = "NVDA"
    raw_df = download_daily_data(ticker)
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df, ticker)