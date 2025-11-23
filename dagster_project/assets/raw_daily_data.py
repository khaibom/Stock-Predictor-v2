from datetime import date

import yfinance as yf
import pandas as pd

from dagster import asset, Output, Shape, Field
from .methods.save_data import save_data
from .methods.logging import log_df

start = '2010-01-01'
end = date.today()
config_schema = Shape({
    'start_date': Field(str, default_value=str(start), description='Start date in "YYYY-MM-DD"'),
    'end_date': Field(str, default_value=str(end), description='End date in "YYYY-MM-DD"'),
    'ticker': Field(str, default_value='NVD.DE', description='Yahoo Finance ticker symbol'),
    'interval': Field(str, default_value='1d', description='Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo'),
})

@asset(
    name="asset_market_raw",
    group_name="raw_daily_data",
    kinds={"python"},
    config_schema=config_schema,
)
def asset_market_raw(context):
    config = context.op_config
    start_date = config.get("start_date", str(start))
    end_date = config.get("end_date", str(end))
    ticker = config.get("ticker")
    interval = config.get("interval", "1d")

    def download_daily_data(tk=ticker, sd=start_date, ed=end_date, intv=interval):
        # Check for yfinance API limits
        if intv in ['1m', '2m', '5m', '15m', '30m']:
            context.log.warning(f"Note: yfinance limits minute-level data to last 60 days")
        elif intv in ['1h', '60m', '90m']:
            context.log.warning(f"Note: yfinance limits hourly data to approximately last 730 days (~2 years)")
            context.log.warning(f"   Requested: {sd} to {ed}")
            
        df = yf.download(tickers=tk, start=sd, end=ed, interval=intv, auto_adjust=False)

        # Flatten column names if grouped by ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Standardize column names to lowercase / snake_case
        # yfinance returns "Date" for daily data and "Datetime" for intraday data
        # We normalize both to "datetime" for consistency and accuracy
        rename_map = {
            "Date": "datetime",      # Daily data (time will be 00:00:00)
            "Datetime": "datetime",  # Intraday data (preserves time information)
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)
        
        # Ensure we have a 'datetime' column (could be from Date or Datetime)
        if 'datetime' not in df.columns:
            context.log.error(f"No date/datetime column found. Available columns: {df.columns.tolist()}")
            raise ValueError("Expected 'Date' or 'Datetime' column from yfinance data")
        
        df = df[["datetime", "open", "high", "low", "close", "adj_close", "volume"]]
        return df

    def clean_data(df):
        context.log.info(f"Initial columns: {df.columns.tolist()}")

        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    context.log.info(f"Converting column: {col}")
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    context.log.error(f"Error converting column {col}: {e}")
                    raise e

        # Convert datetime column (preserves time information for intraday, sets 00:00:00 for daily)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        context.log.info(f"Missing values per column:\n {df.isnull().sum()}")

        df_cleaned = df.dropna()
        if "close" in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned["close"] > 1]

        context.log.info(f"Remaining rows after cleaning: {len(df_cleaned)}")
        return df_cleaned

    raw_df = download_daily_data()
    cleaned_df = clean_data(raw_df)
    
    context.log.info(f"Downloaded {len(cleaned_df)} rows of {interval} data for {ticker}")

    log_df(cleaned_df, context, 'asset_market_raw')
    save_data(df=cleaned_df,
              filename=f"{ticker}_{interval}.csv",
              dir=f"data/raw/{ticker}",
              context=context,
              asset="asset_market_raw"
              )
    return Output((cleaned_df, ticker, interval),
                  metadata={"num_rows": cleaned_df.shape[0],
                            "num_columns": cleaned_df.shape[1],
                            "ticker": ticker,
                            "interval": interval,
                            })

