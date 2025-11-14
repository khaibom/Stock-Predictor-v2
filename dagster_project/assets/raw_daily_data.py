from datetime import date

import yfinance as yf
import pandas as pd
import os

from dagster import asset, Output, Shape, Field

start = '2020-01-01'
end = date.today()
date_config_schema = Shape({
    'start_date': Field(str, default_value=str(start), description='Start date in "YYYY-MM-DD"'),
    'end_date': Field(str, default_value=str(end), description='End date in "YYYY-MM-DD"')
})

@asset(
    name="asset_market_raw",
    group_name="raw_daily_data",
    kinds={"python"},
    config_schema=date_config_schema,
)
def asset_market_raw(context):
    config = context.op_config
    start_date = config.get("start_date", str(start))
    end_date = config.get("end_date", str(end))
    def download_daily_data(ticker="NVDA", start=start_date, end=end_date):
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)

        # Flatten column names if grouped by ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        return df

    def clean_data(df):
        context.log.info(f"Initial columns: {df.columns.tolist()}")

        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    context.log.info(f"Converting column: {col}")
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    context.log.error(f"Error converting column {col}: {e}")
                    raise e

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        context.log.info(f"Missing values per column:\n {df.isnull().sum()}")

        df_cleaned = df.dropna()
        if "Close" in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned["Close"] > 1]

        context.log.info(f"Remaining rows after cleaning: {len(df_cleaned)}")
        return df_cleaned

    def save_data(df, ticker="NVDA"):
        os.makedirs("data/raw", exist_ok=True)
        path = f"data/raw/{ticker.lower()}_daily.csv"
        df.to_csv(path, index=False)
        context.log.info(f"Saved cleaned data to {path}")

    ticker = "NVDA"
    raw_df = download_daily_data(ticker)
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df, ticker)

    context.log.info(cleaned_df.info())
    context.log.info(cleaned_df.head())
    context.log.info(cleaned_df.tail())
    return Output(cleaned_df,
                  metadata={"num_rows": cleaned_df.shape[0],
                            "num_columns": cleaned_df.shape[1],
                            })

