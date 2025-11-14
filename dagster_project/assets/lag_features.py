import os

from dagster import asset, Output


@asset(
    name="asset_features_lagged",
    group_name="add_features",
    kinds={"python"}
)
def asset_features_lagged(context, asset_market_raw):
    def add_lag_features(df, lags=[1, 2, 3], cols=["adj_close", "daily_return"]):
        """
        Add lagged versions of selected columns.
        For example, for lag=1 and col="adj_close", adds new column "adj_close_lag1".

        Parameters:
            df: DataFrame with stock data
            lags: list of lag values (e.g. [1, 2, 3])
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

    def save_lagged_data(df, ticker="NVDA"):
        os.makedirs("data/processed", exist_ok=True)
        path = f"data/processed/{ticker.lower()}_daily_lagged.csv"
        df.to_csv(path, index=False)
        context.log.info(f"Saved lagged feature data to {path}")

    # Load cleaned data
    ticker = "NVDA"
    path = f"data/raw/{ticker.lower()}_daily.csv"
    df = asset_market_raw

    # Add features
    df = add_daily_return(df)
    df = add_lag_features(df, lags=[1, 2, 3], cols=["adj_close", "daily_return"])

    # Drop initial rows with NaN (due to lagging)
    df = df.dropna().reset_index(drop=True)

    # Save
    save_lagged_data(df, ticker)

    context.log.info(df.info())
    context.log.info(df.head())
    context.log.info(df.tail())
    return Output(df,
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            })