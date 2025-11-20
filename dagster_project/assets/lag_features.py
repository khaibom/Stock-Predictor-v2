import os

from dagster import asset, Output
from .methods.save_data import save_data
from .methods.logging import log_df

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

    # Load cleaned data
    df, ticker = asset_market_raw

    # Add features
    df = add_daily_return(df)
    df = add_lag_features(df, lags=[1, 2, 3], cols=["adj_close", "daily_return"])

    # Drop initial rows with NaN (due to lagging)
    df = df.dropna().reset_index(drop=True)

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