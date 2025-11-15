from dagster import asset, Field, Output
from .methods.save_data import save_data

reg_config_schema = {
    "days_ahead": Field(int, default_value=1, description="Prediction horizon in trading days"),
    "mode": Field(str, default_value="return", description="Target type: 'return' or 'level'"),
}
@asset(
    name="target_price",
    group_name="add_targets",
    kinds={"python", "regression"},
    config_schema=reg_config_schema,
)
def target_price(context, asset_features_full):
    """
    Creates a numeric target:
    - mode='return': y = future_return over N days
    - mode='level' : y = future_close over N days
    """
    df, ticker = asset_features_full
    n = context.op_config["days_ahead"]
    mode = context.op_config["mode"].lower()

    future_close = df["close"].shift(-n)

    if mode == "level":
        df[f"y_price_level_{n}d"] = future_close
    else:  # default 'return'
        df[f"y_price_return_{n}d"] = (future_close / df["close"]) - 1.0

    print(str(df.info()))
    context.log.info(df.head())
    context.log.info(df.tail())
    save_data(df=df,
              filename="nvda_target_price.csv",
              dir="data/processed",
              context=context,
              asset="target_price"
              )
    return Output((df, ticker),
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            })



clf_config_schema = {
    "days_ahead": Field(int, default_value=1, description="Prediction horizon in trading days"),
    "threshold": Field(float, default_value=0.0, description="Neutral band for classification (e.g., 0.001 = 0.1%)"),
    "labels": Field(
        dict,
        default_value={"up": 1, "down": 0, "flat": None},
        description="Label mapping; set flat=None to drop neutral samples",
    ),
}
@asset(
    name="target_updown",
    group_name="add_targets",
    kinds={"python", "classification"},
    config_schema=clf_config_schema,
)
def target_updown(context, target_price):
    """
    Creates a direction label:
    - Up if return > +threshold
    - Down if return < -threshold
    - Flat if |return| <= threshold (optional; dropped if labels['flat'] is None)
    """
    df, ticker = target_price
    n = context.op_config["days_ahead"]
    thr = context.op_config["threshold"]
    labels = context.op_config["labels"]

    future_ret = (df["close"].shift(-n) / df["close"]) - 1.0

    up_label = labels.get("up", 1)
    down_label = labels.get("down", -1)
    flat_label = labels.get("flat", 0)

    def map_dir(r):
        if r > thr:
            return up_label
        if r < -thr:
            return down_label
        return flat_label

    df[f"y_updown_{n}d"] = future_ret.apply(map_dir)


    print(str(df.info()))
    context.log.info(df.head())
    context.log.info(df.tail())
    save_data(df=df,
              filename="nvda_target_updown.csv",
              dir="data/processed",
              context=context,
              asset="target_updown"
              )
    return Output((df, ticker),
           metadata={"num_rows": df.shape[0],
                     "num_columns": df.shape[1],
                     })