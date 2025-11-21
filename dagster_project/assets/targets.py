import numpy as np
from dagster import asset, Field, Output
from .methods.save_data import save_data
from .methods.logging import log_df

reg_config_schema = {
    "days_ahead": Field(int, default_value=1, description="Prediction horizon in TRADING days (weekends excluded automatically)"),
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
    - mode='return': y = future_return over N TRADING days (weekends auto-skipped by pandas)
    - mode='level' : y = future_close over N TRADING days
    
    NOTE: Since data only contains trading days (no weekends), shift(-n) automatically
    shifts by n TRADING days, not calendar days.
    """
    df, ticker = asset_features_full
    n = context.op_config["days_ahead"]
    mode = context.op_config["mode"].lower()

    # shift(-n) on trading-day-only data = n trading days ahead
    future_close = df["close"].shift(-n)

    if mode == "level":
        df[f"y_price_level_{n}d"] = future_close
    else:  # default 'return'
        df[f"y_price_return_{n}d"] = (future_close / df["close"]) - 1.0

    print(f"\n[target_price] Shape: {df.shape} | Columns & Types:\n{df.dtypes}\n")

    log_df(df, context, 'target_price')
    save_data(df=df,
              filename=f"{ticker}_target_price.csv",
              dir=f"data/processed/{ticker}",
              context=context,
              asset="target_price"
              )
    return Output((df, ticker, n),  # Return days_ahead for dynamic column name construction
                  metadata={"num_rows": df.shape[0],
                            "num_columns": df.shape[1],
                            "days_ahead": n,
                            })



clf_config_schema = {
    "days_ahead": Field(int, default_value=1, description="Prediction horizon in TRADING days (weekends excluded)"),
    "threshold": Field(float, default_value=0.01, description="Threshold for binary classification (e.g., 0.01 = 1.0%). Movements < threshold are DROPPED."),
    "mode": Field(str, default_value="binary", description="'binary' (Up/Down only) or 'ternary' (Up/Down/Flat)"),
}
@asset(
    name="target_updown",
    group_name="add_targets",
    kinds={"python", "classification"},
    config_schema=clf_config_schema,
)
def target_updown(context, target_price):
    """
    Creates a direction label for BINARY classification:
    - Up (1) if return > +threshold  
    - Down (0) if return < -threshold
    - Samples with |return| <= threshold are DROPPED (too noisy)
    
    This creates cleaner, more learnable labels by removing ambiguous "flat" movements.
    
    NOTE: Uses n TRADING days ahead (weekends auto-skipped).
    """
    df, ticker, n_from_price = target_price  # Get days_ahead from target_price
    n = context.op_config["days_ahead"]
    
    # Verify consistency between regression and classification configs
    if n != n_from_price:
        context.log.warning(f"Classification days_ahead ({n}) != Regression days_ahead ({n_from_price}). Using classification config ({n}).")
    thr = context.op_config["threshold"]
    mode = context.op_config["mode"]

    # Calculate future return over n trading days
    future_ret = (df["close"].shift(-n) / df["close"]) - 1.0

    if mode == "binary":
        # Binary classification: Up (1) vs Down (0), drop middle
        def map_dir(r):
            if r > thr:
                return 1  # Up
            elif r < -thr:
                return 0  # Down
            else:
                return np.nan  # Drop ambiguous samples
        
        df[f"y_updown_{n}d"] = future_ret.apply(map_dir)
        context.log.info(f"Binary classification: Up (return > {thr:.1%}) vs Down (return < -{thr:.1%}), dropping |return| <= {thr:.1%}")
    
    else:  # ternary (original 3-class)
        def map_dir(r):
            if r > thr:
                return 1  # Up
            elif r < -thr:
                return -1  # Down (will be converted to 0 later)
            else:
                return 0  # Flat (will be converted to 1 later)
        
        df[f"y_updown_{n}d"] = future_ret.apply(map_dir)
        context.log.info(f"Ternary classification: Up/Down/Flat with threshold {thr:.1%}")

    # Log class distribution
    target_col = f"y_updown_{n}d"
    class_counts = df[target_col].value_counts().sort_index()
    total_before = len(df)
    valid_samples = df[target_col].notna().sum()
    dropped_samples = total_before - valid_samples
    
    context.log.info(f"Target distribution ({n}-day ahead):")
    context.log.info(f"  Total samples: {total_before}")
    context.log.info(f"  Valid samples: {valid_samples}")
    context.log.info(f"  Dropped samples: {dropped_samples} ({100*dropped_samples/total_before:.1f}%)")
    for label, count in class_counts.items():
        if not np.isnan(label):
            pct = 100 * count / valid_samples
            label_name = "Up" if label == 1 else ("Down" if label == 0 else "Flat")
            context.log.info(f"  {label_name} ({int(label)}): {count} samples ({pct:.1f}%)")

    print(f"\n[target_updown] Shape: {df.shape} | Columns & Types:\n{df.dtypes}\n")
    print(f"Class distribution: {class_counts.to_dict()}\n")

    log_df(df, context, 'target_updown')
    save_data(df=df,
              filename=f"{ticker}_target_updown.csv",
              dir=f"data/processed/{ticker}",
              context=context,
              asset="target_updown"
              )
    return Output((df, ticker, n),  # Return days_ahead for dynamic column name construction
           metadata={"num_rows": df.shape[0],
                     "num_columns": df.shape[1],
                     "ticker": ticker,
                     "days_ahead": n,
                     })