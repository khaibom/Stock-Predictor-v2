import numpy as np
import pandas as pd
from dagster import asset, Output
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from .methods.save_data import save_data
from .methods.logging import log_df

@asset(
    name="asset_preprocessed_data",
    group_name="preprocessing",
    kinds={"python"}
)
def asset_preprocessed_data(context, target_updown):
    df, ticker, interval, days_ahead = target_updown  # Unpack interval and days_ahead from target_updown
    
    context.log.info(f"Preprocessing data for {ticker} at {interval} interval")

    # Normalize datetime column name (legacy support for 'date' column)
    if 'date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'date': 'datetime'})
        context.log.info("Renamed legacy 'date' column to 'datetime'")
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        context.log.error(f"No datetime column found. Available columns: {df.columns.tolist()}")
        raise ValueError("Expected 'datetime' column in dataframe")

    # Sort by time for time series split (datetime preserves time info for intraday, 00:00:00 for daily)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Temporal features - interval-aware
    # For intraday data, add hour and minute features; for daily+, use day/week/month
    # Note: For daily data (1d), hour/minute will be 0, so logic remains the same
    if interval.endswith('m') or interval in ['1h', '60m', '90m']:
        # Intraday: add hour and minute cyclic features (preserves time accuracy)
        df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["datetime"].dt.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["datetime"].dt.minute / 60)
        context.log.info(f"Added intraday temporal features (hour, minute) for {interval} interval")
    
    # Always add day-level features (useful for all intervals)
    # For daily data (1d), this is the primary temporal information
    df["dow_sin"] = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    
    # Time progression feature (days for daily, hours for intraday)
    if interval.endswith('m') or interval in ['1h', '60m', '90m']:
        # For intraday, use hours since first timestamp for better granularity
        df["time_progression"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds() / 3600.0
    else:
        # For daily data (1d), use days since first date (same logic as before)
        df["time_progression"] = (df["datetime"] - df["datetime"].min()).dt.days
    
    context.log.info(f"Created temporal features: dow_sin, dow_cos, month_sin, month_cos, time_progression")

    # Construct target column names dynamically based on days_ahead from upstream assets
    target_cls = f"y_updown_{days_ahead}d"  # Dynamic: automatically matches config
    target_reg = f"y_price_return_{days_ahead}d"  # Dynamic: automatically matches config
    features = [c for c in df.columns if c not in (target_cls, target_reg, "datetime", "close")] # use adj_close instead of close
    
    context.log.info(f"Using targets (days_ahead={days_ahead}): regression='{target_reg}', classification='{target_cls}'")

    # Features bounded to [0, 1] or [0, 100] range
    # Also includes cyclic temporal features which are already in [-1, 1]
    # Use pattern matching to be interval-agnostic
    bounded_01 = [
        "RSI_14", "Stoch_k", "Stoch_d", "MFI_14",
        "BB_position", "KC_position",
    ]
    
    # Add price_position features dynamically (column names vary by interval)
    # For 1d: price_position_5d, price_position_10d, price_position_20d
    # For 1h: price_position_6d, price_position_13d, price_position_26d, etc.
    bounded_01.extend([c for c in features if c.startswith('price_position_')])
    
    # Cyclic temporal features (already in [-1, 1], no scaling needed but keep separate)
    cyclic_features = [
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    
    # Add intraday cyclic features if they exist
    if "hour_sin" in features:
        cyclic_features.extend(["hour_sin", "hour_cos", "minute_sin", "minute_cos"])

    # Features that are always positive and potentially very large (need log transform)
    # Use pattern matching to dynamically find columns (interval-agnostic)
    heavy_pos = [
        "volume", "Volume_SMA_20", "ATR_14", "BB_width", "High_Low_Spread",
        "KC_width",
    ]
    
    # Add volume and close statistics dynamically (column names vary by interval)
    # For 1d: volume_mean_5d, volume_mean_10d, volume_mean_20d
    # For 1h: volume_mean_6d, volume_mean_13d, volume_mean_26d, etc.
    heavy_pos.extend([c for c in features if c.startswith(('volume_mean_', 'volume_std_', 'close_std_'))])
    
    # Also add volume lags dynamically
    heavy_pos.extend([c for c in features if c.startswith('volume_lag')])

    # Features that can be very large and signed (need signed log transform)
    heavy_signed_obv = ["OBV"]

    # Features centered around zero (returns, momentum indicators, relative measures)
    # Use pattern matching to be interval-agnostic
    zero_center_std = [
        "daily_return", 
        "MACD", "MACD_signal", "MACD_diff", "MACD_acceleration", "MACD_to_signal_ratio",
        "ROC_10", "ROC_to_ATR",
        "Close_to_EMA20", "Close_to_EMA50", "EMA20_to_EMA50",
        "volume_change",
        "BB_width_pct", "volume_price_trend",
        "RSI_change", "RSI_slope_5d",
        "Stoch_K_D_diff", "Stoch_change",
        "efficiency_ratio_10d",
        "ADX_14", "ADX_pos", "ADX_neg", "CCI_20", "WillR_14", "UO", "CMF_20",
        "time_progression",  # Time-based feature
    ]
    
    # Add return features dynamically (column names vary by interval)
    # For 1d: return_5d, return_10d, return_20d, return_mean_5d, etc.
    # For 1h: return_6d, return_13d, return_26d, return_mean_6d, etc.
    zero_center_std.extend([c for c in features if c.startswith(('return_', 'pct_from_', 'volume_ratio_', 'hl_pct_', 'realized_vol_'))])
    
    # Add return lags dynamically
    zero_center_std.extend([c for c in features if c.startswith('daily_return_lag')])
    
    # Add volume_change variations
    zero_center_std.extend([c for c in features if c.startswith('volume_change_') and c != 'volume_change'])

    # Price-like features (absolute levels)
    # Use pattern matching to be interval-agnostic
    price_like_std = [
        "open", "high", "low", "close", "adj_close",
        "EMA_20", "EMA_50", "BB_high", "BB_low", "BB_mid",
        "KC_high", "KC_low", "KC_mid",
        "DC_high", "DC_low", "DC_mid",
    ]
    
    # Add close statistics dynamically (column names vary by interval)
    # For 1d: close_mean_5d, close_min_10d, etc.
    # For 1h: close_mean_6d, close_min_13d, etc.
    price_like_std.extend([c for c in features if c.startswith(('close_mean_', 'close_min_', 'close_max_'))])
    
    # Add price lags dynamically
    price_like_std.extend([c for c in features if c.startswith(('adj_close_lag', 'high_lag', 'low_lag'))])

    def signed_log1p(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.log1p(np.abs(x))

    X = df.loc[:, features].copy()

    # 1) bounded 0-100 -> [0,1]
    # Only scale columns that actually exist
    for c in bounded_01:
        if c in X.columns:
            X[c] = X[c] / 100.0

    # 2) Time-based split
    # last row as X_predict
    X_predict = X.iloc[[-1]].copy()
    X = X.iloc[:-1].copy()

    y_cls = df[target_cls].copy()[:-1]
    y_reg = df[target_reg].copy()[:-1]
    
    # Drop samples with NaN in classification target (binary mode drops "flat" samples)
    valid_cls_mask = y_cls.notna()
    valid_reg_mask = y_reg.notna()
    
    # Use samples that are valid for BOTH tasks
    valid_mask = valid_cls_mask & valid_reg_mask
    
    n_dropped = (~valid_mask).sum()
    n_total = len(X)
    context.log.info(f"Dropping {n_dropped}/{n_total} samples ({100*n_dropped/n_total:.1f}%) with NaN targets")
    
    X = X[valid_mask].reset_index(drop=True)
    y_cls = y_cls[valid_mask].reset_index(drop=True)
    y_reg = y_reg[valid_mask].reset_index(drop=True)

    last_close = float(df["close"].iloc[-1])   # last close price, for price prediction
    last_datetime = df["datetime"].iloc[-1]   # last timestamp (preserves time for intraday)

    n = len(X)
    train_ratio = 0.7
    val_ratio = 0.15
    # => test = 0.15

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X.iloc[:n_train].copy()
    X_val = X.iloc[n_train:n_train + n_val].copy()
    X_test = X.iloc[n_train + n_val:].copy()

    y_train_cls = y_cls[:n_train]
    y_val_cls = y_cls[n_train:n_train + n_val]
    y_test_cls = y_cls[n_train + n_val:]

    y_train_reg = y_reg[:n_train]
    y_val_reg = y_reg[n_train:n_train + n_val]
    y_test_reg = y_reg[n_train + n_val:]

    print(f"\n[asset_preprocessed_data - BEFORE SCALING] X_train: {X_train.shape} | Columns & Types:\n{X_train.dtypes}\n")

    # 3) Fit scalers on TRAIN only, then transform all splits
    # 3.1) heavy positive -> log1p then RobustScaler
    robust_scaler = RobustScaler()
    if heavy_pos:
        for c in heavy_pos:
            X_train[c] = np.log1p(X_train[c].clip(lower=0))
            X_val[c] = np.log1p(X_val[c].clip(lower=0))
            X_test[c] = np.log1p(X_test[c].clip(lower=0))
            X_predict[c] = np.log1p(X_predict[c].clip(lower=0))

        cols = [c for c in heavy_pos if c in X_train.columns]
        if cols:
            robust_scaler.fit(X_train[cols])
            X_train[cols] = robust_scaler.transform(X_train[cols])
            X_val[cols] = robust_scaler.transform(X_val[cols])
            X_test[cols] = robust_scaler.transform(X_test[cols])
            X_predict[cols] = robust_scaler.transform(X_predict[cols])

    # 3.2) OBV -> signed log1p then StandardScaler (or RobustScaler if extremely spiky)
    if heavy_signed_obv:
        cols = [c for c in heavy_signed_obv if c in X_train.columns]
        if cols:
            for c in cols:
                X_train[c] = signed_log1p(X_train[c].astype(float))
                X_val[c] = signed_log1p(X_val[c].astype(float))
                X_test[c] = signed_log1p(X_test[c].astype(float))
                X_predict[c] = signed_log1p(X_predict[c].astype(float))

            std_obv = StandardScaler()
            std_obv.fit(X_train[cols])
            X_train[cols] = std_obv.transform(X_train[cols])
            X_val[cols] = std_obv.transform(X_val[cols])
            X_test[cols] = std_obv.transform(X_test[cols])
            X_predict[cols] = std_obv.transform(X_predict[cols])

    # 3.3) zero-centered indicators & returns -> StandardScaler
    cols = [c for c in zero_center_std if c in X_train.columns]
    if cols:
        std_zero = StandardScaler()
        std_zero.fit(X_train[cols])
        X_train[cols] = std_zero.transform(X_train[cols])
        X_val[cols] = std_zero.transform(X_val[cols])
        X_test[cols] = std_zero.transform(X_test[cols])
        X_predict[cols] = std_zero.transform(X_predict[cols])

    # 3.4) price-like levels -> StandardScaler  (consider replacing with relative features before this step)
    cols = [c for c in price_like_std if c in X_train.columns]
    if cols:
        std_price = StandardScaler()
        std_price.fit(X_train[cols])
        X_train[cols] = std_price.transform(X_train[cols])
        X_val[cols] = std_price.transform(X_val[cols])
        X_test[cols] = std_price.transform(X_test[cols])
        X_predict[cols] = std_price.transform(X_predict[cols])
    
    # 3.5) Time progression -> StandardScaler (for trend)
    if "time_progression" in X_train.columns:
        std_time = StandardScaler()
        X_train["time_progression"] = std_time.fit_transform(X_train[["time_progression"]])
        X_val["time_progression"] = std_time.transform(X_val[["time_progression"]])
        X_test["time_progression"] = std_time.transform(X_test[["time_progression"]])
        X_predict["time_progression"] = std_time.transform(X_predict[["time_progression"]])

    # 4) LSTM-friendly range on everything (except cyclic features which are already in [-1,1])
    # Exclude cyclic features from MinMaxScaler as they're already properly bounded
    non_cyclic_features = [f for f in features if f not in cyclic_features]
    
    mm_all = MinMaxScaler(feature_range=(-1, 1))
    mm_all.fit(X_train[non_cyclic_features])

    # Apply MinMaxScaler to non-cyclic features
    X_train_non_cyclic_scaled = mm_all.transform(X_train[non_cyclic_features])
    X_val_non_cyclic_scaled = mm_all.transform(X_val[non_cyclic_features])
    X_test_non_cyclic_scaled = mm_all.transform(X_test[non_cyclic_features])
    X_predict_non_cyclic_scaled = mm_all.transform(X_predict[non_cyclic_features])
    
    # Combine scaled non-cyclic features with cyclic features (keep cyclic as-is)
    X_train_scaled = pd.DataFrame(X_train_non_cyclic_scaled, columns=non_cyclic_features, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_non_cyclic_scaled, columns=non_cyclic_features, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_non_cyclic_scaled, columns=non_cyclic_features, index=X_test.index)
    X_predict_scaled = pd.DataFrame(X_predict_non_cyclic_scaled, columns=non_cyclic_features, index=X_predict.index)
    
    # Add back cyclic features (already properly bounded in [-1, 1])
    for cyclic_feat in cyclic_features:
        if cyclic_feat in X_train.columns:
            X_train_scaled[cyclic_feat] = X_train[cyclic_feat].values
            X_val_scaled[cyclic_feat] = X_val[cyclic_feat].values
            X_test_scaled[cyclic_feat] = X_test[cyclic_feat].values
            X_predict_scaled[cyclic_feat] = X_predict[cyclic_feat].values
    
    # Reorder columns to match original feature order
    X_train_scaled = X_train_scaled[features]
    X_val_scaled = X_val_scaled[features]
    X_test_scaled = X_test_scaled[features]
    X_predict_scaled = X_predict_scaled[features]

    print(f"\n[asset_preprocessed_data - AFTER SCALING] X_train_scaled: {X_train_scaled.shape} | Columns & Types:\n{X_train_scaled.dtypes}\n")

    save_data(df=X_train_scaled, filename=f"{ticker}_{interval}_X_train_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_val_scaled, filename=f"{ticker}_{interval}_X_val_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_test_scaled, filename=f"{ticker}_{interval}_X_test_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_predict_scaled, filename=f"{ticker}_{interval}_X_predict_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

    save_data(df=y_train_cls, filename=f"{ticker}_{interval}_y_train_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_val_cls, filename=f"{ticker}_{interval}_y_val_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_test_cls, filename=f"{ticker}_{interval}_y_test_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

    save_data(df=y_train_reg, filename=f"{ticker}_{interval}_y_train_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_val_reg, filename=f"{ticker}_{interval}_y_val_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_test_reg, filename=f"{ticker}_{interval}_y_test_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

    log_df(X_train_scaled,context, 'X_train_scaled')
    log_df(X_val_scaled,context, 'X_val_scaled')
    log_df(X_test_scaled,context, 'X_test_scaled')
    log_df(X_predict_scaled,context, 'X_predict_scaled')

    log_df(y_train_cls,context, 'y_train_cls')
    log_df(y_val_cls,context, 'y_val_cls')
    log_df(y_test_cls,context, 'y_test_cls')

    log_df(y_train_reg,context, 'y_train_reg')
    log_df(y_val_reg,context, 'y_val_reg')
    log_df(y_test_reg,context, 'y_test_reg')

    output_value = {
        "ticker": ticker,
        "interval": interval,
        "features": features,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "X_predict": X_predict_scaled,
        "y_train_cls": y_train_cls,
        "y_val_cls": y_val_cls,
        "y_test_cls": y_test_cls,
        "y_train_reg": y_train_reg,
        "y_val_reg": y_val_reg,
        "y_test_reg": y_test_reg,
        "last_close": last_close,
        "last_date": last_datetime,  # For backward compatibility with existing code
    }
    metadata = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "n_features": len(features),
        "ticker": ticker,
        "interval": interval,
        "last_close": last_close,
        "last_datetime": str(last_datetime),  # More accurate name
        "last_date-predict": str(last_datetime),  # Backward compatibility
    }

    return Output(value=output_value, metadata=metadata)