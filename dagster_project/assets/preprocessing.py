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
    df, ticker = target_updown

    # sort by time for time series split
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # days since first date
    df["date_days"] = (df["date"] - df["date"].min()).dt.days

    # cyclic encodings for weekday and month
    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["date"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["date"].dt.month - 1) / 12)

    #targets + features (updated to 3-day prediction)
    target_cls = "y_updown_3d"  # Changed from 1d to 3d
    target_reg = "y_price_return_3d"  # Changed from 1d to 3d
    features = [c for c in df.columns if c not in (target_cls, target_reg, "date", "close")] # use adj_close instead of close
    
    context.log.info(f"Using targets: regression='{target_reg}', classification='{target_cls}'")

    # Features bounded to [0, 1] or [0, 100] range
    bounded_01 = [
        "RSI_14", "Stoch_k", "Stoch_d", "MFI_14",
        "BB_position", "KC_position", 
        "price_position_5d", "price_position_10d", "price_position_20d",
    ]

    # Features that are always positive and potentially very large (need log transform)
    heavy_pos = [
        "volume", "Volume_SMA_20", "ATR_14", "BB_width", "High_Low_Spread",
        "volume_mean_5d", "volume_mean_10d", "volume_mean_20d",
        "volume_std_5d", "volume_std_10d", "volume_std_20d",
        "KC_width", "close_std_5d", "close_std_10d", "close_std_20d",
    ]
    # Also add volume lags
    for lag in [1, 2, 3, 5, 7, 10]:
        heavy_pos.append(f"volume_lag{lag}")

    # Features that can be very large and signed (need signed log transform)
    heavy_signed_obv = ["OBV"]

    # Features centered around zero (returns, momentum indicators, relative measures)
    zero_center_std = [
        "daily_return", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
        "return_mean_5d", "return_mean_10d", "return_mean_20d",
        "return_std_5d", "return_std_10d", "return_std_20d",
        "MACD", "MACD_signal", "MACD_diff", "MACD_acceleration", "MACD_to_signal_ratio",
        "ROC_10", "ROC_to_ATR",
        "Close_to_EMA20", "Close_to_EMA50", "EMA20_to_EMA50",
        "pct_from_high_5d", "pct_from_high_10d", "pct_from_high_20d",
        "pct_from_low_5d", "pct_from_low_10d", "pct_from_low_20d",
        "volume_change", "volume_change_5d",
        "volume_ratio_5d", "volume_ratio_10d", "volume_ratio_20d",
        "hl_pct_5d", "hl_pct_10d", "hl_pct_20d",
        "realized_vol_10d", "realized_vol_20d",
        "BB_width_pct", "volume_price_trend",
        "RSI_change", "RSI_slope_5d",
        "Stoch_K_D_diff", "Stoch_change",
        "efficiency_ratio_10d",
        "ADX_14", "ADX_pos", "ADX_neg", "CCI_20", "WillR_14", "UO", "CMF_20",
    ]
    # Add return lags
    for lag in [1, 2, 3, 5, 7, 10]:
        zero_center_std.append(f"daily_return_lag{lag}")

    # Price-like features (absolute levels)
    price_like_std = [
        "open", "high", "low", "close", "adj_close",
        "EMA_20", "EMA_50", "BB_high", "BB_low", "BB_mid",
        "KC_high", "KC_low", "KC_mid",
        "DC_high", "DC_low", "DC_mid",
        "close_mean_5d", "close_mean_10d", "close_mean_20d",
        "close_min_5d", "close_min_10d", "close_min_20d",
        "close_max_5d", "close_max_10d", "close_max_20d",
    ]
    # Add price lags
    for lag in [1, 2, 3, 5, 7, 10]:
        price_like_std.append(f"adj_close_lag{lag}")
        price_like_std.append(f"high_lag{lag}")
        price_like_std.append(f"low_lag{lag}")

    def signed_log1p(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.log1p(np.abs(x))

    X = df.loc[:, features].copy()

    # 1) bounded 0-100 -> [0,1]
    for c in bounded_01:
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

    last_close = float(df["close"].iloc[-1])   # yesterday's close, for price prediction
    last_date = df["date"].iloc[-1]

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

    # 4) LSTM-friendly range on everything (except labels), you can add:
    mm_all = MinMaxScaler(feature_range=(-1, 1))
    mm_all.fit(X_train)

    X_train_scaled = mm_all.transform(X_train)
    X_val_scaled = mm_all.transform(X_val)
    X_test_scaled = mm_all.transform(X_test)
    X_predict_scaled = mm_all.transform(X_predict)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=features, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    X_predict_scaled = pd.DataFrame(X_predict_scaled, columns=features, index=X_predict.index)

    print(f"\n[asset_preprocessed_data - AFTER SCALING] X_train_scaled: {X_train_scaled.shape} | Columns & Types:\n{X_train_scaled.dtypes}\n")

    save_data(df=X_train_scaled, filename=f"{ticker}_X_train_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_val_scaled, filename=f"{ticker}_X_val_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_test_scaled, filename=f"{ticker}_X_test_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=X_predict_scaled, filename=f"{ticker}_X_predict_scaled.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

    save_data(df=y_train_cls, filename=f"{ticker}_y_train_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_val_cls, filename=f"{ticker}_y_val_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_test_cls, filename=f"{ticker}_y_test_cls.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

    save_data(df=y_train_reg, filename=f"{ticker}_y_train_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_val_reg, filename=f"{ticker}_y_val_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")
    save_data(df=y_test_reg, filename=f"{ticker}_y_test_reg.csv", dir=f"data/processed/{ticker}", context=context, asset="asset_preprocessed_data")

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
        "last_date": last_date,
    }
    metadata = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "n_features": len(features),
        "ticker": ticker,
        "last_close": last_close,
        "last_date-predict": str(last_date),
    }

    return Output(value=output_value, metadata=metadata)