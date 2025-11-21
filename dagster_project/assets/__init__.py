# Import all assets from different modules
from .raw_daily_data import asset_market_raw
from .lag_features import asset_features_lagged
from .indicator_features import asset_features_full
from .targets import target_price, target_updown
from .preprocessing import asset_preprocessed_data

# LSTM assets
from .lstm_model import (
    lstm_trained_model_reg,
    lstm_predictions_reg,
    lstm_trained_model_cls,
    lstm_predictions_cls,
)

# XGBoost assets
from .xgboost_model import (
    xgb_trained_model_reg,
    xgb_predictions_reg,
    xgb_trained_model_cls,
    xgb_predictions_cls,
)

__all__ = [
    # Data pipeline
    "asset_market_raw",
    "asset_features_lagged",
    "asset_features_full",
    "target_price",
    "target_updown",
    "asset_preprocessed_data",
    
    # LSTM models
    "lstm_trained_model_reg",
    "lstm_predictions_reg",
    "lstm_trained_model_cls",
    "lstm_predictions_cls",
    
    # XGBoost models
    "xgb_trained_model_reg",
    "xgb_predictions_reg",
    "xgb_trained_model_cls",
    "xgb_predictions_cls",
]

