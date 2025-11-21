from dagster import Definitions, load_assets_from_modules

from .assets import raw_daily_data, lag_features, indicator_features, targets, preprocessing, lstm_model, xgboost_model, ensemble_model
from .jobs import (
    job_get_data,
    job_lstm_predict_reg,
    job_lstm_predict_cls,
    job_lstm_full,
    job_xgb_predict_reg,
    job_xgb_predict_cls,
    job_xgb_full,
    job_ensemble_predict_reg,
    job_ensemble_predict_cls,
    job_ensemble_full,
    job_compare_models,
)
from .schedules import schedule_tuesday_saturday_0000

all_assets = load_assets_from_modules([raw_daily_data, lag_features, indicator_features, targets, preprocessing, lstm_model, xgboost_model, ensemble_model])

all_jobs = [
    # Data pipeline
    job_get_data,
    
    # LSTM jobs
    job_lstm_predict_reg,
    job_lstm_predict_cls,
    job_lstm_full,
    
    # XGBoost jobs
    job_xgb_predict_reg,
    job_xgb_predict_cls,
    job_xgb_full,
    
    # Ensemble jobs
    job_ensemble_predict_reg,
    job_ensemble_predict_cls,
    job_ensemble_full,
    
    # Comparison job
    job_compare_models,
]

all_schedules = [schedule_tuesday_saturday_0000]

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    schedules=all_schedules,
)
