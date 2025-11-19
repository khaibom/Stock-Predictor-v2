from dagster import Definitions, load_assets_from_modules

from .assets import raw_daily_data, lag_features, indicator_features, targets, preprocessing, lstm_model
from .jobs import (
    job_get_data,
    job_lstm_predict_reg,
    job_lstm_predict_cls,
    job_lstm_full
)
from .schedules import schedule_weekday_0800

all_assets = load_assets_from_modules([raw_daily_data, lag_features, indicator_features, targets, preprocessing, lstm_model])
all_jobs = [
    job_get_data,
    job_lstm_predict_reg,
    job_lstm_predict_cls,
    job_lstm_full
]
all_schedules = [schedule_weekday_0800]

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    schedules=all_schedules,
)
