from dagster import Definitions, load_assets_from_modules

from .assets import raw_daily_data, lag_features, indicator_features, targets, preprocessing
from .jobs import job_weekday_0800, job_weekday_0810
from .schedules import schedule_weekday_0800, schedule_weekday_0810

all_assets = load_assets_from_modules([raw_daily_data, lag_features, indicator_features, targets, preprocessing])
all_jobs = [job_weekday_0800, job_weekday_0810]
all_schedules = [schedule_weekday_0800, schedule_weekday_0810]

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    schedules=all_schedules,
)
