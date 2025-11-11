from dagster import Definitions, load_assets_from_modules

from .assets import raw_daily_data, lag_features, indicator_features

all_assets = load_assets_from_modules([raw_daily_data, lag_features, indicator_features])

defs = Definitions(
    assets=all_assets,
)
