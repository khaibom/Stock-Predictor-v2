from dagster import define_asset_job

selection = ['asset_market_raw', 'asset_features_lagged', 'asset_features_full']
job_weekday_0800 = define_asset_job(
    name='get_data_job',
    selection=selection,
)