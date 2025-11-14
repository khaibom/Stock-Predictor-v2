from dagster import define_asset_job

selection = ['raw_daily_data', 'lag_features', 'indicator_features']
job_weekday_0800 = define_asset_job(
    name='get_data_job',
    selection=selection,
)