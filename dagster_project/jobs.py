from dagster import define_asset_job

selection = ['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 'target_price', 'target_updown', 'asset_preprocessed_data']
job_weekday_0800 = define_asset_job(
    name='get_data_job',
    selection=selection,
)

selection = ['asset_preprocessed_data']
job_weekday_0810 = define_asset_job(
    name='prepare_data_for_LSTM',
    selection=selection,
)