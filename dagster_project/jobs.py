from dagster import define_asset_job

selection = ['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 'target_price', 'target_updown', 'asset_preprocessed_data']
job_get_data = define_asset_job(
    name='get_data_job',
    selection=selection,
)

# LSTM REGRESSION jobs
job_lstm_predict_reg = define_asset_job(
    name='lstm_predict_pipeline_reg',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_reg', 'lstm_predictions_reg'],
)

# LSTM CLASSIFICATION jobs
job_lstm_predict_cls = define_asset_job(
    name='lstm_predict_pipeline_cls',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_cls', 'lstm_predictions_cls'],
)

# COMBINED job - Train and predict both regression and classification
job_lstm_full = define_asset_job(
    name='lstm_full_pipeline',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 
               'target_price', 'target_updown', 'asset_preprocessed_data', 
               'lstm_trained_model_reg', 'lstm_predictions_reg',
               'lstm_trained_model_cls', 'lstm_predictions_cls'],
)