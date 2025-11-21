from dagster import define_asset_job

# Data pipeline selection (shared by all jobs)
selection = ['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 'target_price', 'target_updown', 'asset_preprocessed_data']

# ============================================================================
# DATA PIPELINE JOB
# ============================================================================
job_get_data = define_asset_job(
    name='get_data_job',
    selection=selection,
)

# ============================================================================
# LSTM JOBS
# ============================================================================

# LSTM REGRESSION job
job_lstm_predict_reg = define_asset_job(
    name='lstm_predict_pipeline_reg',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_reg', 'lstm_predictions_reg'],
)

# LSTM CLASSIFICATION job
job_lstm_predict_cls = define_asset_job(
    name='lstm_predict_pipeline_cls',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_cls', 'lstm_predictions_cls'],
)

# LSTM FULL job - Train and predict both regression and classification
job_lstm_full = define_asset_job(
    name='lstm_full_pipeline',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 
               'target_price', 'target_updown', 'asset_preprocessed_data', 
               'lstm_trained_model_reg', 'lstm_predictions_reg',
               'lstm_trained_model_cls', 'lstm_predictions_cls'],
)

# ============================================================================
# XGBOOST JOBS
# ============================================================================

# XGBoost REGRESSION job
job_xgb_predict_reg = define_asset_job(
    name='xgb_predict_pipeline_reg',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'xgb_trained_model_reg', 'xgb_predictions_reg'],
)

# XGBoost CLASSIFICATION job
job_xgb_predict_cls = define_asset_job(
    name='xgb_predict_pipeline_cls',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'xgb_trained_model_cls', 'xgb_predictions_cls'],
)

# XGBoost FULL job - Train and predict both regression and classification
job_xgb_full = define_asset_job(
    name='xgb_full_pipeline',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 
               'target_price', 'target_updown', 'asset_preprocessed_data', 
               'xgb_trained_model_reg', 'xgb_predictions_reg',
               'xgb_trained_model_cls', 'xgb_predictions_cls'],
)

# ============================================================================
# ENSEMBLE JOBS - Combine LSTM and XGBoost
# ============================================================================

# ENSEMBLE REGRESSION job
job_ensemble_predict_reg = define_asset_job(
    name='ensemble_predict_pipeline_reg',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_reg', 'xgb_trained_model_reg',
               'ensemble_predictions_reg'],
)

# ENSEMBLE CLASSIFICATION job
job_ensemble_predict_cls = define_asset_job(
    name='ensemble_predict_pipeline_cls',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full',
               'target_price', 'target_updown', 'asset_preprocessed_data',
               'lstm_trained_model_cls', 'xgb_trained_model_cls',
               'ensemble_predictions_cls'],
)

# ENSEMBLE FULL job - Train and predict with ensemble for both regression and classification
job_ensemble_full = define_asset_job(
    name='ensemble_full_pipeline',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 
               'target_price', 'target_updown', 'asset_preprocessed_data', 
               'lstm_trained_model_reg', 'xgb_trained_model_reg', 'ensemble_predictions_reg',
               'lstm_trained_model_cls', 'xgb_trained_model_cls', 'ensemble_predictions_cls'],
)

# ============================================================================
# COMPARISON JOB - Run both LSTM and XGBoost
# ============================================================================

# COMPARISON job - Train and predict with BOTH LSTM and XGBoost (all models)
job_compare_models = define_asset_job(
    name='compare_all_models',
    selection=['asset_market_raw', 'asset_features_lagged', 'asset_features_full', 
               'target_price', 'target_updown', 'asset_preprocessed_data',
               # LSTM models
               'lstm_trained_model_reg', 'lstm_predictions_reg',
               'lstm_trained_model_cls', 'lstm_predictions_cls',
               # XGBoost models
               'xgb_trained_model_reg', 'xgb_predictions_reg',
               'xgb_trained_model_cls', 'xgb_predictions_cls',
               # Ensemble models
               'ensemble_predictions_reg', 'ensemble_predictions_cls'],
)