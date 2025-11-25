# Hyperparameter Tuning Module for Deep Learning Models
# Uses Keras Tuner for LSTM and Optuna for XGBoost

import numpy as np
import keras_tuner as kt
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from keras import Sequential
from pathlib import Path


class LSTMHyperModel(kt.HyperModel):
    """
    Hyperparameter search space for LSTM models.
    Uses Keras Tuner's HyperModel interface.
    """
    
    def __init__(self, lookback, n_features, model_type='regression', num_classes=2):
        """
        Args:
            lookback: Number of timesteps for LSTM
            n_features: Number of input features
            model_type: 'regression' or 'classification'
            num_classes: Number of classes (for classification only)
        """
        self.lookback = lookback
        self.n_features = n_features
        self.model_type = model_type
        self.num_classes = num_classes
    
    def build(self, hp):
        """
        Build model with hyperparameters from Keras Tuner.
        
        Hyperparameters being tuned:
        - lstm_units: 32, 64, 96, 128, 192
        - dense_units: 16, 32, 48, 64, 96
        - dropout: 0.1 - 0.5
        - learning_rate: 1e-4 - 1e-2 (log scale)
        - use_bidirectional: True/False
        - l2_reg: 1e-6 - 1e-3 (log scale)
        """
        from tensorflow.keras.layers import Bidirectional, BatchNormalization, LSTM, Dense, Dropout
        from tensorflow.keras.regularizers import l2
        
        # Hyperparameters to tune
        lstm_units = hp.Choice('lstm_units', values=[32, 64, 96, 128, 192])
        dense_units = hp.Choice('dense_units', values=[16, 32, 48, 64, 96])
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        use_bidirectional = hp.Boolean('use_bidirectional')
        l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-3, sampling='log')
        
        model = Sequential()
        
        # First LSTM layer
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units, return_sequences=True,
                     kernel_regularizer=l2(l2_reg),
                     recurrent_regularizer=l2(l2_reg),
                     input_shape=(self.lookback, self.n_features))
            ))
        else:
            model.add(LSTM(lstm_units, return_sequences=True,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg),
                          input_shape=(self.lookback, self.n_features)))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units // 2, return_sequences=True,
                     kernel_regularizer=l2(l2_reg),
                     recurrent_regularizer=l2(l2_reg))
            ))
        else:
            model.add(LSTM(lstm_units // 2, return_sequences=True,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg)))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout * 0.67))
        
        # Third LSTM layer
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units // 4, return_sequences=False,
                     kernel_regularizer=l2(l2_reg),
                     recurrent_regularizer=l2(l2_reg))
            ))
        else:
            model.add(LSTM(lstm_units // 4, return_sequences=False,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg)))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout * 0.67))
        
        # Dense layers
        model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout * 0.33))
        
        model.add(Dense(dense_units // 2, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout * 0.33))
        
        # Output layer
        if self.model_type == 'regression':
            model.add(Dense(1, activation='linear'))
            loss_fn = tf.keras.losses.Huber(delta=1.0)
            metrics = [
                "mae",
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ]
        else:  # classification
            if self.num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                loss_fn = 'binary_crossentropy'
            else:
                model.add(Dense(self.num_classes, activation='softmax'))
                loss_fn = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0
            ),
            loss=loss_fn,
            metrics=metrics
        )
        
        return model


def tune_lstm_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    lookback, n_features,
    model_type='regression',
    num_classes=2,
    max_trials=50,
    executions_per_trial=1,
    tuner_type='bayesian',  # 'bayesian', 'random', or 'hyperband'
    project_name='lstm_tuning',
    directory='models/tuning',
    context=None
):
    """
    Tune LSTM hyperparameters using Keras Tuner.
    
    Args:
        X_train, y_train: Training data (sequences)
        X_val, y_val: Validation data (sequences)
        lookback: LSTM lookback window
        n_features: Number of features
        model_type: 'regression' or 'classification'
        num_classes: Number of classes (for classification)
        max_trials: Number of hyperparameter combinations to try
        executions_per_trial: Number of times to train each combination (for averaging)
        tuner_type: 'bayesian' (best), 'random', or 'hyperband' (fastest)
        project_name: Name for saving tuning results
        directory: Directory to save tuning results
        context: Dagster context for logging
        
    Returns:
        best_hps: Best hyperparameters found
        best_model: Best model trained with those hyperparameters
        tuner: The tuner object (for inspection)
    """
    if context:
        context.log.info(f"Starting hyperparameter tuning with {tuner_type.upper()} search")
        context.log.info(f"   Max trials: {max_trials}")
        context.log.info(f"   Executions per trial: {executions_per_trial}")
    
    # Create HyperModel
    hypermodel = LSTMHyperModel(
        lookback=lookback,
        n_features=n_features,
        model_type=model_type,
        num_classes=num_classes
    )
    
    # Choose tuner
    if tuner_type == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
            overwrite=True
        )
    elif tuner_type == 'random':
        tuner = kt.RandomSearch(
            hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name,
            overwrite=True
        )
    elif tuner_type == 'hyperband':
        tuner = kt.Hyperband(
            hypermodel,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory=directory,
            project_name=project_name,
            overwrite=True
        )
    else:
        raise ValueError(f"Unknown tuner_type: {tuner_type}. Choose 'bayesian', 'random', or 'hyperband'")
    
    # Early stopping for tuning
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Run search
    if context:
        context.log.info("Searching for best hyperparameters...")
        context.log.info("This may take a while. Progress bars will show for each trial.")
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # Max epochs per trial
        batch_size=32,
        callbacks=[stop_early],
        verbose=2  # Show progress: 2 = one line per epoch, 1 = progress bar
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    if context:
        context.log.info("Best hyperparameters found:")
        context.log.info(f"   lstm_units: {best_hps.get('lstm_units')}")
        context.log.info(f"   dense_units: {best_hps.get('dense_units')}")
        context.log.info(f"   dropout: {best_hps.get('dropout'):.3f}")
        context.log.info(f"   learning_rate: {best_hps.get('learning_rate'):.6f}")
        context.log.info(f"   use_bidirectional: {best_hps.get('use_bidirectional')}")
        context.log.info(f"   l2_reg: {best_hps.get('l2_reg'):.6f}")
    
    # Build best model
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_hps, best_model, tuner


def tune_xgboost_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    task='regression',
    n_trials=100,
    timeout=3600,  # 1 hour timeout
    context=None
):
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        task: 'regression' or 'classification'
        n_trials: Number of trials for Optuna
        timeout: Maximum time for tuning (seconds)
        context: Dagster context for logging
        
    Returns:
        best_params: Best hyperparameters found
        study: The Optuna study object (for inspection)
    """
    import xgboost as xgb
    
    if context:
        context.log.info(f"Starting XGBoost hyperparameter tuning with Optuna")
        context.log.info(f"   Number of trials: {n_trials}")
        context.log.info(f"   Timeout: {timeout}s ({timeout/60:.1f} minutes)")
    
    def objective(trial):
        """Optuna objective function."""
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        if task == 'regression':
            params['objective'] = 'reg:squarederror'
            eval_metric = 'rmse'
        else:  # classification
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = ['logloss', 'error']
            # For imbalanced data
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, scale_pos_weight * 2)
            eval_metric = 'logloss'
        
        # Early stopping
        params['early_stopping_rounds'] = 50
        
        # Train model
        model = xgb.XGBRegressor(**params) if task == 'regression' else xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Get validation score
        if task == 'regression':
            from sklearn.metrics import mean_squared_error
            y_pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))  # RMSE
        else:
            from sklearn.metrics import log_loss
            y_pred_proba = model.predict_proba(X_val)
            score = log_loss(y_val, y_pred_proba)
        
        return score
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    
    if context:
        context.log.info("Best XGBoost hyperparameters found:")
        for key, value in best_params.items():
            context.log.info(f"   {key}: {value}")
        context.log.info(f"   Best validation score: {study.best_value:.6f}")
    
    return best_params, study


def get_best_hyperparameters_summary(best_hps_dict):
    """
    Convert hyperparameters to a clean dictionary for logging/saving.
    
    Args:
        best_hps_dict: Dictionary or Keras Tuner HyperParameters object
        
    Returns:
        Dictionary of hyperparameters
    """
    if hasattr(best_hps_dict, 'values'):
        # Keras Tuner HyperParameters object
        return {k: v for k, v in best_hps_dict.values.items()}
    else:
        # Already a dictionary
        return best_hps_dict

