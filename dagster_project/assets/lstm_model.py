import numpy as np
import pandas as pd
from pathlib import Path
from dagster import asset, Output, Field, MetadataValue
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.src.layers import Dropout, LSTM, Dense
from keras import Sequential
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .methods.save_data import save_data
from .methods.logging import log_df

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
LOOKBACK = 30  # timesteps for LSTM window
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CHARTS_DIR = Path("models/charts")
CHARTS_DIR.mkdir(exist_ok=True)


def make_sequences(X_df: pd.DataFrame, y_arr: np.ndarray, lookback: int):
    """
    Build sequences for LSTM:
    - X_df: DataFrame of shape (N, F)
    - y_arr: array of shape (N,)
    - Returns X_seq: (M, lookback, F), y_seq: (M,)
    Skips NaNs in y if present.
    """
    X_values = X_df.values
    y_arr = np.asarray(y_arr, dtype=float)

    assert len(X_values) == len(y_arr), "X and y must have same length"

    X_seq, y_seq = [], []
    n = len(X_values)
    for i in range(lookback, n):
        if np.isnan(y_arr[i]):
            continue
        X_seq.append(X_values[i - lookback:i])
        y_seq.append(y_arr[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def plot_training_history(history, ticker: str, model_type: str, save_path: Path):
    """
    Plot training/validation loss curves.
    """
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{ticker} {model_type} - Training/Validation Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Main metric (MAE for regression, Accuracy for classification)
    plt.subplot(1, 2, 2)
    if 'mae' in history.history:
        plt.plot(history.history["mae"], label="train_mae", linewidth=2)
        plt.plot(history.history["val_mae"], label="val_mae", linewidth=2)
        plt.ylabel("MAE", fontsize=12)
        plt.title(f"{ticker} {model_type} - MAE", fontsize=14, fontweight='bold')
    elif 'accuracy' in history.history:
        plt.plot(history.history["accuracy"], label="train_accuracy", linewidth=2)
        plt.plot(history.history["val_accuracy"], label="val_accuracy", linewidth=2)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title(f"{ticker} {model_type} - Accuracy", fontsize=14, fontweight='bold')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_predictions_comparison(y_true, y_pred, ticker: str, model_type: str, save_path: Path):
    """
    Plot true vs predicted values on test set.
    """
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: All predictions
    plt.subplot(1, 2, 1)
    plt.plot(y_true, label="True", linewidth=2, alpha=0.7)
    plt.plot(y_pred, label="Predicted", linewidth=2, alpha=0.7)
    plt.xlabel("Sample Index (Test Set)", fontsize=12)
    plt.ylabel("Return" if "Regression" in model_type else "Class", fontsize=12)
    plt.title(f"{ticker} {model_type} - All Test Predictions", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Last N predictions (zoomed in)
    N = min(200, len(y_true))
    plt.subplot(1, 2, 2)
    plt.plot(range(-N, 0), y_true[-N:], label="True", linewidth=2, alpha=0.7)
    plt.plot(range(-N, 0), y_pred[-N:], label="Predicted", linewidth=2, alpha=0.7)
    plt.xlabel("Relative Index (Last N Samples)", fontsize=12)
    plt.ylabel("Return" if "Regression" in model_type else "Class", fontsize=12)
    plt.title(f"{ticker} {model_type} - Last {N} Test Predictions", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_scatter_comparison(y_true, y_pred, ticker: str, model_type: str, save_path: Path):
    """
    Scatter plot of true vs predicted values.
    """
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{ticker} {model_type} - True vs Predicted (Scatter)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_confusion_matrix_chart(cm, ticker: str, save_path: Path):
    """
    Plot confusion matrix for classification.
    """
    import seaborn as sns
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Flat', 'Up'],
                yticklabels=['Down', 'Flat', 'Up'],
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"{ticker} Classification - Confusion Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def build_lstm_model_reg(
    lookback: int,
    n_features: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
    use_bidirectional: bool = True,
    l2_reg: float = 1e-4,
):
    """
    Build enhanced LSTM model for regression.
    
    Improvements:
    - Bidirectional LSTM for capturing patterns in both directions
    - Batch normalization for training stability
    - L2 regularization to prevent overfitting
    - Larger capacity with more units
    - Progressive dimensionality reduction
    """
    from tensorflow.keras.layers import Bidirectional, BatchNormalization
    from tensorflow.keras.regularizers import l2
    
    model = Sequential()
    
    # First LSTM layer (bidirectional)
    if use_bidirectional:
        model.add(Bidirectional(
            LSTM(lstm_units, return_sequences=True, 
                 kernel_regularizer=l2(l2_reg),
                 recurrent_regularizer=l2(l2_reg)),
            input_shape=(lookback, n_features)
        ))
    else:
        model.add(LSTM(lstm_units, return_sequences=True, 
                      kernel_regularizer=l2(l2_reg),
                      recurrent_regularizer=l2(l2_reg),
                      input_shape=(lookback, n_features)))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second LSTM layer (bidirectional)
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
    model.add(Dropout(0.2))
    
    # Third LSTM layer (no sequences returned)
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
    model.add(Dropout(0.2))
    
    # Dense layers with batch normalization
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(dense_units // 2, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(1, activation='linear'))
    
    # Use Huber loss which is more robust to outliers than MSE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping to prevent exploding gradients
        ),
        loss=tf.keras.losses.Huber(delta=1.0),  # More robust than MSE
        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape")
        ],
    )
    return model


def build_lstm_model_cls(
    lookback: int,
    n_features: int,
    num_classes: int = 3,
    lstm_units: int = 128,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
    use_bidirectional: bool = True,
    l2_reg: float = 1e-4,
):
    """
    Build enhanced LSTM model for classification.
    
    Improvements:
    - Bidirectional LSTM for capturing patterns in both directions
    - Batch normalization for training stability
    - L2 regularization to prevent overfitting
    - Class weights support for imbalanced data
    - Softmax output for multi-class classification
    """
    from tensorflow.keras.layers import Bidirectional, BatchNormalization
    from tensorflow.keras.regularizers import l2
    
    model = Sequential()
    
    # First LSTM layer (bidirectional)
    if use_bidirectional:
        model.add(Bidirectional(
            LSTM(lstm_units, return_sequences=True, 
                 kernel_regularizer=l2(l2_reg),
                 recurrent_regularizer=l2(l2_reg)),
            input_shape=(lookback, n_features)
        ))
    else:
        model.add(LSTM(lstm_units, return_sequences=True, 
                      kernel_regularizer=l2(l2_reg),
                      recurrent_regularizer=l2(l2_reg),
                      input_shape=(lookback, n_features)))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second LSTM layer (bidirectional)
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
    model.add(Dropout(0.2))
    
    # Third LSTM layer (no sequences returned)
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
    model.add(Dropout(0.2))
    
    # Dense layers with batch normalization
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(dense_units // 2, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.1))
    
    # Output layer - softmax for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Use categorical crossentropy for multi-class classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping
        ),
        loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
        metrics=[
            'accuracy',
        ],
    )
    return model


# ============================================================================
# REGRESSION MODEL ASSETS
# ============================================================================

training_config_schema_reg = {
    "lookback": Field(int, default_value=30, description="Number of timesteps for LSTM window"),
    "lstm_units": Field(int, default_value=128, description="Number of units in first LSTM layer"),
    "dense_units": Field(int, default_value=64, description="Number of units in first dense layer"),
    "learning_rate": Field(float, default_value=1e-3, description="Initial learning rate for optimizer"),
    "epochs": Field(int, default_value=200, description="Maximum number of training epochs"),
    "batch_size": Field(int, default_value=32, description="Batch size for training"),
    "patience": Field(int, default_value=120, description="Early stopping patience"),
    "use_bidirectional": Field(bool, default_value=True, description="Use bidirectional LSTM layers"),
    "l2_reg": Field(float, default_value=1e-4, description="L2 regularization factor"),
}


@asset(
    name="lstm_trained_model_reg",
    group_name="LSTM",
    kinds={"python", "regression"},
    config_schema=training_config_schema_reg,
)
def lstm_trained_model_reg(context, asset_preprocessed_data):
    """
    Train LSTM regression model on preprocessed data.
    Saves the trained model to disk and returns model metadata.
    """
    # Extract config
    lookback = context.op_config.get("lookback", LOOKBACK)
    lstm_units = context.op_config.get("lstm_units", 128)
    dense_units = context.op_config.get("dense_units", 64)
    learning_rate = context.op_config.get("learning_rate", 1e-3)
    epochs = context.op_config.get("epochs", 200)
    batch_size = context.op_config.get("batch_size", 32)
    patience = context.op_config.get("patience", 120)
    use_bidirectional = context.op_config.get("use_bidirectional", True)
    l2_reg = context.op_config.get("l2_reg", 1e-4)

    # Extract data from preprocessing asset
    ticker = asset_preprocessed_data["ticker"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    y_train_reg = asset_preprocessed_data["y_train_reg"]
    y_val_reg = asset_preprocessed_data["y_val_reg"]
    y_test_reg = asset_preprocessed_data["y_test_reg"]

    context.log.info(f"[REGRESSION] Creating sequences with lookback={lookback}")
    
    # Create sequences for LSTM
    X_train_seq, y_train_seq = make_sequences(X_train, y_train_reg, lookback)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val_reg, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test_reg, lookback)

    context.log.info(f"[REGRESSION] Sequence shapes:")
    context.log.info(f"  X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    context.log.info(f"  X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
    context.log.info(f"  X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

    # Build model
    n_features = X_train_seq.shape[-1]
    context.log.info(f"[REGRESSION] Building enhanced LSTM model with {n_features} features")
    context.log.info(f"[REGRESSION] Using bidirectional: {use_bidirectional}, L2 reg: {l2_reg}")
    
    model = build_lstm_model_reg(
        lookback=lookback,
        n_features=n_features,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        use_bidirectional=use_bidirectional,
        l2_reg=l2_reg,
    )

    context.log.info("[REGRESSION] Model architecture:")
    model.summary(print_fn=lambda x: context.log.info(x))

    # Set up callbacks for improved training
    callback_list = [
        # Early stopping with more patience
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1,
        ),
        # Save best model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / ticker / f"{ticker}_best_model_reg.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # Train model
    context.log.info(f"[REGRESSION] Training enhanced model for up to {epochs} epochs...")
    context.log.info(f"[REGRESSION] Early stopping patience: {patience}, LR reduction patience: {patience // 3}")
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )

    # Evaluate on test set
    test_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    test_loss = test_results[0]
    test_mae = test_results[1]
    test_rmse = test_results[2] if len(test_results) > 2 else 0
    test_mape = test_results[3] if len(test_results) > 3 else 0
    
    context.log.info(f"[REGRESSION] Test Loss (Huber): {test_loss:.6f}")
    context.log.info(f"[REGRESSION] Test MAE: {test_mae:.6f}")
    context.log.info(f"[REGRESSION] Test RMSE: {test_rmse:.6f}")
    context.log.info(f"[REGRESSION] Test MAPE: {test_mape:.2f}%")

    # Get predictions for test set
    y_test_pred = model.predict(X_test_seq, verbose=0).flatten()

    # Generate visualizations
    context.log.info("[REGRESSION] Generating training history plot...")
    training_plot_path = CHARTS_DIR / ticker / f"{ticker}_training_history_reg.png"
    plot_training_history(history, ticker, "LSTM Regression", training_plot_path)
    
    context.log.info("[REGRESSION] Generating predictions comparison plot...")
    predictions_plot_path = CHARTS_DIR / ticker / f"{ticker}_predictions_comparison_reg.png"
    plot_predictions_comparison(y_test_seq, y_test_pred, ticker, "LSTM Regression", predictions_plot_path)
    
    context.log.info("[REGRESSION] Generating scatter plot...")
    scatter_plot_path = CHARTS_DIR / ticker / f"{ticker}_scatter_reg.png"
    plot_scatter_comparison(y_test_seq, y_test_pred, ticker, "LSTM Regression", scatter_plot_path)

    # Save model
    model_path = MODEL_DIR / ticker / f"{ticker}_lstm_model_reg.keras"
    model.save(model_path)
    context.log.info(f"[REGRESSION] Model saved to {model_path}")

    # Save training history with all metrics
    history_data = {
        'epoch': range(len(history.history['loss'])),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_mae': history.history['mae'],
        'val_mae': history.history['val_mae'],
    }
    
    # Add additional metrics if available
    if 'rmse' in history.history:
        history_data['train_rmse'] = history.history['rmse']
        history_data['val_rmse'] = history.history['val_rmse']
    if 'mape' in history.history:
        history_data['train_mape'] = history.history['mape']
        history_data['val_mape'] = history.history['val_mape']
    if 'lr' in history.history:
        history_data['learning_rate'] = history.history['lr']
    
    history_df = pd.DataFrame(history_data)
    save_data(
        df=history_df,
        filename=f"{ticker}_training_history_reg.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_trained_model_reg"
    )

    # Save test predictions
    test_results_df = pd.DataFrame({
        'y_true': y_test_seq,
        'y_pred': y_test_pred,
    })
    save_data(
        df=test_results_df,
        filename=f"{ticker}_test_predictions_reg.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_trained_model_reg"
    )

    output_value = {
        "ticker": ticker,
        "model_path": str(model_path),
        "lookback": lookback,
        "n_features": n_features,
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_mape": float(test_mape),
        "best_epoch": len(history.history['loss']),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_mae": float(min(history.history['val_mae'])),
    }

    metadata = {
        "ticker": ticker,
        "model_path": str(model_path),
        "model_type": "Enhanced Bidirectional LSTM Regression" if use_bidirectional else "Enhanced LSTM Regression",
        "lookback": lookback,
        "n_features": n_features,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "test_loss_huber": float(test_loss),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_mape": float(test_mape),
        "train_samples": int(X_train_seq.shape[0]),
        "val_samples": int(X_val_seq.shape[0]),
        "test_samples": int(X_test_seq.shape[0]),
        "total_epochs": len(history.history['loss']),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_mae": float(min(history.history['val_mae'])),
        "final_lr": float(history.history.get('lr', [learning_rate])[-1]) if 'lr' in history.history else learning_rate,
        # Add visualizations to metadata
        "training_history_plot": MetadataValue.path(str(training_plot_path)),
        "predictions_comparison_plot": MetadataValue.path(str(predictions_plot_path)),
        "scatter_plot": MetadataValue.path(str(scatter_plot_path)),
        "preview_training_plot": MetadataValue.md(f"![Training History]({training_plot_path.absolute().as_uri()})"),
        "preview_predictions_plot": MetadataValue.md(f"![Predictions Comparison]({predictions_plot_path.absolute().as_uri()})"),
    }

    return Output(value=output_value, metadata=metadata)


prediction_config_schema_reg = {
    "lookback": Field(int, default_value=30, description="Number of timesteps for LSTM window (must match training)"),
}


@asset(
    name="lstm_predictions_reg",
    group_name="LSTM",
    kinds={"python", "regression"},
    config_schema=prediction_config_schema_reg,
)
def lstm_predictions_reg(context, asset_preprocessed_data, lstm_trained_model_reg):
    """
    Use trained LSTM regression model to make predictions on future data.
    Predicts next-day return and converts to price prediction.
    """
    # Extract config
    lookback = context.op_config.get("lookback", LOOKBACK)

    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    X_predict = asset_preprocessed_data["X_predict"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]

    # Load model
    model_path = lstm_trained_model_reg["model_path"]
    context.log.info(f"[REGRESSION] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Build continuous feature matrix to extract last lookback rows
    X_all = pd.concat([X_train, X_val, X_test, X_predict], axis=0)
    
    if len(X_all) < lookback:
        raise ValueError(f"Not enough rows to build prediction window. Need {lookback}, have {len(X_all)}")

    context.log.info(f"[REGRESSION] Using last {lookback} rows from combined data for prediction")
    
    # Get last lookback rows and reshape for LSTM
    latest_window = X_all.iloc[-lookback:].values.astype(np.float32)
    n_features = latest_window.shape[1]
    latest_window = latest_window.reshape(1, lookback, n_features)

    # Predict next-day return
    next_return_pred = float(model.predict(latest_window, verbose=0)[0, 0])
    context.log.info(f"[REGRESSION] Predicted next-day return: {next_return_pred:.6f}")

    # Convert return to price prediction
    next_price_pred = float(last_close * (1.0 + next_return_pred))
    context.log.info(f"[REGRESSION] Last close price: {last_close:.2f}")
    context.log.info(f"[REGRESSION] Predicted next close price: {next_price_pred:.2f}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_return': [next_return_pred],
        'predicted_price': [next_price_pred],
        'return_percent': [next_return_pred * 100],
    })
    
    save_data(
        df=predictions_df,
        filename=f"{ticker}_latest_predictions_reg.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_predictions_reg"
    )

    log_df(predictions_df, context, 'lstm_predictions_reg')

    output_value = {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "predicted_return": float(next_return_pred),
        "predicted_price": float(next_price_pred),
        "return_percent": float(next_return_pred * 100),
    }

    metadata = {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "predicted_return": float(next_return_pred),
        "predicted_price": float(next_price_pred),
        "return_percent": float(next_return_pred * 100),
        "lookback_used": lookback,
    }

    return Output(value=output_value, metadata=metadata)


# ============================================================================
# CLASSIFICATION MODEL ASSETS
# ============================================================================

training_config_schema_cls = {
    "lookback": Field(int, default_value=30, description="Number of timesteps for LSTM window"),
    "lstm_units": Field(int, default_value=64, description="Number of units in first LSTM layer"),
    "dense_units": Field(int, default_value=32, description="Number of units in first dense layer"),
    "learning_rate": Field(float, default_value=5e-3, description="Initial learning rate for optimizer"),
    "epochs": Field(int, default_value=200, description="Maximum number of training epochs"),
    "batch_size": Field(int, default_value=16, description="Batch size for training"),
    "patience": Field(int, default_value=180, description="Early stopping patience"),
    "use_bidirectional": Field(bool, default_value=False, description="Use bidirectional LSTM layers"),
    "l2_reg": Field(float, default_value=5e-5, description="L2 regularization factor"),
}


@asset(
    name="lstm_trained_model_cls",
    group_name="LSTM",
    kinds={"python", "classification"},
    config_schema=training_config_schema_cls,
)
def lstm_trained_model_cls(context, asset_preprocessed_data):
    """
    Train LSTM classification model on preprocessed data.
    Predicts up/down/flat direction for next-day stock movement.
    Saves the trained model to disk and returns model metadata.
    """
    # Extract config
    lookback = context.op_config.get("lookback", LOOKBACK)
    lstm_units = context.op_config.get("lstm_units", 128)
    dense_units = context.op_config.get("dense_units", 64)
    learning_rate = context.op_config.get("learning_rate", 1e-3)
    epochs = context.op_config.get("epochs", 200)
    batch_size = context.op_config.get("batch_size", 32)
    patience = context.op_config.get("patience", 180)
    use_bidirectional = context.op_config.get("use_bidirectional", True)
    l2_reg = context.op_config.get("l2_reg", 1e-4)

    # Extract data from preprocessing asset
    ticker = asset_preprocessed_data["ticker"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    y_train_cls = asset_preprocessed_data["y_train_cls"]
    y_val_cls = asset_preprocessed_data["y_val_cls"]
    y_test_cls = asset_preprocessed_data["y_test_cls"]

    context.log.info(f"[CLASSIFICATION] Creating sequences with lookback={lookback}")
    
    # Create sequences for LSTM
    X_train_seq, y_train_seq = make_sequences(X_train, y_train_cls, lookback)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val_cls, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test_cls, lookback)

    # Convert labels to 0-indexed for sparse_categorical_crossentropy
    # Original: -1 (down), 0 (flat), 1 (up) → New: 0 (down), 1 (flat), 2 (up)
    y_train_seq = (y_train_seq + 1).astype(int)
    y_val_seq = (y_val_seq + 1).astype(int)
    y_test_seq = (y_test_seq + 1).astype(int)
    
    num_classes = len(np.unique(y_train_seq))
    context.log.info(f"[CLASSIFICATION] Number of classes: {num_classes}")

    context.log.info(f"[CLASSIFICATION] Sequence shapes:")
    context.log.info(f"  X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    context.log.info(f"  X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
    context.log.info(f"  X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

    # Class distribution
    unique, counts = np.unique(y_train_seq, return_counts=True)
    context.log.info(f"[CLASSIFICATION] Class distribution (train): {dict(zip(unique, counts))}")

    # Build model
    n_features = X_train_seq.shape[-1]
    context.log.info(f"[CLASSIFICATION] Building enhanced LSTM model with {n_features} features")
    context.log.info(f"[CLASSIFICATION] Using bidirectional: {use_bidirectional}, L2 reg: {l2_reg}")
    
    model = build_lstm_model_cls(
        lookback=lookback,
        n_features=n_features,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        use_bidirectional=use_bidirectional,
        l2_reg=l2_reg,
    )

    context.log.info("[CLASSIFICATION] Model architecture:")
    model.summary(print_fn=lambda x: context.log.info(x))

    # Calculate class weights for imbalanced data
    class_weights = {}
    for cls in unique:
        class_weights[int(cls)] = len(y_train_seq) / (num_classes * counts[cls])
    context.log.info(f"[CLASSIFICATION] Class weights: {class_weights}")

    # Set up callbacks for improved training
    callback_list = [
        # Early stopping with more patience
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-7,
            verbose=1,
        ),
        # Save best model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / ticker / f"{ticker}_best_model_cls.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode='max',
            verbose=0,
        ),
    ]

    # Train model
    context.log.info(f"[CLASSIFICATION] Training enhanced model for up to {epochs} epochs...")
    context.log.info(f"[CLASSIFICATION] Early stopping patience: {patience}, LR reduction patience: {patience // 3}")
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callback_list,
        verbose=1,
    )

    # Evaluate on test set
    test_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    
    # Get predictions for test set
    y_test_pred_proba = model.predict(X_test_seq, verbose=0)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    # Calculate detailed metrics using scikit-learn
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
    
    test_precision = precision_score(y_test_seq, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_seq, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_seq, y_test_pred, average='weighted', zero_division=0)
    
    context.log.info(f"[CLASSIFICATION] Test Loss: {test_loss:.6f}")
    context.log.info(f"[CLASSIFICATION] Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    context.log.info(f"[CLASSIFICATION] Test Precision (weighted): {test_precision:.4f}")
    context.log.info(f"[CLASSIFICATION] Test Recall (weighted): {test_recall:.4f}")
    context.log.info(f"[CLASSIFICATION] Test F1 Score (weighted): {f1:.4f}")
    
    # Log classification report
    context.log.info("[CLASSIFICATION] Classification Report:")
    report = classification_report(y_test_seq, y_test_pred, 
                                   target_names=['Down', 'Flat', 'Up'],
                                   zero_division=0)
    context.log.info(f"\n{report}")
    
    # Log confusion matrix
    cm = confusion_matrix(y_test_seq, y_test_pred)
    context.log.info(f"[CLASSIFICATION] Confusion Matrix:\n{cm}")

    # Generate visualizations
    context.log.info("[CLASSIFICATION] Generating training history plot...")
    training_plot_path = CHARTS_DIR / ticker / f"{ticker}_training_history_cls.png"
    plot_training_history(history, ticker, "LSTM Classification", training_plot_path)
    
    context.log.info("[CLASSIFICATION] Generating predictions comparison plot...")
    predictions_plot_path = CHARTS_DIR / ticker / f"{ticker}_predictions_comparison_cls.png"
    plot_predictions_comparison(y_test_seq, y_test_pred, ticker, "LSTM Classification", predictions_plot_path)
    
    context.log.info("[CLASSIFICATION] Generating confusion matrix plot...")
    cm_plot_path = CHARTS_DIR / ticker / f"{ticker}_confusion_matrix_cls.png"
    plot_confusion_matrix_chart(cm, ticker, cm_plot_path)

    # Save model
    model_path = MODEL_DIR / ticker / f"{ticker}_lstm_model_cls.keras"
    model.save(model_path)
    context.log.info(f"[CLASSIFICATION] Model saved to {model_path}")

    # Save training history with all metrics
    history_data = {
        'epoch': range(len(history.history['loss'])),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
    }
    
    # Add additional metrics if available
    if 'precision' in history.history:
        history_data['train_precision'] = history.history['precision']
        history_data['val_precision'] = history.history['val_precision']
    if 'recall' in history.history:
        history_data['train_recall'] = history.history['recall']
        history_data['val_recall'] = history.history['val_recall']
    if 'lr' in history.history:
        history_data['learning_rate'] = history.history['lr']
    
    history_df = pd.DataFrame(history_data)
    save_data(
        df=history_df,
        filename=f"{ticker}_training_history_cls.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_trained_model_cls"
    )

    # Save test predictions
    test_results_df = pd.DataFrame({
        'y_true': y_test_seq,
        'y_pred': y_test_pred,
        'prob_down': y_test_pred_proba[:, 0],
        'prob_flat': y_test_pred_proba[:, 1],
        'prob_up': y_test_pred_proba[:, 2],
    })
    save_data(
        df=test_results_df,
        filename=f"{ticker}_test_predictions_cls.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_trained_model_cls"
    )

    output_value = {
        "ticker": ticker,
        "model_path": str(model_path),
        "lookback": lookback,
        "n_features": n_features,
        "num_classes": num_classes,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(f1),
        "best_epoch": len(history.history['loss']),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
    }

    metadata = {
        "ticker": ticker,
        "model_path": str(model_path),
        "model_type": "Enhanced Bidirectional LSTM Classification" if use_bidirectional else "Enhanced LSTM Classification",
        "lookback": lookback,
        "n_features": n_features,
        "num_classes": num_classes,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_precision_weighted": float(test_precision),
        "test_recall_weighted": float(test_recall),
        "test_f1_weighted": float(f1),
        "train_samples": int(X_train_seq.shape[0]),
        "val_samples": int(X_val_seq.shape[0]),
        "test_samples": int(X_test_seq.shape[0]),
        "total_epochs": len(history.history['loss']),
        "best_val_loss": float(min(history.history['val_loss'])),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "final_lr": float(history.history.get('lr', [learning_rate])[-1]) if 'lr' in history.history else learning_rate,
        # Add visualizations to metadata
        "training_history_plot": MetadataValue.path(str(training_plot_path)),
        "predictions_comparison_plot": MetadataValue.path(str(predictions_plot_path)),
        "confusion_matrix_plot": MetadataValue.path(str(cm_plot_path)),
        "preview_training_plot": MetadataValue.md(f"![Training History]({training_plot_path.absolute().as_uri()})"),
        "preview_confusion_matrix": MetadataValue.md(f"![Confusion Matrix]({cm_plot_path.absolute().as_uri()})"),
    }

    return Output(value=output_value, metadata=metadata)


prediction_config_schema_cls = {
    "lookback": Field(int, default_value=30, description="Number of timesteps for LSTM window (must match training)"),
}


@asset(
    name="lstm_predictions_cls",
    group_name="LSTM",
    kinds={"python", "classification"},
    config_schema=prediction_config_schema_cls,
)
def lstm_predictions_cls(context, asset_preprocessed_data, lstm_trained_model_cls):
    """
    Use trained LSTM classification model to make predictions on future data.
    Predicts next-day direction (down/flat/up) with probabilities.
    """
    # Extract config
    lookback = context.op_config.get("lookback", LOOKBACK)

    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    X_predict = asset_preprocessed_data["X_predict"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]

    # Load model
    model_path = lstm_trained_model_cls["model_path"]
    context.log.info(f"[CLASSIFICATION] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Build continuous feature matrix to extract last lookback rows
    X_all = pd.concat([X_train, X_val, X_test, X_predict], axis=0)
    
    if len(X_all) < lookback:
        raise ValueError(f"Not enough rows to build prediction window. Need {lookback}, have {len(X_all)}")

    context.log.info(f"[CLASSIFICATION] Using last {lookback} rows from combined data for prediction")
    
    # Get last lookback rows and reshape for LSTM
    latest_window = X_all.iloc[-lookback:].values.astype(np.float32)
    n_features = latest_window.shape[1]
    latest_window = latest_window.reshape(1, lookback, n_features)

    # Predict probabilities for each class
    pred_proba = model.predict(latest_window, verbose=0)[0]
    pred_class = int(np.argmax(pred_proba))
    
    # Convert back to original labels: 0→-1 (down), 1→0 (flat), 2→1 (up)
    pred_label = pred_class - 1
    
    # Map to direction names
    direction_map = {-1: 'Down', 0: 'Flat', 1: 'Up'}
    pred_direction = direction_map[pred_label]
    
    prob_down = float(pred_proba[0])
    prob_flat = float(pred_proba[1])
    prob_up = float(pred_proba[2])
    confidence = float(np.max(pred_proba))
    
    context.log.info(f"[CLASSIFICATION] Predicted direction: {pred_direction}")
    context.log.info(f"[CLASSIFICATION] Confidence: {confidence:.2%}")
    context.log.info(f"[CLASSIFICATION] Probabilities - Down: {prob_down:.2%}, Flat: {prob_flat:.2%}, Up: {prob_up:.2%}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_direction': [pred_direction],
        'predicted_label': [pred_label],
        'confidence': [confidence],
        'prob_down': [prob_down],
        'prob_flat': [prob_flat],
        'prob_up': [prob_up],
    })
    
    save_data(
        df=predictions_df,
        filename=f"{ticker}_latest_predictions_cls.csv",
        dir=f"models/{ticker}",
        context=context,
        asset="lstm_predictions_cls"
    )

    log_df(predictions_df, context, 'lstm_predictions_cls')

    output_value = {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "predicted_direction": pred_direction,
        "predicted_label": int(pred_label),
        "confidence": float(confidence),
        "prob_down": float(prob_down),
        "prob_flat": float(prob_flat),
        "prob_up": float(prob_up),
    }

    metadata = {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "predicted_direction": pred_direction,
        "predicted_label": int(pred_label),
        "confidence": float(confidence),
        "prob_down": float(prob_down),
        "prob_flat": float(prob_flat),
        "prob_up": float(prob_up),
        "lookback_used": lookback,
    }

    return Output(value=output_value, metadata=metadata)
