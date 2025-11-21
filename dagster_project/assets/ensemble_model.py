import numpy as np
import pandas as pd
from pathlib import Path
from dagster import asset, Output, Field, MetadataValue
import pickle
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .methods.save_data import save_data
from .methods.logging import log_df

# Configuration
MODEL_DIR = Path("models/ensemble")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = Path("models/ensemble/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_ensemble_comparison(y_true, y_lstm, y_xgb, y_ensemble, ticker: str, model_type: str, save_path: Path):
    """Compare LSTM, XGBoost, and Ensemble predictions."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All predictions
    ax1 = axes[0, 0]
    ax1.plot(y_true, label='True', linewidth=2.5, alpha=0.8, color='black')
    ax1.plot(y_lstm, label='LSTM', linewidth=1.5, alpha=0.7, linestyle='--')
    ax1.plot(y_xgb, label='XGBoost', linewidth=1.5, alpha=0.7, linestyle='--')
    ax1.plot(y_ensemble, label='Ensemble', linewidth=2, alpha=0.9)
    ax1.set_xlabel('Sample Index (Test Set)', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'{ticker} {model_type} - All Models Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Last N predictions (zoomed)
    N = min(100, len(y_true))
    ax2 = axes[0, 1]
    ax2.plot(range(-N, 0), y_true[-N:], label='True', linewidth=2.5, alpha=0.8, color='black')
    ax2.plot(range(-N, 0), y_lstm[-N:], label='LSTM', linewidth=1.5, alpha=0.7, linestyle='--')
    ax2.plot(range(-N, 0), y_xgb[-N:], label='XGBoost', linewidth=1.5, alpha=0.7, linestyle='--')
    ax2.plot(range(-N, 0), y_ensemble[-N:], label='Ensemble', linewidth=2, alpha=0.9)
    ax2.set_xlabel('Relative Index (Last N)', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'Last {N} Predictions (Zoomed)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error comparison (regression) or Agreement (classification)
    ax3 = axes[1, 0]
    if 'Regression' in model_type:
        lstm_errors = np.abs(y_true - y_lstm)
        xgb_errors = np.abs(y_true - y_xgb)
        ensemble_errors = np.abs(y_true - y_ensemble)
        
        ax3.plot(lstm_errors, label='LSTM MAE', alpha=0.7, linewidth=1.5)
        ax3.plot(xgb_errors, label='XGBoost MAE', alpha=0.7, linewidth=1.5)
        ax3.plot(ensemble_errors, label='Ensemble MAE', alpha=0.9, linewidth=2)
        ax3.set_ylabel('Absolute Error', fontsize=12)
        ax3.set_title('Absolute Errors Comparison', fontsize=14, fontweight='bold')
    else:
        # Classification: Show agreement
        lstm_correct = (y_true == y_lstm).astype(int)
        xgb_correct = (y_true == y_xgb).astype(int)
        ensemble_correct = (y_true == y_ensemble).astype(int)
        
        window = min(50, len(y_true) // 10)
        lstm_rolling = pd.Series(lstm_correct).rolling(window).mean()
        xgb_rolling = pd.Series(xgb_correct).rolling(window).mean()
        ensemble_rolling = pd.Series(ensemble_correct).rolling(window).mean()
        
        ax3.plot(lstm_rolling, label='LSTM Accuracy', alpha=0.7, linewidth=1.5)
        ax3.plot(xgb_rolling, label='XGBoost Accuracy', alpha=0.7, linewidth=1.5)
        ax3.plot(ensemble_rolling, label='Ensemble Accuracy', alpha=0.9, linewidth=2)
        ax3.set_ylabel(f'Rolling Accuracy (window={window})', fontsize=12)
        ax3.set_title('Rolling Accuracy Comparison', fontsize=14, fontweight='bold')
    
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics bar chart
    ax4 = axes[1, 1]
    if 'Regression' in model_type:
        mae_lstm = mean_absolute_error(y_true, y_lstm)
        mae_xgb = mean_absolute_error(y_true, y_xgb)
        mae_ensemble = mean_absolute_error(y_true, y_ensemble)
        
        rmse_lstm = np.sqrt(mean_squared_error(y_true, y_lstm))
        rmse_xgb = np.sqrt(mean_squared_error(y_true, y_xgb))
        rmse_ensemble = np.sqrt(mean_squared_error(y_true, y_ensemble))
        
        metrics = ['MAE', 'RMSE']
        lstm_vals = [mae_lstm, rmse_lstm]
        xgb_vals = [mae_xgb, rmse_xgb]
        ensemble_vals = [mae_ensemble, rmse_ensemble]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        ax4.bar(x - width, lstm_vals, width, label='LSTM', alpha=0.7)
        ax4.bar(x, xgb_vals, width, label='XGBoost', alpha=0.7)
        ax4.bar(x + width, ensemble_vals, width, label='Ensemble', alpha=0.9)
        
        ax4.set_ylabel('Error', fontsize=12)
        ax4.set_title('Performance Metrics (Lower is Better)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
    else:
        acc_lstm = accuracy_score(y_true, y_lstm)
        acc_xgb = accuracy_score(y_true, y_xgb)
        acc_ensemble = accuracy_score(y_true, y_ensemble)
        
        prec_lstm = precision_score(y_true, y_lstm, average='weighted', zero_division=0)
        prec_xgb = precision_score(y_true, y_xgb, average='weighted', zero_division=0)
        prec_ensemble = precision_score(y_true, y_ensemble, average='weighted', zero_division=0)
        
        rec_lstm = recall_score(y_true, y_lstm, average='weighted', zero_division=0)
        rec_xgb = recall_score(y_true, y_xgb, average='weighted', zero_division=0)
        rec_ensemble = recall_score(y_true, y_ensemble, average='weighted', zero_division=0)
        
        f1_lstm = f1_score(y_true, y_lstm, average='weighted', zero_division=0)
        f1_xgb = f1_score(y_true, y_xgb, average='weighted', zero_division=0)
        f1_ensemble = f1_score(y_true, y_ensemble, average='weighted', zero_division=0)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        lstm_vals = [acc_lstm, prec_lstm, rec_lstm, f1_lstm]
        xgb_vals = [acc_xgb, prec_xgb, rec_xgb, f1_xgb]
        ensemble_vals = [acc_ensemble, prec_ensemble, rec_ensemble, f1_ensemble]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        ax4.bar(x - width, lstm_vals, width, label='LSTM', alpha=0.7)
        ax4.bar(x, xgb_vals, width, label='XGBoost', alpha=0.7)
        ax4.bar(x + width, ensemble_vals, width, label='Ensemble', alpha=0.9)
        
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Performance Metrics (Higher is Better)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
    
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


# ============================================================================
# ENSEMBLE REGRESSION ASSET
# ============================================================================

ensemble_config_schema_reg = {
    "lstm_weight": Field(float, default_value=0.3, description="Weight for LSTM predictions (0-1)"),
    "xgb_weight": Field(float, default_value=0.7, description="Weight for XGBoost predictions (0-1). Should sum to 1 with LSTM weight."),
    "strategy": Field(str, default_value="weighted", description="Ensemble strategy: 'weighted', 'average', 'median', or 'adaptive'"),
}


@asset(
    name="ensemble_predictions_reg",
    group_name="Ensemble",
    kinds={"python", "regression"},
    config_schema=ensemble_config_schema_reg,
)
def ensemble_predictions_reg(context, asset_preprocessed_data, lstm_trained_model_reg, xgb_trained_model_reg):
    """
    Ensemble regression model combining LSTM and XGBoost predictions.
    
    Strategies:
    - 'weighted': Weighted average (default: 30% LSTM, 70% XGBoost)
    - 'average': Simple average (50/50)
    - 'median': Median of predictions
    - 'adaptive': Weight based on recent performance
    """
    # Extract config
    lstm_weight = context.op_config.get("lstm_weight", 0.3)
    xgb_weight = context.op_config.get("xgb_weight", 0.7)
    strategy = context.op_config.get("strategy", "weighted")
    
    # Normalize weights
    total_weight = lstm_weight + xgb_weight
    lstm_weight = lstm_weight / total_weight
    xgb_weight = xgb_weight / total_weight
    
    context.log.info(f"[ENSEMBLE REG] Strategy: {strategy}")
    context.log.info(f"[ENSEMBLE REG] Weights - LSTM: {lstm_weight:.2f}, XGBoost: {xgb_weight:.2f}")
    
    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_test = asset_preprocessed_data["X_test"]
    X_predict = asset_preprocessed_data["X_predict"]
    y_test_reg = asset_preprocessed_data["y_test_reg"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]
    
    # Load LSTM model and get predictions
    lstm_model_path = lstm_trained_model_reg["model_path"]
    lstm_lookback = lstm_trained_model_reg["lookback"]
    context.log.info(f"[ENSEMBLE REG] Loading LSTM model from {lstm_model_path}")
    
    from .lstm_model import make_sequences, anti_collapse_loss
    
    custom_objects = {'loss': anti_collapse_loss}
    lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects=custom_objects)
    
    X_test_lstm, y_test_lstm = make_sequences(X_test, y_test_reg, lstm_lookback)
    y_test_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    # Load XGBoost model and get predictions
    xgb_model_path = xgb_trained_model_reg["model_path"]
    context.log.info(f"[ENSEMBLE REG] Loading XGBoost model from {xgb_model_path}")
    
    with open(xgb_model_path, 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Align test sets (XGBoost has more samples due to no lookback)
    y_test_pred_xgb_full = xgb_model.predict(X_test)
    y_test_pred_xgb = y_test_pred_xgb_full[-len(y_test_lstm):]  # Align with LSTM
    
    # Create ensemble predictions on test set
    if strategy == "weighted":
        y_test_pred_ensemble = lstm_weight * y_test_pred_lstm + xgb_weight * y_test_pred_xgb
    elif strategy == "average":
        y_test_pred_ensemble = (y_test_pred_lstm + y_test_pred_xgb) / 2
    elif strategy == "median":
        y_test_pred_ensemble = np.median(np.stack([y_test_pred_lstm, y_test_pred_xgb]), axis=0)
    elif strategy == "adaptive":
        # Adaptive: Weight by inverse MAE on rolling window
        window = min(50, len(y_test_lstm) // 5)
        weights = np.ones((len(y_test_lstm), 2))  # [lstm_weight, xgb_weight]
        
        for i in range(window, len(y_test_lstm)):
            mae_lstm = mean_absolute_error(y_test_lstm[i-window:i], y_test_pred_lstm[i-window:i])
            mae_xgb = mean_absolute_error(y_test_lstm[i-window:i], y_test_pred_xgb[i-window:i])
            
            # Inverse MAE weighting
            w_lstm = (1 / (mae_lstm + 1e-6))
            w_xgb = (1 / (mae_xgb + 1e-6))
            total = w_lstm + w_xgb
            
            weights[i, 0] = w_lstm / total
            weights[i, 1] = w_xgb / total
        
        y_test_pred_ensemble = (weights[:, 0] * y_test_pred_lstm + 
                                weights[:, 1] * y_test_pred_xgb)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate metrics
    mae_lstm = mean_absolute_error(y_test_lstm, y_test_pred_lstm)
    mae_xgb = mean_absolute_error(y_test_lstm, y_test_pred_xgb)
    mae_ensemble = mean_absolute_error(y_test_lstm, y_test_pred_ensemble)
    
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_test_pred_lstm))
    rmse_xgb = np.sqrt(mean_squared_error(y_test_lstm, y_test_pred_xgb))
    rmse_ensemble = np.sqrt(mean_squared_error(y_test_lstm, y_test_pred_ensemble))
    
    r2_lstm = r2_score(y_test_lstm, y_test_pred_lstm)
    r2_xgb = r2_score(y_test_lstm, y_test_pred_xgb)
    r2_ensemble = r2_score(y_test_lstm, y_test_pred_ensemble)
    
    context.log.info(f"[ENSEMBLE REG] LSTM      - MAE: {mae_lstm:.6f}, RMSE: {rmse_lstm:.6f}, R²: {r2_lstm:.4f}")
    context.log.info(f"[ENSEMBLE REG] XGBoost   - MAE: {mae_xgb:.6f}, RMSE: {rmse_xgb:.6f}, R²: {r2_xgb:.4f}")
    context.log.info(f"[ENSEMBLE REG] Ensemble  - MAE: {mae_ensemble:.6f}, RMSE: {rmse_ensemble:.6f}, R²: {r2_ensemble:.4f}")
    
    improvement_vs_lstm = ((mae_lstm - mae_ensemble) / mae_lstm) * 100
    improvement_vs_xgb = ((mae_xgb - mae_ensemble) / mae_xgb) * 100
    context.log.info(f"[ENSEMBLE REG] Improvement vs LSTM: {improvement_vs_lstm:.2f}%")
    context.log.info(f"[ENSEMBLE REG] Improvement vs XGBoost: {improvement_vs_xgb:.2f}%")
    
    # Predict next day
    # Get LSTM prediction
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_all = pd.concat([X_train, X_val, X_test, X_predict], axis=0)
    latest_window_lstm = X_all.iloc[-lstm_lookback:].values.astype(np.float32)
    latest_window_lstm = latest_window_lstm.reshape(1, lstm_lookback, latest_window_lstm.shape[1])
    next_return_lstm = float(lstm_model.predict(latest_window_lstm, verbose=0)[0, 0])
    
    # Get XGBoost prediction
    next_return_xgb = float(xgb_model.predict(X_predict)[0])
    
    # Ensemble prediction
    if strategy == "weighted":
        next_return_ensemble = lstm_weight * next_return_lstm + xgb_weight * next_return_xgb
    elif strategy in ["average", "median"]:
        next_return_ensemble = (next_return_lstm + next_return_xgb) / 2
    elif strategy == "adaptive":
        # Use last adaptive weights
        next_return_ensemble = (weights[-1, 0] * next_return_lstm + 
                               weights[-1, 1] * next_return_xgb)
    
    next_price_ensemble = float(last_close * (1.0 + next_return_ensemble))
    
    context.log.info(f"[ENSEMBLE REG] Next-day predictions:")
    context.log.info(f"  LSTM return: {next_return_lstm:.6f}")
    context.log.info(f"  XGBoost return: {next_return_xgb:.6f}")
    context.log.info(f"  Ensemble return: {next_return_ensemble:.6f}")
    context.log.info(f"  Ensemble price: {next_price_ensemble:.2f} (from {last_close:.2f})")
    
    # Generate comparison chart
    context.log.info("[ENSEMBLE REG] Generating comparison chart...")
    comparison_plot_path = CHARTS_DIR / ticker / f"{ticker}_ensemble_comparison_reg.png"
    plot_ensemble_comparison(y_test_lstm, y_test_pred_lstm, y_test_pred_xgb, 
                             y_test_pred_ensemble, ticker, "Ensemble Regression", 
                             comparison_plot_path)
    
    # Save test predictions
    test_results_df = pd.DataFrame({
        'y_true': y_test_lstm,
        'y_pred_lstm': y_test_pred_lstm,
        'y_pred_xgb': y_test_pred_xgb,
        'y_pred_ensemble': y_test_pred_ensemble,
        'error_lstm': np.abs(y_test_lstm - y_test_pred_lstm),
        'error_xgb': np.abs(y_test_lstm - y_test_pred_xgb),
        'error_ensemble': np.abs(y_test_lstm - y_test_pred_ensemble),
    })
    save_data(test_results_df, f"{ticker}_ensemble_test_predictions_reg.csv",
              f"models/ensemble/{ticker}", context, "ensemble_predictions_reg")
    
    # Save latest prediction
    latest_pred_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_return_lstm': [next_return_lstm],
        'predicted_return_xgb': [next_return_xgb],
        'predicted_return_ensemble': [next_return_ensemble],
        'predicted_price_ensemble': [next_price_ensemble],
        'strategy': [strategy],
        'lstm_weight': [lstm_weight],
        'xgb_weight': [xgb_weight],
    })
    save_data(latest_pred_df, f"{ticker}_ensemble_latest_prediction_reg.csv",
              f"models/ensemble/{ticker}", context, "ensemble_predictions_reg")
    
    output_value = {
        "ticker": ticker,
        "strategy": strategy,
        "lstm_weight": float(lstm_weight),
        "xgb_weight": float(xgb_weight),
        "mae_lstm": float(mae_lstm),
        "mae_xgb": float(mae_xgb),
        "mae_ensemble": float(mae_ensemble),
        "improvement_vs_lstm": float(improvement_vs_lstm),
        "improvement_vs_xgb": float(improvement_vs_xgb),
        "next_return_ensemble": float(next_return_ensemble),
        "next_price_ensemble": float(next_price_ensemble),
    }
    
    metadata = {
        "ticker": ticker,
        "strategy": strategy,
        "lstm_weight": float(lstm_weight),
        "xgb_weight": float(xgb_weight),
        "mae_lstm": float(mae_lstm),
        "mae_xgb": float(mae_xgb),
        "mae_ensemble": float(mae_ensemble),
        "rmse_lstm": float(rmse_lstm),
        "rmse_xgb": float(rmse_xgb),
        "rmse_ensemble": float(rmse_ensemble),
        "r2_lstm": float(r2_lstm),
        "r2_xgb": float(r2_xgb),
        "r2_ensemble": float(r2_ensemble),
        "improvement_vs_lstm_pct": float(improvement_vs_lstm),
        "improvement_vs_xgb_pct": float(improvement_vs_xgb),
        "next_return_ensemble": float(next_return_ensemble),
        "next_price_ensemble": float(next_price_ensemble),
        "comparison_plot": MetadataValue.path(str(comparison_plot_path)),
    }
    
    return Output(value=output_value, metadata=metadata)


# ============================================================================
# ENSEMBLE CLASSIFICATION ASSET
# ============================================================================

ensemble_config_schema_cls = {
    "strategy": Field(str, default_value="soft_voting", description="Ensemble strategy: 'soft_voting', 'hard_voting', 'weighted_soft', or 'stacking'"),
    "lstm_weight": Field(float, default_value=0.4, description="Weight for LSTM (used in weighted_soft strategy)"),
    "xgb_weight": Field(float, default_value=0.6, description="Weight for XGBoost (used in weighted_soft strategy)"),
}


@asset(
    name="ensemble_predictions_cls",
    group_name="Ensemble",
    kinds={"python", "classification"},
    config_schema=ensemble_config_schema_cls,
)
def ensemble_predictions_cls(context, asset_preprocessed_data, lstm_trained_model_cls, xgb_trained_model_cls):
    """
    Ensemble classification model combining LSTM and XGBoost predictions.
    
    Strategies:
    - 'soft_voting': Average predicted probabilities (default)
    - 'hard_voting': Majority vote on predicted classes
    - 'weighted_soft': Weighted average of probabilities
    - 'stacking': Use model performance as confidence
    """
    # Extract config
    strategy = context.op_config.get("strategy", "soft_voting")
    lstm_weight = context.op_config.get("lstm_weight", 0.4)
    xgb_weight = context.op_config.get("xgb_weight", 0.6)
    
    # Normalize weights
    total_weight = lstm_weight + xgb_weight
    lstm_weight = lstm_weight / total_weight
    xgb_weight = xgb_weight / total_weight
    
    context.log.info(f"[ENSEMBLE CLS] Strategy: {strategy}")
    if strategy == "weighted_soft":
        context.log.info(f"[ENSEMBLE CLS] Weights - LSTM: {lstm_weight:.2f}, XGBoost: {xgb_weight:.2f}")
    
    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_test = asset_preprocessed_data["X_test"]
    X_predict = asset_preprocessed_data["X_predict"]
    y_test_cls = asset_preprocessed_data["y_test_cls"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]
    
    # Load LSTM model
    lstm_model_path = lstm_trained_model_cls["model_path"]
    lstm_lookback = lstm_trained_model_cls["lookback"]
    num_classes = lstm_trained_model_cls["num_classes"]
    context.log.info(f"[ENSEMBLE CLS] Loading LSTM model from {lstm_model_path}")
    
    from .lstm_model import make_sequences
    
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    X_test_lstm, y_test_lstm = make_sequences(X_test, y_test_cls, lstm_lookback)
    
    # Get LSTM predictions
    y_test_pred_proba_lstm_raw = lstm_model.predict(X_test_lstm, verbose=0)
    
    # Handle binary vs multi-class
    if num_classes == 2:
        if len(y_test_pred_proba_lstm_raw.shape) == 2 and y_test_pred_proba_lstm_raw.shape[1] == 1:
            # Sigmoid output: (n, 1)
            prob_up_lstm = y_test_pred_proba_lstm_raw.flatten()
            prob_down_lstm = 1.0 - prob_up_lstm
            y_test_pred_proba_lstm = np.column_stack([prob_down_lstm, prob_up_lstm])
        else:
            y_test_pred_proba_lstm = y_test_pred_proba_lstm_raw
        y_test_pred_lstm = (y_test_pred_proba_lstm[:, 1] > 0.5).astype(int)
    else:
        y_test_pred_proba_lstm = y_test_pred_proba_lstm_raw
        y_test_pred_lstm = np.argmax(y_test_pred_proba_lstm, axis=1)
    
    # Load XGBoost model
    xgb_model_path = xgb_trained_model_cls["model_path"]
    context.log.info(f"[ENSEMBLE CLS] Loading XGBoost model from {xgb_model_path}")
    
    with open(xgb_model_path, 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Align test sets
    y_test_pred_xgb_full = xgb_model.predict(X_test)
    y_test_pred_proba_xgb_full = xgb_model.predict_proba(X_test)
    
    y_test_pred_xgb = y_test_pred_xgb_full[-len(y_test_lstm):]
    y_test_pred_proba_xgb = y_test_pred_proba_xgb_full[-len(y_test_lstm):]
    
    # Create ensemble predictions
    if strategy == "soft_voting":
        # Average probabilities
        y_test_pred_proba_ensemble = (y_test_pred_proba_lstm + y_test_pred_proba_xgb) / 2
        y_test_pred_ensemble = np.argmax(y_test_pred_proba_ensemble, axis=1)
    
    elif strategy == "weighted_soft":
        # Weighted average of probabilities
        y_test_pred_proba_ensemble = (lstm_weight * y_test_pred_proba_lstm + 
                                      xgb_weight * y_test_pred_proba_xgb)
        y_test_pred_ensemble = np.argmax(y_test_pred_proba_ensemble, axis=1)
    
    elif strategy == "hard_voting":
        # Majority vote
        votes = np.column_stack([y_test_pred_lstm, y_test_pred_xgb])
        y_test_pred_ensemble = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=1, 
            arr=votes
        )
        y_test_pred_proba_ensemble = (y_test_pred_proba_lstm + y_test_pred_proba_xgb) / 2
    
    elif strategy == "stacking":
        # Weight by model accuracy on validation set
        acc_lstm = accuracy_score(y_test_lstm, y_test_pred_lstm)
        acc_xgb = accuracy_score(y_test_lstm, y_test_pred_xgb)
        
        total_acc = acc_lstm + acc_xgb
        w_lstm = acc_lstm / total_acc
        w_xgb = acc_xgb / total_acc
        
        context.log.info(f"[ENSEMBLE CLS] Stacking weights - LSTM: {w_lstm:.3f}, XGBoost: {w_xgb:.3f}")
        
        y_test_pred_proba_ensemble = w_lstm * y_test_pred_proba_lstm + w_xgb * y_test_pred_proba_xgb
        y_test_pred_ensemble = np.argmax(y_test_pred_proba_ensemble, axis=1)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate metrics
    acc_lstm = accuracy_score(y_test_lstm, y_test_pred_lstm)
    acc_xgb = accuracy_score(y_test_lstm, y_test_pred_xgb)
    acc_ensemble = accuracy_score(y_test_lstm, y_test_pred_ensemble)
    
    prec_lstm = precision_score(y_test_lstm, y_test_pred_lstm, average='weighted', zero_division=0)
    prec_xgb = precision_score(y_test_lstm, y_test_pred_xgb, average='weighted', zero_division=0)
    prec_ensemble = precision_score(y_test_lstm, y_test_pred_ensemble, average='weighted', zero_division=0)
    
    rec_lstm = recall_score(y_test_lstm, y_test_pred_lstm, average='weighted', zero_division=0)
    rec_xgb = recall_score(y_test_lstm, y_test_pred_xgb, average='weighted', zero_division=0)
    rec_ensemble = recall_score(y_test_lstm, y_test_pred_ensemble, average='weighted', zero_division=0)
    
    f1_lstm = f1_score(y_test_lstm, y_test_pred_lstm, average='weighted', zero_division=0)
    f1_xgb = f1_score(y_test_lstm, y_test_pred_xgb, average='weighted', zero_division=0)
    f1_ensemble = f1_score(y_test_lstm, y_test_pred_ensemble, average='weighted', zero_division=0)
    
    context.log.info(f"[ENSEMBLE CLS] LSTM     - Acc: {acc_lstm:.4f}, Prec: {prec_lstm:.4f}, Rec: {rec_lstm:.4f}, F1: {f1_lstm:.4f}")
    context.log.info(f"[ENSEMBLE CLS] XGBoost  - Acc: {acc_xgb:.4f}, Prec: {prec_xgb:.4f}, Rec: {rec_xgb:.4f}, F1: {f1_xgb:.4f}")
    context.log.info(f"[ENSEMBLE CLS] Ensemble - Acc: {acc_ensemble:.4f}, Prec: {prec_ensemble:.4f}, Rec: {rec_ensemble:.4f}, F1: {f1_ensemble:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_lstm, y_test_pred_ensemble)
    
    # Predict next day
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_all = pd.concat([X_train, X_val, X_test, X_predict], axis=0)
    
    # LSTM prediction
    latest_window_lstm = X_all.iloc[-lstm_lookback:].values.astype(np.float32)
    latest_window_lstm = latest_window_lstm.reshape(1, lstm_lookback, latest_window_lstm.shape[1])
    pred_proba_lstm_raw = lstm_model.predict(latest_window_lstm, verbose=0)[0]
    
    if num_classes == 2:
        if len(pred_proba_lstm_raw.shape) == 0 or pred_proba_lstm_raw.shape == ():
            prob_up_lstm = float(pred_proba_lstm_raw)
            prob_down_lstm = 1.0 - prob_up_lstm
            pred_proba_lstm = np.array([prob_down_lstm, prob_up_lstm])
        elif len(pred_proba_lstm_raw) == 1:
            prob_up_lstm = float(pred_proba_lstm_raw[0])
            prob_down_lstm = 1.0 - prob_up_lstm
            pred_proba_lstm = np.array([prob_down_lstm, prob_up_lstm])
        else:
            pred_proba_lstm = pred_proba_lstm_raw
    else:
        pred_proba_lstm = pred_proba_lstm_raw
    
    # XGBoost prediction
    pred_proba_xgb = xgb_model.predict_proba(X_predict)[0]
    
    # Ensemble prediction
    if strategy == "soft_voting":
        pred_proba_ensemble = (pred_proba_lstm + pred_proba_xgb) / 2
    elif strategy == "weighted_soft":
        pred_proba_ensemble = lstm_weight * pred_proba_lstm + xgb_weight * pred_proba_xgb
    elif strategy == "hard_voting":
        pred_class_lstm = np.argmax(pred_proba_lstm)
        pred_class_xgb = np.argmax(pred_proba_xgb)
        pred_class_ensemble = int(np.bincount([pred_class_lstm, pred_class_xgb]).argmax())
        pred_proba_ensemble = (pred_proba_lstm + pred_proba_xgb) / 2
    elif strategy == "stacking":
        pred_proba_ensemble = w_lstm * pred_proba_lstm + w_xgb * pred_proba_xgb
    
    pred_class_ensemble = int(np.argmax(pred_proba_ensemble))
    confidence_ensemble = float(np.max(pred_proba_ensemble))
    
    direction_map = {0: 'Down', 1: 'Up'} if num_classes == 2 else {0: 'Down', 1: 'Flat', 2: 'Up'}
    pred_direction_ensemble = direction_map[pred_class_ensemble]
    
    context.log.info(f"[ENSEMBLE CLS] Next-day prediction: {pred_direction_ensemble} (confidence: {confidence_ensemble:.2%})")
    
    # Generate comparison chart
    context.log.info("[ENSEMBLE CLS] Generating comparison chart...")
    comparison_plot_path = CHARTS_DIR / ticker / f"{ticker}_ensemble_comparison_cls.png"
    plot_ensemble_comparison(y_test_lstm, y_test_pred_lstm, y_test_pred_xgb, 
                             y_test_pred_ensemble, ticker, "Ensemble Classification", 
                             comparison_plot_path)
    
    # Save test predictions
    test_results_df = pd.DataFrame({
        'y_true': y_test_lstm,
        'y_pred_lstm': y_test_pred_lstm,
        'y_pred_xgb': y_test_pred_xgb,
        'y_pred_ensemble': y_test_pred_ensemble,
    })
    
    # Add probabilities
    for i in range(num_classes):
        test_results_df[f'prob_class{i}_lstm'] = y_test_pred_proba_lstm[:, i]
        test_results_df[f'prob_class{i}_xgb'] = y_test_pred_proba_xgb[:, i]
        test_results_df[f'prob_class{i}_ensemble'] = y_test_pred_proba_ensemble[:, i]
    
    save_data(test_results_df, f"{ticker}_ensemble_test_predictions_cls.csv",
              f"models/ensemble/{ticker}", context, "ensemble_predictions_cls")
    
    # Save latest prediction
    latest_pred_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_direction': [pred_direction_ensemble],
        'predicted_class': [pred_class_ensemble],
        'confidence': [confidence_ensemble],
        'strategy': [strategy],
    })
    
    for i in range(num_classes):
        latest_pred_df[f'prob_class{i}'] = [pred_proba_ensemble[i]]
    
    save_data(latest_pred_df, f"{ticker}_ensemble_latest_prediction_cls.csv",
              f"models/ensemble/{ticker}", context, "ensemble_predictions_cls")
    
    output_value = {
        "ticker": ticker,
        "strategy": strategy,
        "acc_lstm": float(acc_lstm),
        "acc_xgb": float(acc_xgb),
        "acc_ensemble": float(acc_ensemble),
        "predicted_direction": pred_direction_ensemble,
        "predicted_class": int(pred_class_ensemble),
        "confidence": float(confidence_ensemble),
    }
    
    metadata = {
        "ticker": ticker,
        "strategy": strategy,
        "acc_lstm": float(acc_lstm),
        "acc_xgb": float(acc_xgb),
        "acc_ensemble": float(acc_ensemble),
        "prec_lstm": float(prec_lstm),
        "prec_xgb": float(prec_xgb),
        "prec_ensemble": float(prec_ensemble),
        "rec_lstm": float(rec_lstm),
        "rec_xgb": float(rec_xgb),
        "rec_ensemble": float(rec_ensemble),
        "f1_lstm": float(f1_lstm),
        "f1_xgb": float(f1_xgb),
        "f1_ensemble": float(f1_ensemble),
        "predicted_direction": pred_direction_ensemble,
        "predicted_class": int(pred_class_ensemble),
        "confidence": float(confidence_ensemble),
        "comparison_plot": MetadataValue.path(str(comparison_plot_path)),
    }
    
    return Output(value=output_value, metadata=metadata)

