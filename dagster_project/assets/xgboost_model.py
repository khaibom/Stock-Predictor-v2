import numpy as np
import pandas as pd
from pathlib import Path
from dagster import asset, Output, Field, MetadataValue
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .methods.save_data import save_data
from .methods.logging import log_df

# For reproducibility
np.random.seed(42)

# Configuration
MODEL_DIR = Path("models/xgb")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = Path("models/xgb/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_xgb_training_history(evals_result, ticker: str, model_type: str, save_path: Path):
    """Plot XGBoost training/validation metrics over iterations."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # XGBoost 2.x uses 'validation_0', 'validation_1' instead of 'train', 'validation'
    eval_set_keys = list(evals_result.keys())
    if len(eval_set_keys) < 2:
        # Fallback if only one eval set
        eval_set_keys = [eval_set_keys[0], eval_set_keys[0]]
    
    train_key = eval_set_keys[0]  # 'validation_0' = train set
    val_key = eval_set_keys[1]     # 'validation_1' = validation set
    
    # Get metric name (first metric available)
    train_metrics = list(evals_result[train_key].keys())
    metric_name = train_metrics[0] if train_metrics else 'error'
    
    # Plot 1: Training metric
    ax1 = axes[0]
    train_vals = evals_result[train_key][metric_name]
    val_vals = evals_result[val_key][metric_name]
    
    ax1.plot(train_vals, label='Train', linewidth=2)
    ax1.plot(val_vals, label='Validation', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel(metric_name.upper(), fontsize=12)
    ax1.set_title(f'{ticker} {model_type} - Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed on last iterations
    last_n = min(100, len(train_vals))
    ax2 = axes[1]
    ax2.plot(range(len(train_vals)-last_n, len(train_vals)), train_vals[-last_n:], 
             label='Train', linewidth=2)
    ax2.plot(range(len(val_vals)-last_n, len(val_vals)), val_vals[-last_n:], 
             label='Validation', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel(metric_name.upper(), fontsize=12)
    ax2.set_title(f'Last {last_n} Iterations (Zoomed)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_feature_importance(model, feature_names, ticker: str, model_type: str, save_path: Path, top_n=20):
    """Plot top N most important features."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{ticker} {model_type} - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(save_path), feature_importance_df


def plot_predictions_comparison(y_true, y_pred, ticker: str, model_type: str, save_path: Path):
    """Plot true vs predicted values over time."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 5))
    
    # Plot 1: All predictions
    plt.subplot(1, 2, 1)
    plt.plot(y_true, label='True', linewidth=2, alpha=0.7)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Sample Index (Test Set)', fontsize=12)
    plt.ylabel('Return' if 'Regression' in model_type else 'Class', fontsize=12)
    plt.title(f'{ticker} {model_type} - All Test Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Last N predictions
    N = min(200, len(y_true))
    plt.subplot(1, 2, 2)
    plt.plot(range(-N, 0), y_true[-N:], label='True', linewidth=2, alpha=0.7)
    plt.plot(range(-N, 0), y_pred[-N:], label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Relative Index (Last N Samples)', fontsize=12)
    plt.ylabel('Return' if 'Regression' in model_type else 'Class', fontsize=12)
    plt.title(f'Last {N} Test Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_scatter_comparison(y_true, y_pred, ticker: str, model_type: str, save_path: Path):
    """Scatter plot of true vs predicted values (regression only)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{ticker} {model_type} - True vs Predicted (Scatter)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_confusion_matrix_chart(cm, ticker: str, save_path: Path, labels=['Down', 'Up']):
    """Plot confusion matrix for classification."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    figsize = (6, 5) if len(labels) == 2 else (8, 6)
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{ticker} Classification - Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_roc_curve_chart(y_true, y_pred_proba, ticker: str, save_path: Path):
    """Plot ROC curve for binary classification."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{ticker} Classification - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


# ============================================================================
# REGRESSION MODEL ASSETS
# ============================================================================

training_config_schema_reg = {
    "n_estimators": Field(int, default_value=300, description="Number of boosting rounds (reduced to prevent overfitting)"),
    "max_depth": Field(int, default_value=4, description="Maximum tree depth (reduced to prevent overfitting)"),
    "learning_rate": Field(float, default_value=0.05, description="Learning rate (eta) (increased for faster convergence)"),
    "subsample": Field(float, default_value=0.7, description="Subsample ratio of training data (more randomness)"),
    "colsample_bytree": Field(float, default_value=0.7, description="Subsample ratio of columns per tree (more randomness)"),
    "min_child_weight": Field(int, default_value=5, description="Minimum sum of instance weight in child (increased regularization)"),
    "gamma": Field(float, default_value=0.5, description="Minimum loss reduction for split (increased regularization)"),
    "reg_alpha": Field(float, default_value=0.5, description="L1 regularization (increased)"),
    "reg_lambda": Field(float, default_value=2.0, description="L2 regularization (increased)"),
    "early_stopping_rounds": Field(int, default_value=30, description="Early stopping patience (stop sooner)"),
}


@asset(
    name="xgb_trained_model_reg",
    group_name="XGBoost",
    kinds={"python", "regression"},
    config_schema=training_config_schema_reg,
)
def xgb_trained_model_reg(context, asset_preprocessed_data):
    """
    Train XGBoost regression model on preprocessed data.
    Predicts next-day return using gradient boosting.
    """
    # Extract config
    n_estimators = context.op_config.get("n_estimators", 1000)
    max_depth = context.op_config.get("max_depth", 6)
    learning_rate = context.op_config.get("learning_rate", 0.01)
    subsample = context.op_config.get("subsample", 0.8)
    colsample_bytree = context.op_config.get("colsample_bytree", 0.8)
    min_child_weight = context.op_config.get("min_child_weight", 3)
    gamma = context.op_config.get("gamma", 0.1)
    reg_alpha = context.op_config.get("reg_alpha", 0.1)
    reg_lambda = context.op_config.get("reg_lambda", 1.0)
    early_stopping_rounds = context.op_config.get("early_stopping_rounds", 50)

    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    features = asset_preprocessed_data["features"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    y_train_reg = asset_preprocessed_data["y_train_reg"]
    y_val_reg = asset_preprocessed_data["y_val_reg"]
    y_test_reg = asset_preprocessed_data["y_test_reg"]

    context.log.info(f"[XGB REGRESSION] Training data shapes:")
    context.log.info(f"  X_train: {X_train.shape}, y_train: {y_train_reg.shape}")
    context.log.info(f"  X_val: {X_val.shape}, y_val: {y_val_reg.shape}")
    context.log.info(f"  X_test: {X_test.shape}, y_test: {y_test_reg.shape}")
    context.log.info(f"  Number of features: {len(features)}")

    # Build model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective='reg:squarederror',
        eval_metric=['rmse', 'mae'],  # Moved here for XGBoost 2.x
        early_stopping_rounds=early_stopping_rounds,  # Moved here for XGBoost 2.x
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # Faster training
    )

    context.log.info("[XGB REGRESSION] Model parameters:")
    context.log.info(f"  n_estimators: {n_estimators}, max_depth: {max_depth}")
    context.log.info(f"  learning_rate: {learning_rate}, subsample: {subsample}")
    context.log.info(f"  L1: {reg_alpha}, L2: {reg_lambda}")

    # Train model with early stopping
    context.log.info(f"[XGB REGRESSION] Training model with early stopping (patience={early_stopping_rounds})...")
    model.fit(
        X_train, y_train_reg,
        eval_set=[(X_train, y_train_reg), (X_val, y_val_reg)],
        verbose=False
    )

    best_iteration = model.best_iteration
    context.log.info(f"[XGB REGRESSION] Training complete. Best iteration: {best_iteration}")

    # Get training history
    evals_result = model.evals_result()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train_reg, y_train_pred)
    val_mae = mean_absolute_error(y_val_reg, y_val_pred)
    test_mae = mean_absolute_error(y_test_reg, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_reg, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
    
    test_r2 = r2_score(y_test_reg, y_test_pred)

    context.log.info(f"[XGB REGRESSION] Train MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
    context.log.info(f"[XGB REGRESSION] Val MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
    context.log.info(f"[XGB REGRESSION] Test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")

    # Generate visualizations
    context.log.info("[XGB REGRESSION] Generating training history plot...")
    training_plot_path = CHARTS_DIR / ticker / f"{ticker}_training_history_xgb_reg.png"
    plot_xgb_training_history(evals_result, ticker, "XGBoost Regression", training_plot_path)
    
    context.log.info("[XGB REGRESSION] Generating feature importance plot...")
    importance_plot_path = CHARTS_DIR / ticker / f"{ticker}_feature_importance_xgb_reg.png"
    _, feature_importance_df = plot_feature_importance(model, features, ticker, "XGBoost Regression", importance_plot_path)
    
    context.log.info("[XGB REGRESSION] Generating predictions comparison plot...")
    predictions_plot_path = CHARTS_DIR / ticker / f"{ticker}_predictions_comparison_xgb_reg.png"
    plot_predictions_comparison(y_test_reg, y_test_pred, ticker, "XGBoost Regression", predictions_plot_path)
    
    context.log.info("[XGB REGRESSION] Generating scatter plot...")
    scatter_plot_path = CHARTS_DIR / ticker / f"{ticker}_scatter_xgb_reg.png"
    plot_scatter_comparison(y_test_reg, y_test_pred, ticker, "XGBoost Regression", scatter_plot_path)

    # Save model using pickle (consistent with classification, preserves all attributes)
    import pickle
    model_path = MODEL_DIR / ticker / f"{ticker}_xgb_model_reg.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    context.log.info(f"[XGB REGRESSION] Model saved to {model_path}")

    # Save training history (XGBoost 2.x uses 'validation_0', 'validation_1' keys)
    eval_set_keys = list(evals_result.keys())
    train_key = eval_set_keys[0]  # 'validation_0' = train set
    val_key = eval_set_keys[1] if len(eval_set_keys) > 1 else train_key  # 'validation_1' = validation set
    
    history_df = pd.DataFrame({
        'iteration': range(len(evals_result[train_key]['rmse'])),
        'train_rmse': evals_result[train_key]['rmse'],
        'val_rmse': evals_result[val_key]['rmse'],
        'train_mae': evals_result[train_key]['mae'],
        'val_mae': evals_result[val_key]['mae'],
    })
    save_data(history_df, f"{ticker}_training_history_xgb_reg.csv", 
              f"models/xgb/{ticker}", context, "xgb_trained_model_reg")

    # Save feature importance
    save_data(feature_importance_df, f"{ticker}_feature_importance_xgb_reg.csv",
              f"models/xgb/{ticker}", context, "xgb_trained_model_reg")

    # Save test predictions
    test_results_df = pd.DataFrame({
        'y_true': y_test_reg,
        'y_pred': y_test_pred,
        'error': y_test_reg - y_test_pred,
        'abs_error': np.abs(y_test_reg - y_test_pred),
    })
    save_data(test_results_df, f"{ticker}_test_predictions_xgb_reg.csv",
              f"models/xgb/{ticker}", context, "xgb_trained_model_reg")

    output_value = {
        "ticker": ticker,
        "model_path": str(model_path),
        "n_features": len(features),
        "best_iteration": int(best_iteration),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }

    metadata = {
        "ticker": ticker,
        "model_path": str(model_path),
        "model_type": "XGBoost Regression",
        "n_features": len(features),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "best_iteration": int(best_iteration),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "training_history_plot": MetadataValue.path(str(training_plot_path)),
        "feature_importance_plot": MetadataValue.path(str(importance_plot_path)),
        "predictions_comparison_plot": MetadataValue.path(str(predictions_plot_path)),
        "scatter_plot": MetadataValue.path(str(scatter_plot_path)),
    }

    return Output(value=output_value, metadata=metadata)


@asset(
    name="xgb_predictions_reg",
    group_name="XGBoost",
    kinds={"python", "regression"},
)
def xgb_predictions_reg(context, asset_preprocessed_data, xgb_trained_model_reg):
    """Use trained XGBoost regression model to predict next-day return."""
    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_predict = asset_preprocessed_data["X_predict"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]

    # Load model
    model_path = xgb_trained_model_reg["model_path"]
    context.log.info(f"[XGB REGRESSION] Loading model from {model_path}")
    
    # Use pickle to load (consistent with classification)
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict
    next_return_pred = float(model.predict(X_predict)[0])
    context.log.info(f"[XGB REGRESSION] Predicted next-day return: {next_return_pred:.6f}")

    # Convert to price
    next_price_pred = float(last_close * (1.0 + next_return_pred))
    context.log.info(f"[XGB REGRESSION] Last close: {last_close:.2f}")
    context.log.info(f"[XGB REGRESSION] Predicted next close: {next_price_pred:.2f}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_return': [next_return_pred],
        'predicted_price': [next_price_pred],
        'return_percent': [next_return_pred * 100],
    })
    save_data(predictions_df, f"{ticker}_latest_predictions_xgb_reg.csv",
              f"models/xgb/{ticker}", context, "xgb_predictions_reg")

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
    }

    return Output(value=output_value, metadata=metadata)


# ============================================================================
# CLASSIFICATION MODEL ASSETS
# ============================================================================

training_config_schema_cls = {
    "n_estimators": Field(int, default_value=200, description="Number of boosting rounds (reduced to prevent overfitting)"),
    "max_depth": Field(int, default_value=3, description="Maximum tree depth (reduced to prevent overfitting)"),
    "learning_rate": Field(float, default_value=0.05, description="Learning rate (eta) (increased for faster convergence)"),
    "subsample": Field(float, default_value=0.7, description="Subsample ratio of training data (more randomness)"),
    "colsample_bytree": Field(float, default_value=0.7, description="Subsample ratio of columns per tree (more randomness)"),
    "min_child_weight": Field(int, default_value=7, description="Minimum sum of instance weight in child (increased regularization)"),
    "gamma": Field(float, default_value=1.0, description="Minimum loss reduction for split (increased regularization)"),
    "reg_alpha": Field(float, default_value=1.0, description="L1 regularization (increased)"),
    "reg_lambda": Field(float, default_value=3.0, description="L2 regularization (increased)"),
    "early_stopping_rounds": Field(int, default_value=20, description="Early stopping patience (stop sooner)"),
    "scale_pos_weight": Field(float, default_value=1.0, description="Balance of positive vs negative weights"),
}


@asset(
    name="xgb_trained_model_cls",
    group_name="XGBoost",
    kinds={"python", "classification"},
    config_schema=training_config_schema_cls,
)
def xgb_trained_model_cls(context, asset_preprocessed_data):
    """
    Train XGBoost classification model on preprocessed data.
    Predicts next-day direction (up/down) using gradient boosting.
    """
    # Extract config
    n_estimators = context.op_config.get("n_estimators", 1000)
    max_depth = context.op_config.get("max_depth", 5)
    learning_rate = context.op_config.get("learning_rate", 0.01)
    subsample = context.op_config.get("subsample", 0.8)
    colsample_bytree = context.op_config.get("colsample_bytree", 0.8)
    min_child_weight = context.op_config.get("min_child_weight", 3)
    gamma = context.op_config.get("gamma", 0.1)
    reg_alpha = context.op_config.get("reg_alpha", 0.1)
    reg_lambda = context.op_config.get("reg_lambda", 1.0)
    early_stopping_rounds = context.op_config.get("early_stopping_rounds", 50)
    scale_pos_weight = context.op_config.get("scale_pos_weight", 1.0)

    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    features = asset_preprocessed_data["features"]
    X_train = asset_preprocessed_data["X_train"]
    X_val = asset_preprocessed_data["X_val"]
    X_test = asset_preprocessed_data["X_test"]
    y_train_cls = asset_preprocessed_data["y_train_cls"].astype(int)
    y_val_cls = asset_preprocessed_data["y_val_cls"].astype(int)
    y_test_cls = asset_preprocessed_data["y_test_cls"].astype(int)

    num_classes = len(np.unique(y_train_cls))
    context.log.info(f"[XGB CLASSIFICATION] Number of classes: {num_classes}")
    
    # Class distribution
    unique, counts = np.unique(y_train_cls, return_counts=True)
    context.log.info(f"[XGB CLASSIFICATION] Class distribution: {dict(zip(unique, counts))}")

    # Auto-calculate scale_pos_weight if binary and not set
    if num_classes == 2 and scale_pos_weight == 1.0:
        neg_count = counts[0] if unique[0] == 0 else counts[1]
        pos_count = counts[1] if unique[1] == 1 else counts[0]
        scale_pos_weight = neg_count / pos_count
        context.log.info(f"[XGB CLASSIFICATION] Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Build model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight if num_classes == 2 else None,
        objective='binary:logistic' if num_classes == 2 else 'multi:softprob',
        eval_metric=['logloss', 'error'],  # Moved here for XGBoost 2.x
        early_stopping_rounds=early_stopping_rounds,  # Moved here for XGBoost 2.x
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )

    context.log.info("[XGB CLASSIFICATION] Training model...")
    model.fit(
        X_train, y_train_cls,
        eval_set=[(X_train, y_train_cls), (X_val, y_val_cls)],
        verbose=False
    )

    best_iteration = model.best_iteration
    context.log.info(f"[XGB CLASSIFICATION] Best iteration: {best_iteration}")

    # Get training history
    evals_result = model.evals_result()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_test_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    train_acc = accuracy_score(y_train_cls, y_train_pred)
    val_acc = accuracy_score(y_val_cls, y_val_pred)
    test_acc = accuracy_score(y_test_cls, y_test_pred)
    
    test_precision = precision_score(y_test_cls, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_cls, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_cls, y_test_pred, average='weighted', zero_division=0)

    context.log.info(f"[XGB CLASSIFICATION] Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    context.log.info(f"[XGB CLASSIFICATION] Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    context.log.info(f"[XGB CLASSIFICATION] Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    context.log.info(f"[XGB CLASSIFICATION] Test Precision: {test_precision:.4f}")
    context.log.info(f"[XGB CLASSIFICATION] Test Recall: {test_recall:.4f}")
    context.log.info(f"[XGB CLASSIFICATION] Test F1: {test_f1:.4f}")

    # Classification report
    labels = ['Down', 'Up'] if num_classes == 2 else ['Down', 'Flat', 'Up']
    report = classification_report(y_test_cls, y_test_pred, target_names=labels, zero_division=0)
    context.log.info(f"[XGB CLASSIFICATION] Classification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_cls, y_test_pred)
    context.log.info(f"[XGB CLASSIFICATION] Confusion Matrix:\n{cm}")

    # ROC-AUC (binary only)
    roc_auc = None
    if num_classes == 2:
        roc_auc = roc_auc_score(y_test_cls, y_test_pred_proba[:, 1])
        context.log.info(f"[XGB CLASSIFICATION] ROC-AUC: {roc_auc:.4f}")

    # Generate visualizations
    context.log.info("[XGB CLASSIFICATION] Generating plots...")
    training_plot_path = CHARTS_DIR / ticker / f"{ticker}_training_history_xgb_cls.png"
    plot_xgb_training_history(evals_result, ticker, "XGBoost Classification", training_plot_path)
    
    importance_plot_path = CHARTS_DIR / ticker / f"{ticker}_feature_importance_xgb_cls.png"
    _, feature_importance_df = plot_feature_importance(model, features, ticker, "XGBoost Classification", importance_plot_path)
    
    predictions_plot_path = CHARTS_DIR / ticker / f"{ticker}_predictions_comparison_xgb_cls.png"
    plot_predictions_comparison(y_test_cls, y_test_pred, ticker, "XGBoost Classification", predictions_plot_path)
    
    cm_plot_path = CHARTS_DIR / ticker / f"{ticker}_confusion_matrix_xgb_cls.png"
    plot_confusion_matrix_chart(cm, ticker, cm_plot_path, labels)
    
    # ROC curve (binary only)
    roc_plot_path = None
    if num_classes == 2:
        roc_plot_path = CHARTS_DIR / ticker / f"{ticker}_roc_curve_xgb_cls.png"
        plot_roc_curve_chart(y_test_cls, y_test_pred_proba[:, 1], ticker, roc_plot_path)

    # Save model using pickle (preserves sklearn attributes like classes_)
    # XGBoost 2.x: save_model() doesn't preserve sklearn-specific attributes
    import pickle
    model_path = MODEL_DIR / ticker / f"{ticker}_xgb_model_cls.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    context.log.info(f"[XGB CLASSIFICATION] Model saved to {model_path}")

    # Save training history (XGBoost 2.x uses 'validation_0', 'validation_1' keys)
    eval_set_keys = list(evals_result.keys())
    train_key = eval_set_keys[0]  # 'validation_0' = train set
    val_key = eval_set_keys[1] if len(eval_set_keys) > 1 else train_key  # 'validation_1' = validation set
    
    history_df = pd.DataFrame({
        'iteration': range(len(evals_result[train_key]['logloss'])),
        'train_logloss': evals_result[train_key]['logloss'],
        'val_logloss': evals_result[val_key]['logloss'],
        'train_error': evals_result[train_key]['error'],
        'val_error': evals_result[val_key]['error'],
    })
    save_data(history_df, f"{ticker}_training_history_xgb_cls.csv",
              f"models/xgb/{ticker}", context, "xgb_trained_model_cls")

    # Save feature importance
    save_data(feature_importance_df, f"{ticker}_feature_importance_xgb_cls.csv",
              f"models/xgb/{ticker}", context, "xgb_trained_model_cls")

    # Save test predictions
    if num_classes == 2:
        test_results_df = pd.DataFrame({
            'y_true': y_test_cls,
            'y_pred': y_test_pred,
            'prob_down': y_test_pred_proba[:, 0],
            'prob_up': y_test_pred_proba[:, 1],
        })
    else:
        test_results_df = pd.DataFrame({
            'y_true': y_test_cls,
            'y_pred': y_test_pred,
            'prob_down': y_test_pred_proba[:, 0],
            'prob_flat': y_test_pred_proba[:, 1],
            'prob_up': y_test_pred_proba[:, 2],
        })
    save_data(test_results_df, f"{ticker}_test_predictions_xgb_cls.csv",
              f"models/xgb/{ticker}", context, "xgb_trained_model_cls")

    output_value = {
        "ticker": ticker,
        "model_path": str(model_path),
        "n_features": len(features),
        "num_classes": num_classes,
        "best_iteration": int(best_iteration),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_roc_auc": float(roc_auc) if roc_auc else None,
    }

    metadata = {
        "ticker": ticker,
        "model_path": str(model_path),
        "model_type": "XGBoost Classification",
        "n_features": len(features),
        "num_classes": num_classes,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "best_iteration": int(best_iteration),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_roc_auc": float(roc_auc) if roc_auc else None,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "training_history_plot": MetadataValue.path(str(training_plot_path)),
        "feature_importance_plot": MetadataValue.path(str(importance_plot_path)),
        "predictions_comparison_plot": MetadataValue.path(str(predictions_plot_path)),
        "confusion_matrix_plot": MetadataValue.path(str(cm_plot_path)),
    }
    
    if roc_plot_path:
        metadata["roc_curve_plot"] = MetadataValue.path(str(roc_plot_path))

    return Output(value=output_value, metadata=metadata)


@asset(
    name="xgb_predictions_cls",
    group_name="XGBoost",
    kinds={"python", "classification"},
)
def xgb_predictions_cls(context, asset_preprocessed_data, xgb_trained_model_cls):
    """Use trained XGBoost classification model to predict next-day direction."""
    # Extract data
    ticker = asset_preprocessed_data["ticker"]
    X_predict = asset_preprocessed_data["X_predict"]
    last_close = asset_preprocessed_data["last_close"]
    last_date = asset_preprocessed_data["last_date"]

    # Load model
    model_path = xgb_trained_model_cls["model_path"]
    num_classes = xgb_trained_model_cls["num_classes"]
    context.log.info(f"[XGB CLASSIFICATION] Loading model from {model_path}")
    
    # XGBoost 2.x: Use pickle to load (preserves sklearn attributes)
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    pred_class = int(model.predict(X_predict)[0])
    pred_proba = model.predict_proba(X_predict)[0]

    # Extract probabilities
    if num_classes == 2:
        prob_down = float(pred_proba[0])
        prob_up = float(pred_proba[1])
        prob_flat = 0.0
        direction_map = {0: 'Down', 1: 'Up'}
        pred_direction = direction_map[pred_class]
        confidence = max(prob_down, prob_up)
    else:
        prob_down = float(pred_proba[0])
        prob_flat = float(pred_proba[1])
        prob_up = float(pred_proba[2])
        direction_map = {0: 'Down', 1: 'Flat', 2: 'Up'}
        pred_direction = direction_map[pred_class]
        confidence = float(np.max(pred_proba))

    context.log.info(f"[XGB CLASSIFICATION] Predicted direction: {pred_direction}")
    context.log.info(f"[XGB CLASSIFICATION] Confidence: {confidence:.2%}")
    context.log.info(f"[XGB CLASSIFICATION] Probabilities - Down: {prob_down:.2%}, Up: {prob_up:.2%}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'last_date': [last_date],
        'last_close': [last_close],
        'predicted_direction': [pred_direction],
        'predicted_class': [pred_class],
        'confidence': [confidence],
        'prob_down': [prob_down],
        'prob_flat': [prob_flat],
        'prob_up': [prob_up],
    })
    save_data(predictions_df, f"{ticker}_latest_predictions_xgb_cls.csv",
              f"models/xgb/{ticker}", context, "xgb_predictions_cls")

    output_value = {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "predicted_direction": pred_direction,
        "predicted_class": int(pred_class),
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
        "predicted_class": int(pred_class),
        "confidence": float(confidence),
        "prob_down": float(prob_down),
        "prob_flat": float(prob_flat),
        "prob_up": float(prob_up),
    }

    return Output(value=output_value, metadata=metadata)

