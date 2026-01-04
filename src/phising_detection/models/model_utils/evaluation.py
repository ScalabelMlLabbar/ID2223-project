"""
Model evaluation utilities for phishing detection.

This module provides:
- Model evaluation metrics
- Feature importance analysis
- Model comparison utilities
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a model on any dataset (train/val/test).

    Simple function that works with any X and y - no need for separate
    functions for train/val/test.

    Args:
        model: Trained model
        X: Features (can be train, val, or test)
        y: True labels (can be train, val, or test)

    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X)

    # Probability predictions for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = y_pred

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_proba)
    }

    return metrics


def evaluate_model_detailed(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    dataset_name: str = "Dataset"
) -> Dict[str, Any]:
    """
    Detailed evaluation with metrics, confusion matrix, and classification report.
    Works with any dataset (train/val/test).

    Args:
        model: Trained model
        X: Features
        y: True labels
        model_name: Name of model for logging
        dataset_name: Name of dataset for logging (e.g., "Test", "Validation")

    Returns:
        Dictionary with metrics and confusion matrix
    """
    # Get basic metrics
    metrics = evaluate_model(model, X, y)

    # Confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    # Log detailed results
    logger.info("\n" + "=" * 80)
    logger.info(f"{model_name} - {dataset_name.upper()} SET PERFORMANCE")
    logger.info("=" * 80)
    logger.info(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")
    logger.info(f"  Recall:    {metrics['Recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['F1 Score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y, y_pred,
                                             target_names=['Legitimate', 'Phishing']))

    return {
        **metrics,
        'confusion_matrix': cm
    }


def compare_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models with train/val metrics and cross-validation.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        cv_folds: Number of cross-validation folds

    Returns:
        DataFrame with model comparison metrics sorted by validation accuracy
    """
    logger.info("=" * 80)
    logger.info("Comparing Models...")
    logger.info("=" * 80)

    results = []

    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")

        # Use the same evaluate_model() for both train and val
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

        results.append({
            'Model': name,
            'Train Accuracy': train_metrics['Accuracy'],
            'Val Accuracy': val_metrics['Accuracy'],
            'Precision': val_metrics['Precision'],
            'Recall': val_metrics['Recall'],
            'F1 Score': val_metrics['F1 Score'],
            'ROC-AUC': val_metrics['ROC-AUC'],
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Overfit (Train-Val)': train_metrics['Accuracy'] - val_metrics['Accuracy']
        })

        # Log results
        logger.info(f"  Train Accuracy: {train_metrics['Accuracy']:.4f}")
        logger.info(f"  Val Accuracy:   {val_metrics['Accuracy']:.4f}")
        logger.info(f"  Precision:      {val_metrics['Precision']:.4f}")
        logger.info(f"  Recall:         {val_metrics['Recall']:.4f}")
        logger.info(f"  F1 Score:       {val_metrics['F1 Score']:.4f}")
        logger.info(f"  ROC-AUC:        {val_metrics['ROC-AUC']:.4f}")
        logger.info(f"  CV Mean ± Std:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val Accuracy', ascending=False)

    return results_df


def analyze_feature_importance(
    models: Dict[str, Any],
    feature_names: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and analyze feature importance from models.

    Args:
        models: Dictionary of trained models
        feature_names: List of feature names

    Returns:
        Tuple of (importance_df, mean_importance_df)
    """
    logger.info("=" * 80)
    logger.info("Analyzing Feature Importance...")
    logger.info("=" * 80)

    importance_data = []

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            logger.info(f"\n{name} - Feature Importances (tree-based):")

            for feature, importance in zip(feature_names, importances):
                importance_data.append({
                    'Model': name,
                    'Feature': feature,
                    'Importance': importance,
                    'Type': 'Tree-based'
                })
                logger.info(f"  {feature}: {importance:.4f}")

        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            importances = np.abs(coef)
            logger.info(f"\n{name} - Feature Importances (absolute coefficients):")

            for feature, importance in zip(feature_names, importances):
                importance_data.append({
                    'Model': name,
                    'Feature': feature,
                    'Importance': importance,
                    'Type': 'Coefficient-based'
                })
                logger.info(f"  {feature}: {importance:.4f}")
        else:
            logger.info(f"\n{name} - No feature importance available")

    if not importance_data:
        logger.warning("No models with feature importance found!")
        return pd.DataFrame(), pd.DataFrame()

    # Create DataFrame
    importance_df = pd.DataFrame(importance_data)

    # Calculate mean importance across all models
    mean_importance = importance_df.groupby('Feature')['Importance'].agg(['mean', 'std']).reset_index()
    mean_importance.columns = ['Feature', 'Mean Importance', 'Std Importance']
    mean_importance = mean_importance.sort_values('Mean Importance', ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("MEAN FEATURE IMPORTANCE ACROSS ALL MODELS")
    logger.info("=" * 80)
    logger.info("\n" + mean_importance.to_string(index=False))

    return importance_df, mean_importance


def select_best_model(results_df: pd.DataFrame, models: Dict[str, Any], metric: str = 'Val Accuracy') -> tuple[str, Any]:
    """
    Select the best model based on a metric.

    Args:
        results_df: DataFrame with model evaluation results
        models: Dictionary of trained models
        metric: Metric to use for selection (default: 'Val Accuracy')

    Returns:
        Tuple of (best_model_name, best_model)
    """
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    logger.info("\n" + "=" * 80)
    logger.info(f"SELECTED BEST MODEL (based on {metric}): {best_model_name}")
    logger.info(f"  Val Accuracy: {results_df.iloc[0]['Val Accuracy']:.4f}")
    logger.info(f"  F1 Score:     {results_df.iloc[0]['F1 Score']:.4f}")
    logger.info(f"  ROC-AUC:      {results_df.iloc[0]['ROC-AUC']:.4f}")
    logger.info("=" * 80)

    return best_model_name, best_model
