"""
Visualization utilities for model evaluation.

This module provides:
- Model comparison plots
- Confusion matrix visualizations
- Feature importance plots
"""

import logging
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_model_comparison(results_df: pd.DataFrame, output_path: str = 'model_comparison.png'):
    """
    Create visualization comparing model performance.

    Args:
        results_df: DataFrame with model metrics
        output_path: Path to save the plot
    """
    logger.info("Creating model comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Validation Accuracy comparison
    ax = axes[0, 0]
    results_df_sorted = results_df.sort_values('Val Accuracy')
    ax.barh(results_df_sorted['Model'], results_df_sorted['Val Accuracy'])
    ax.set_xlabel('Validation Accuracy')
    ax.set_title('Model Comparison: Validation Accuracy')
    ax.set_xlim(0, 1)
    for i, v in enumerate(results_df_sorted['Val Accuracy']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    # 2. Precision, Recall, F1 comparison
    ax = axes[0, 1]
    metrics_df = results_df[['Model', 'Precision', 'Recall', 'F1 Score']].set_index('Model')
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison: Precision, Recall, F1')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation=45)

    # 3. ROC-AUC comparison
    ax = axes[1, 0]
    results_df_sorted = results_df.sort_values('ROC-AUC')
    ax.barh(results_df_sorted['Model'], results_df_sorted['ROC-AUC'])
    ax.set_xlabel('ROC-AUC Score')
    ax.set_title('Model Comparison: ROC-AUC')
    ax.set_xlim(0, 1)
    for i, v in enumerate(results_df_sorted['ROC-AUC']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    # 4. Overfitting analysis
    ax = axes[1, 1]
    results_df_sorted = results_df.sort_values('Overfit (Train-Val)')
    colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green'
              for x in results_df_sorted['Overfit (Train-Val)']]
    ax.barh(results_df_sorted['Model'], results_df_sorted['Overfit (Train-Val)'], color=colors)
    ax.set_xlabel('Train - Val Accuracy')
    ax.set_title('Overfitting Analysis (Lower is Better)')
    ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='High overfit threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved model comparison plot to {output_path}")
    plt.close()


def plot_confusion_matrices(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str = 'confusion_matrices.png'
):
    """
    Create confusion matrices for all models.

    Args:
        models: Dictionary of trained models
        X: Features
        y: True labels
        output_path: Path to save the plot
    """
    logger.info("Creating confusion matrices...")

    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_xticklabels(['Legitimate', 'Phishing'])
        ax.set_yticklabels(['Legitimate', 'Phishing'])

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrices to {output_path}")
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    mean_importance_df: pd.DataFrame,
    output_path: str = 'feature_importance.png'
):
    """
    Create visualizations of feature importance.

    Args:
        importance_df: DataFrame with feature importances per model
        mean_importance_df: DataFrame with mean importances
        output_path: Path to save the plot
    """
    logger.info("Creating feature importance visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Mean feature importance across all models
    ax = axes[0]
    ax.barh(mean_importance_df['Feature'], mean_importance_df['Mean Importance'])
    ax.set_xlabel('Mean Importance Score')
    ax.set_title('Feature Importance (Averaged Across All Models)')
    ax.invert_yaxis()

    # Add error bars
    ax.errorbar(
        mean_importance_df['Mean Importance'],
        range(len(mean_importance_df)),
        xerr=mean_importance_df['Std Importance'],
        fmt='none',
        ecolor='gray',
        alpha=0.5
    )

    for i, v in enumerate(mean_importance_df['Mean Importance']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    # 2. Feature importance heatmap per model
    ax = axes[1]

    # Pivot to get features as rows and models as columns
    pivot_df = importance_df.pivot_table(
        index='Feature',
        columns='Model',
        values='Importance',
        aggfunc='first'
    )

    # Plot heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Importance Score'}
    )
    ax.set_title('Feature Importance by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to {output_path}")
    plt.close()
