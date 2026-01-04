"""
Model selection script for phishing detection.

This script:
1. Loads the urlscan_features dataset from Hopsworks
2. Performs z-score normalization on continuous variables
3. Trains and evaluates multiple classification models
4. Compares model performance with metrics and visualizations
"""

import sys
import os
import logging
import argparse
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Add src folder to path
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(src_folder)

from phising_detection.utils import hopsworks_utils as hw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(
    project,
    feature_group_name: str = "urlscan_features",
    version: int = 1
) -> pd.DataFrame:
    """
    Load features from Hopsworks feature group.

    Args:
        project: Hopsworks project object
        feature_group_name: Name of feature group
        version: Feature group version

    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Loading feature group: {feature_group_name} v{version}")
    df = hw.read_feature_group(project, feature_group_name, version)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """
    Prepare data for modeling with train/val/test splits and normalization.

    Args:
        df: Input DataFrame
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger.info("Preparing data for modeling...")

    # Define feature columns (excluding metadata and target)
    feature_columns = [
        'domain_age_days',
        'secure_percentage',
        'has_umbrella_rank',
        'umbrella_rank',
        'has_tls',
        'tls_valid_days',
        'url_length',
        'subdomain_count'
    ]

    # Continuous features that need z-score normalization
    continuous_features = [
        'domain_age_days',
        'secure_percentage',
        'umbrella_rank',
        'tls_valid_days',
        'url_length',
        'subdomain_count'
    ]

    # Binary/categorical features (no normalization needed)
    categorical_features = [
        'has_umbrella_rank',
        'has_tls'
    ]

    # Extract features and target
    X = df[feature_columns].copy()
    y = df['is_phishing'].copy()

    # Log initial class distribution
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    class_balance = y.value_counts(normalize=True).to_dict()
    logger.info(f"Class balance: {class_balance}")

    # Handle missing values
    logger.info("Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())  # Fill with median for continuous features
    missing_after = X.isnull().sum().sum()
    logger.info(f"Filled {missing_before - missing_after} missing values")

    # First split: separate out test set
    train_size = 1.0 - test_size
    logger.info(f"Splitting data: {train_size:.0%} train+val, {test_size:.0%} test")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate train and validation
    val_size_adjusted = val_size / train_size  # Adjust val_size relative to temp set
    logger.info(f"Splitting train+val: {1-val_size_adjusted:.0%} train, {val_size_adjusted:.0%} val")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    logger.info(f"\nFinal split sizes:")
    logger.info(f"  Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Val set:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    logger.info(f"\nClass distribution:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()} - Balance: {y_train.value_counts(normalize=True).to_dict()}")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()} - Balance: {y_val.value_counts(normalize=True).to_dict()}")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()} - Balance: {y_test.value_counts(normalize=True).to_dict()}")

    # Z-score normalization on continuous features
    logger.info("\nApplying z-score normalization to continuous features...")
    scaler = StandardScaler()

    # Fit on training data only
    X_train_continuous = X_train[continuous_features]
    X_val_continuous = X_val[continuous_features]
    X_test_continuous = X_test[continuous_features]

    X_train_continuous_scaled = scaler.fit_transform(X_train_continuous)
    X_val_continuous_scaled = scaler.transform(X_val_continuous)
    X_test_continuous_scaled = scaler.transform(X_test_continuous)

    # Create normalized DataFrames
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_features] = X_train_continuous_scaled
    X_val_scaled[continuous_features] = X_val_continuous_scaled
    X_test_scaled[continuous_features] = X_test_continuous_scaled

    logger.info("Data preparation complete!")
    logger.info(f"Normalized features: {continuous_features}")
    logger.info(f"Non-normalized features: {categorical_features}")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Train multiple classification models.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary mapping model names to trained model objects
    """
    logger.info("=" * 80)
    logger.info("Training models...")
    logger.info("=" * 80)

    models = {
        # Baseline models (for comparison)
        'Baseline (Most Frequent)': DummyClassifier(
            strategy='most_frequent',
            random_state=42
        ),
        'Baseline (Stratified)': DummyClassifier(
            strategy='stratified',
            random_state=42
        ),
        # Machine learning models
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32),  # 2 hidden layers: 64 and 32 neurons
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} trained successfully!")

    return trained_models


def evaluate_models_on_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all models on validation set for model selection.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels

    Returns:
        DataFrame with model comparison metrics
    """
    logger.info("=" * 80)
    logger.info("Evaluating models on VALIDATION set...")
    logger.info("=" * 80)

    results = []

    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Probability predictions for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = y_val_pred  # Fallback for models without probability

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)

        # Cross-validation score (on training data)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            'Model': name,
            'Train Accuracy': train_accuracy,
            'Val Accuracy': val_accuracy,
            'Precision': val_precision,
            'Recall': val_recall,
            'F1 Score': val_f1,
            'ROC-AUC': val_roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std,
            'Overfit (Train-Val)': train_accuracy - val_accuracy
        })

        # Print detailed report
        logger.info(f"\n{name} Validation Results:")
        logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"  Val Accuracy:   {val_accuracy:.4f}")
        logger.info(f"  Precision:      {val_precision:.4f}")
        logger.info(f"  Recall:         {val_recall:.4f}")
        logger.info(f"  F1 Score:       {val_f1:.4f}")
        logger.info(f"  ROC-AUC:        {val_roc_auc:.4f}")
        logger.info(f"  CV Mean ± Std:  {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"  Overfitting:    {train_accuracy - val_accuracy:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val Accuracy', ascending=False)

    return results_df


def evaluate_best_model_on_test(
    best_model,
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    """
    Perform final evaluation of best model on held-out test set.

    Args:
        best_model: The best performing model from validation
        best_model_name: Name of the best model
        X_test: Test features
        y_test: Test labels
    """
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 80)
    logger.info(f"Evaluating best model: {best_model_name}")

    # Predictions
    y_test_pred = best_model.predict(X_test)

    # Probability predictions for ROC-AUC
    if hasattr(best_model, 'predict_proba'):
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = y_test_pred

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Print results
    logger.info(f"\n{best_model_name} - TEST SET PERFORMANCE:")
    logger.info(f"  Accuracy:  {test_accuracy:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  F1 Score:  {test_f1:.4f}")
    logger.info(f"  ROC-AUC:   {test_roc_auc:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_test_pred,
                                             target_names=['Legitimate', 'Phishing']))

    return {
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'ROC-AUC': test_roc_auc
    }


def plot_model_comparison(results_df: pd.DataFrame, output_path: str = None):
    """
    Create visualization comparing model performance on validation set.

    Args:
        results_df: DataFrame with model metrics
        output_path: Optional path to save the plot
    """
    logger.info("Creating model comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Val Accuracy comparison
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
    ax.set_title('Model Comparison: Precision, Recall, F1 (Validation)')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation=45)

    # 3. ROC-AUC comparison
    ax = axes[1, 0]
    results_df_sorted = results_df.sort_values('ROC-AUC')
    ax.barh(results_df_sorted['Model'], results_df_sorted['ROC-AUC'])
    ax.set_xlabel('ROC-AUC Score')
    ax.set_title('Model Comparison: ROC-AUC (Validation)')
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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Saved plot to model_comparison.png")

    plt.close()


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = None
):
    """
    Create confusion matrices for all models.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        output_path: Optional path to save the plot
    """
    logger.info("Creating confusion matrices...")

    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {output_path}")
    else:
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        logger.info("Saved confusion matrices to confusion_matrices.png")

    plt.close()


def main(
    feature_group_name: str = "urlscan_features",
    feature_group_version: int = 1,
    val_size: float = 0.15,
    test_size: float = 0.15,
    output_dir: str = "."
):
    """
    Main function to run model selection pipeline.

    Args:
        feature_group_name: Name of Hopsworks feature group
        feature_group_version: Version of feature group
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        output_dir: Directory to save outputs
    """
    logger.info("=" * 80)
    logger.info("Starting Model Selection Pipeline")
    logger.info("=" * 80)

    # Connect to Hopsworks
    logger.info("Connecting to Hopsworks...")
    project = hw.connect_to_hopsworks()

    # Load data
    df = load_data(project, feature_group_name, feature_group_version)

    # Prepare data with train/val/test split and normalization
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        df, val_size=val_size, test_size=test_size
    )

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models on VALIDATION set for model selection
    results_df = evaluate_models_on_validation(models, X_train, X_val, y_train, y_val)

    # Display validation results
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON ON VALIDATION SET")
    logger.info("=" * 80)
    logger.info("\n" + results_df.to_string(index=False))

    # Save validation results
    results_path = os.path.join(output_dir, 'model_comparison_validation.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nSaved validation results to {results_path}")

    # Create visualizations based on validation performance
    plot_path = os.path.join(output_dir, 'model_comparison_validation.png')
    plot_model_comparison(results_df, plot_path)

    cm_plot_path = os.path.join(output_dir, 'confusion_matrices_validation.png')
    plot_confusion_matrices(models, X_val, y_val, cm_plot_path)

    # Select best model based on validation performance
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    logger.info("\n" + "=" * 80)
    logger.info(f"SELECTED BEST MODEL (based on validation): {best_model_name}")
    logger.info(f"Val Accuracy: {results_df.iloc[0]['Val Accuracy']:.4f}")
    logger.info(f"F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
    logger.info(f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
    logger.info("=" * 80)

    # Final evaluation on TEST set (only for best model)
    test_metrics = evaluate_best_model_on_test(best_model, best_model_name, X_test, y_test)

    # Save test results
    test_results_path = os.path.join(output_dir, 'best_model_test_results.csv')
    test_df = pd.DataFrame([{
        'Model': best_model_name,
        **test_metrics
    }])
    test_df.to_csv(test_results_path, index=False)
    logger.info(f"\nSaved test results to {test_results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Model Selection Pipeline Complete!")
    logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Model selection for phishing detection"
    )

    parser.add_argument(
        "--feature-group-name",
        type=str,
        default="urlscan_features",
        help="Name of Hopsworks feature group (default: urlscan_features)"
    )
    parser.add_argument(
        "--feature-group-version",
        type=int,
        default=1,
        help="Version of feature group (default: 1)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save outputs (default: current directory)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        feature_group_name=args.feature_group_name,
        feature_group_version=args.feature_group_version,
        test_size=args.test_size,
        output_dir=args.output_dir
    )