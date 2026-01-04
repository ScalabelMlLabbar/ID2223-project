"""
Model configurations for phishing detection.

This module provides:
- Pre-configured model definitions
- Model training utilities
"""

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def get_model_configs() -> Dict[str, Any]:
    """
    Get pre-configured model definitions for phishing detection.

    Returns:
        Dictionary mapping model names to configured model objects
    """
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
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
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

    return models


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series, model_name: str = "Model"):
    """
    Train a single model.

    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training labels
        model_name: Name for logging

    Returns:
        Trained model
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} trained successfully!")
    return model


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
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

    models = get_model_configs()
    trained_models = {}

    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train, name)

    logger.info(f"\nTrained {len(trained_models)} models successfully!")

    return trained_models


def get_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Get hyperparameter grids for model tuning.

    Returns:
        Dictionary mapping model names to parameter grids
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'Neural Network (MLP)': {
            'hidden_layer_sizes': [(64, 32), (128, 64), (64, 32, 16)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'SVM (RBF)': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'SVM (Linear)': {
            'C': [0.1, 1.0, 10.0]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
    return param_grids


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> Any:
    """
    Tune hyperparameters for a model using GridSearchCV.

    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds

    Returns:
        Best model from grid search
    """
    logger.info(f"\nTuning hyperparameters for {model_name}...")

    # Get base model and param grid
    models = get_model_configs()
    param_grids = get_param_grids()

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")

    if model_name not in param_grids:
        logger.warning(f"No param grid for {model_name}, returning base model")
        return train_model(models[model_name], X_train, y_train, model_name)

    # Grid search
    grid_search = GridSearchCV(
        models[model_name],
        param_grids[model_name],
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
