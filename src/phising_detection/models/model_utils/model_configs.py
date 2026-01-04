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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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


def get_extensive_param_grids() -> Dict[str, Dict[str, list]]:
    """
    Get extensive hyperparameter grids for exhaustive search.

    Returns:
        Dictionary mapping model names to extensive parameter grids
    """
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'penalty': ['l2'],
            'max_iter': [1000, 2000]
        }
    }
    return param_grids


def extensive_hyperparameter_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    n_iter: int = 50,
    use_random_search: bool = True
) -> Any:
    """
    Perform extensive hyperparameter search using RandomizedSearchCV.

    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        n_iter: Number of parameter settings sampled (for RandomizedSearchCV)
        use_random_search: Use RandomizedSearchCV (True) or GridSearchCV (False)

    Returns:
        Best model from extensive search with full training results
    """
    logger.info("=" * 80)
    logger.info(f"EXTENSIVE HYPERPARAMETER SEARCH for {model_name}")
    logger.info("=" * 80)

    # Get base model and extensive param grid
    models = get_model_configs()
    param_grids = get_extensive_param_grids()

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")

    if model_name not in param_grids:
        logger.warning(f"No extensive param grid for {model_name}")
        return tune_hyperparameters(model_name, X_train, y_train, cv_folds)

    base_model = models[model_name]
    param_grid = param_grids[model_name]

    # Choose search strategy
    if use_random_search:
        logger.info(f"Using RandomizedSearchCV with {n_iter} iterations")
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True
        )
    else:
        logger.info("Using GridSearchCV (exhaustive search)")
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

    # Fit search
    logger.info(f"Starting extensive search with {cv_folds}-fold CV...")
    search.fit(X_train, y_train)

    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("SEARCH RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    logger.info(f"Best estimator: {search.best_estimator_}")

    # Log top 5 results
    import pandas as pd
    results_df = pd.DataFrame(search.cv_results_)
    top_5 = results_df.nsmallest(5, 'rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    logger.info("\nTop 5 parameter combinations:")
    logger.info("\n" + top_5.to_string(index=False))

    return search.best_estimator_, search
