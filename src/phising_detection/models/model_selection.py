"""
Model selection function for phishing detection.

Simple interface to select the best model using the modular components.
"""

import logging
from typing import Tuple, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phising_detection.models import data_prep, model_configs, evaluation
from phising_detection.utils.hopsworks_utils import connect_to_hopsworks

logger = logging.getLogger(__name__)


def select_model(
    feature_group_name: str = "urlscan_features",
    feature_group_version: int = 1,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[Any, str, dict, StandardScaler]:
    """
    Select the best model for phishing detection.

    This function:
    1. Loads data from Hopsworks
    2. Prepares and splits data
    3. Trains multiple models
    4. Compares them on validation set
    5. Returns the best performing model

    Args:
        feature_group_name: Name of Hopsworks feature group
        feature_group_version: Version of feature group
        val_size: Fraction for validation
        test_size: Fraction for testing

    Returns:
        Tuple of (best_model, model_name, test_metrics, scaler)
    """
    logger.info("=" * 80)
    logger.info("Starting Model Selection")
    logger.info("=" * 80)

    # 1. Load data
    project = connect_to_hopsworks()
    raw_data = data_prep.load_data(project, feature_group_name, feature_group_version)

    # 2. Prepare data (keep unnormalized data for later re-normalization)
    logger.info("\n2. Preparing data...")

    # Split WITHOUT normalization first
    X, y = data_prep.prepare_features(raw_data)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = data_prep.split_data(
        X, y, val_size=val_size, test_size=test_size
    )

    # Normalize the splits
    X_train, X_val, X_test, scaler = data_prep.normalize_features(
        X_train_raw, X_val_raw, X_test_raw
    )

    # 3. Train models
    logger.info("\n3. Training models...")
    models = model_configs.train_models(X_train, y_train)

    # 4. Compare models
    logger.info("\n4. Comparing models...")
    results_df = evaluation.compare_models(models, X_train, X_val, y_train, y_val)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("\n" + results_df.to_string(index=False))

    # 5. Tune top 3 models with hyperparameter search
    logger.info("\n5. Tuning hyperparameters for top 3 models...")
    top_3_names = results_df.head(3)['Model'].tolist()
    logger.info(f"Top 3 models: {top_3_names}")

    tuned_models = {}
    for model_name in top_3_names:
        tuned_models[model_name] = model_configs.tune_hyperparameters(
            model_name, X_train, y_train, cv_folds=5
        )

    # Compare tuned models on validation set
    logger.info("\n6. Comparing tuned models...")
    tuned_results = evaluation.compare_models(tuned_models, X_train, X_val, y_train, y_val)
    logger.info("\n" + tuned_results.to_string(index=False))

    # Select best tuned model
    best_model_name, best_model = evaluation.select_best_model(tuned_results, tuned_models)

    # 7. Combine train+val and retrain with re-normalization
    logger.info(f"\n7. Retraining {best_model_name} on combined train+val set...")

    # Combine RAW (unnormalized) data
    X_train_val_raw = pd.concat([X_train_raw, X_val_raw])
    y_train_val = pd.concat([y_train, y_val])

    # Re-normalize with combined statistics for better estimates
    logger.info("Re-normalizing with combined train+val statistics...")
    final_scaler = StandardScaler()
    X_train_val = X_train_val_raw.copy()
    X_test_final = X_test_raw.copy()

    # Normalize
    continuous_features = data_prep.CONTINUOUS_FEATURES
    X_train_val[continuous_features] = final_scaler.fit_transform(X_train_val_raw[continuous_features])
    X_test_final[continuous_features] = final_scaler.transform(X_test_raw[continuous_features])

    # Update scaler and test set
    scaler = final_scaler
    X_test = X_test_final

    # Train on properly re-normalized combined data
    best_model = model_configs.train_model(best_model, X_train_val, y_train_val, best_model_name)

    # 8. Evaluate on test set
    logger.info("\n8. Evaluating on test set...")
    test_metrics = evaluation.evaluate_model_detailed(
        best_model, X_test, y_test, best_model_name, "Test"
    )

    logger.info("\n" + "=" * 80)
    logger.info("Model Selection Complete!")
    logger.info(f"Selected: {best_model_name}")
    logger.info(f"Test Accuracy: {test_metrics['Accuracy']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['F1 Score']:.4f}")
    logger.info("=" * 80)

    return best_model, best_model_name, test_metrics, scaler


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model, name, metrics, scaler = select_model()

    print(f"\nBest Model: {name}")
    print(f"Metrics: {metrics}")
