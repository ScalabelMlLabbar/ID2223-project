"""
Simple training pipeline example using the modular components.

This demonstrates how to reuse the modular components for a custom training pipeline.
"""

import sys
import os
import logging


from ...utils import hopsworks_utils as hw
from . import data_prep
from . import model_configs
from . import evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_single_model_pipeline(model_name: str = "Random Forest"):
    """
    Example: Train a single specific model.

    Args:
        model_name: Name of the model to train (must match name in model_configs)
    """
    logger.info("=" * 80)
    logger.info(f"Training Pipeline for: {model_name}")
    logger.info("=" * 80)

    # 1. Load and prepare data
    project = hw.connect_to_hopsworks()
    df = data_prep.load_data(project)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = data_prep.prepare_data_pipeline(df)

    # 2. Get the specific model configuration
    all_models = model_configs.get_model_configs()
    if model_name not in all_models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(all_models.keys())}")

    model = all_models[model_name]

    # 3. Train the model
    trained_model = model_configs.train_model(model, X_train, y_train, model_name)

    # 4. Evaluate on validation set
    val_metrics = evaluation.evaluate_model(trained_model, X_val, y_val)
    logger.info(f"\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # 5. Final test evaluation with detailed report
    test_metrics = evaluation.evaluate_model_detailed(
        trained_model, X_test, y_test, model_name, "Test"
    )

    logger.info("\nTraining Pipeline Complete!")
    logger.info("=" * 80)

    return trained_model, scaler, test_metrics


if __name__ == "__main__":
    # Example 1: Train a single model
    logger.info("\n\nExample 1: Train Single Model\n")
    model, scaler, metrics = train_single_model_pipeline("Random Forest")
