"""
Train the best phishing detection model with extensive hyperparameter search
and upload to Hopsworks Model Registry.

This script:
1. Loads data from Hopsworks
2. Performs extensive hyperparameter search on the best model
3. Trains final model on full train+val set
4. Evaluates on test set
5. Uploads model and artifacts to Hopsworks Model Registry
"""

import logging
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .model_utils import data_prep, model_configs, evaluation
from ..utils import hopsworks_utils as hw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_upload_model(
    model_name: str = "Random Forest",
    feature_group_name: str = "urlscan_features",
    feature_group_version: int = 1,
    val_size: float = 0.15,
    test_size: float = 0.15,
    cv_folds: int = 10,
    n_iter: int = 100,
    registry_model_name: str = "phishing_detector"
):
    """
    Train best model with extensive hyperparameter search and upload to Hopsworks.

    Args:
        model_name: Name of model to train (must be in model_configs)
        feature_group_name: Hopsworks feature group name
        feature_group_version: Feature group version
        val_size: Validation set size
        test_size: Test set size
        cv_folds: Number of CV folds for hyperparameter search
        n_iter: Number of iterations for RandomizedSearchCV
        registry_model_name: Name for model in Hopsworks registry
    """
    logger.info("=" * 80)
    logger.info("FINAL MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"CV Folds: {cv_folds}")
    logger.info(f"RandomizedSearchCV iterations: {n_iter}")

    # 1. Connect to Hopsworks and load data
    logger.info("\n1. Loading data from Hopsworks...")
    project = hw.connect_to_hopsworks()
    raw_data = data_prep.load_data(project, feature_group_name, feature_group_version)

    # 2. Prepare data (keep raw data for re-normalization)
    logger.info("\n2. Preparing data...")
    X, y = data_prep.prepare_features(raw_data)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = data_prep.split_data(
        X, y, val_size=val_size, test_size=test_size
    )

    # Initial normalization for hyperparameter search
    X_train, X_val, X_test, _ = data_prep.normalize_features(
        X_train_raw, X_val_raw, X_test_raw
    )

    # 3. Extensive hyperparameter search
    logger.info(f"\n3. Extensive hyperparameter search for {model_name}...")
    best_model, search_results = model_configs.extensive_hyperparameter_search(
        model_name,
        X_train,
        y_train,
        cv_folds=cv_folds,
        n_iter=n_iter,
        use_random_search=True
    )

    # 4. Evaluate on validation set
    logger.info("\n4. Evaluating on validation set...")
    val_metrics = evaluation.evaluate_model_detailed(
        best_model, X_val, y_val, model_name, "Validation"
    )

    # 5. Combine train+val and re-normalize with combined statistics
    logger.info("\n5. Combining train+val and re-normalizing...")
    X_train_val_raw = pd.concat([X_train_raw, X_val_raw])
    y_train_val = pd.concat([y_train, y_val])

    # Re-normalize with combined statistics
    final_scaler = StandardScaler()
    X_train_val = X_train_val_raw.copy()
    X_test_final = X_test_raw.copy()

    continuous_features = data_prep.CONTINUOUS_FEATURES
    X_train_val[continuous_features] = final_scaler.fit_transform(X_train_val_raw[continuous_features])
    X_test_final[continuous_features] = final_scaler.transform(X_test_raw[continuous_features])

    # 6. Train final model on combined train+val
    logger.info(f"\n6. Training final {model_name} on combined train+val set...")
    final_model = model_configs.train_model(
        best_model, X_train_val, y_train_val, f"Final {model_name}"
    )

    # 7. Final evaluation on test set
    logger.info("\n7. Final evaluation on test set...")
    test_metrics = evaluation.evaluate_model_detailed(
        final_model, X_test_final, y_test, model_name, "Test"
    )

    # 8. Prepare model artifacts
    logger.info("\n8. Preparing model artifacts for upload...")

    # Combine all metrics (must be numbers only for Hopsworks)
    all_metrics = {
        'test_accuracy': float(test_metrics['Accuracy']),
        'test_precision': float(test_metrics['Precision']),
        'test_recall': float(test_metrics['Recall']),
        'test_f1_score': float(test_metrics['F1 Score']),
        'test_roc_auc': float(test_metrics['ROC-AUC']),
        'val_accuracy': float(val_metrics['Accuracy']),
        'val_f1_score': float(val_metrics['F1 Score']),
        'best_cv_score': float(search_results.best_score_),
        'n_features': int(X_train_val.shape[1]),
        'n_train_samples': int(len(X_train_val)),
        'n_test_samples': int(len(X_test_final))
    }

    # Model description
    description = f"""
    Phishing Detection Model - {model_name}

    Trained with extensive hyperparameter search:
    - CV Folds: {cv_folds}
    - RandomizedSearchCV iterations: {n_iter}
    - Best CV Score: {search_results.best_score_:.4f}

    Best Parameters: {search_results.best_params_}

    Test Performance:
    - Accuracy: {test_metrics['Accuracy']:.4f}
    - Precision: {test_metrics['Precision']:.4f}
    - Recall: {test_metrics['Recall']:.4f}
    - F1 Score: {test_metrics['F1 Score']:.4f}
    - ROC-AUC: {test_metrics['ROC-AUC']:.4f}

    Features: {list(X_train_val.columns)}
    """

    # 9. Save model and scaler to Hopsworks Model Registry
    logger.info("\n9. Uploading model, scaler, and artifacts to Hopsworks Model Registry...")

    hw.save_model_to_registry(
        project,
        final_model,
        registry_model_name,
        metrics=all_metrics,
        description=description,
        scaler=final_scaler,
        feature_names=list(X_train_val.columns)
    )

    # 10. Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Best Parameters: {search_results.best_params_}")
    logger.info(f"Best CV Score: {search_results.best_score_:.4f}")
    logger.info(f"\nTest Performance:")
    logger.info(f"  Accuracy:  {test_metrics['Accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['Precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['Recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['F1 Score']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['ROC-AUC']:.4f}")
    logger.info(f"\nModel uploaded to Hopsworks Model Registry as: {registry_model_name}")
    logger.info("=" * 80)

    return final_model, final_scaler, test_metrics, search_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train best phishing detection model with extensive hyperparameter search"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Random Forest",
        choices=["Random Forest", "Gradient Boosting", "Logistic Regression"],
        help="Model to train (default: Random Forest)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of RandomizedSearchCV iterations (default: 100)"
    )
    parser.add_argument(
        "--registry-name",
        type=str,
        default="phishing_detector",
        help="Name for model in Hopsworks registry (default: phishing_detector)"
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="urlscan_features",
        help="Hopsworks feature group name (default: urlscan_features)"
    )
    parser.add_argument(
        "--feature-group-version",
        type=int,
        default=1,
        help="Feature group version (default: 1)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, scaler, metrics, search = train_and_upload_model(
        model_name=args.model,
        feature_group_name=args.feature_group,
        feature_group_version=args.feature_group_version,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        registry_model_name=args.registry_name
    )

    print(f"\n✓ Model successfully trained and uploaded!")
    print(f"  Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics['F1 Score']:.4f}")
