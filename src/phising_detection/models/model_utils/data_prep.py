"""
Data preparation utilities for phishing detection models.

This module provides functions for:
- Loading data from Hopsworks
- Train/val/test splitting
- Feature normalization
"""

import logging
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from phising_detection.utils import hopsworks_utils

logger = logging.getLogger(__name__)

# Feature definitions
FEATURE_COLUMNS = [
    'domain_age_days',
    'secure_percentage',
    'has_umbrella_rank',
    'umbrella_rank',
    'has_tls',
    'tls_valid_days',
    'url_length',
    'subdomain_count'
]

CONTINUOUS_FEATURES = [
    'domain_age_days',
    'secure_percentage',
    'umbrella_rank',
    'tls_valid_days',
    'url_length',
    'subdomain_count'
]

CATEGORICAL_FEATURES = [
    'has_umbrella_rank',
    'has_tls'
]


def load_data(project, feature_group_name: str = "urlscan_features", version: int = 1) -> pd.DataFrame:
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

    # Import here to avoid circular dependencies


    df = hopsworks_utils.read_feature_group(project, feature_group_name, version)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    target_column: str = 'is_phishing'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from DataFrame.

    Args:
        df: Input DataFrame
        feature_columns: List of feature column names (uses default if None)
        target_column: Name of target column

    Returns:
        Tuple of (features, target)
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    logger.info("Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()
    logger.info(f"Filled {missing_before - missing_after} missing values")

    # Log class distribution
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    class_balance = y.value_counts(normalize=True).to_dict()
    logger.info(f"Class balance: {class_balance}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target
        val_size: Fraction for validation
        test_size: Fraction for testing
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    train_size = 1.0 - test_size
    logger.info(f"Splitting data: {train_size:.0%} train+val, {test_size:.0%} test")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate train and validation
    val_size_adjusted = val_size / train_size
    logger.info(f"Splitting train+val: {1-val_size_adjusted:.0%} train, {val_size_adjusted:.0%} val")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    # Log split sizes
    total = len(X)
    logger.info(f"\nFinal split sizes:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/total*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/total*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/total*100:.1f}%)")

    # Log class distributions
    logger.info(f"\nClass distribution:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()}")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()}")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_features: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply z-score normalization to continuous features.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        continuous_features: List of continuous feature names (uses default if None)

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES

    logger.info("Applying z-score normalization to continuous features...")
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

    logger.info(f"Normalized features: {continuous_features}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data_pipeline(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, StandardScaler]:
    """
    Complete data preparation pipeline: extract features, split, and normalize.

    Args:
        df: Input DataFrame from Hopsworks
        val_size: Fraction for validation
        test_size: Fraction for testing
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger.info("Starting data preparation pipeline...")

    # Extract features and target
    X, y = prepare_features(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, val_size=val_size, test_size=test_size, random_state=random_state
    )

    # Normalize features
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    logger.info("Data preparation pipeline complete!")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
