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
    random_state: int = 42,
    balanced_test: bool = False,
    balanced_train_val: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target
        val_size: Fraction for validation
        test_size: Fraction for testing
        random_state: Random seed
        balanced_test: If True, create a 50/50 balanced test set
        balanced_train_val: If True, create 50/50 balanced train and validation sets

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if balanced_train_val:
        return split_data_fully_balanced(X, y, val_size, test_size, random_state, balanced_test)
    elif balanced_test:
        return split_data_balanced_test(X, y, val_size, test_size, random_state)

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


def split_data_balanced_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data with a perfectly balanced (50/50) test set.

    This function ensures the test set has equal numbers of phishing and non-phishing
    samples, which is useful for unbiased evaluation metrics.

    Args:
        X: Features
        y: Target
        val_size: Fraction for validation (of remaining data after test split)
        test_size: Fraction for testing (approximate, will be adjusted for balance)
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Creating BALANCED test set (50% phishing, 50% legitimate)...")

    # Separate by class
    phishing_mask = y == 1
    legitimate_mask = y == 0

    X_phishing = X[phishing_mask]
    y_phishing = y[phishing_mask]
    X_legitimate = X[legitimate_mask]
    y_legitimate = y[legitimate_mask]

    # Calculate test set size (equal samples from each class)
    total_samples = len(X)
    desired_test_samples = int(total_samples * test_size)
    test_samples_per_class = desired_test_samples // 2  # Split evenly

    # Ensure we don't exceed available samples
    max_test_per_class = min(len(X_phishing), len(X_legitimate), test_samples_per_class)

    logger.info(f"Test set size: {max_test_per_class * 2} samples ({max_test_per_class} phishing + {max_test_per_class} legitimate)")

    # Split phishing samples into train_val and test
    X_phishing_trainval, X_phishing_test, y_phishing_trainval, y_phishing_test = train_test_split(
        X_phishing, y_phishing,
        test_size=max_test_per_class,
        random_state=random_state
    )

    # Split legitimate samples into train_val and test
    X_legitimate_trainval, X_legitimate_test, y_legitimate_trainval, y_legitimate_test = train_test_split(
        X_legitimate, y_legitimate,
        test_size=max_test_per_class,
        random_state=random_state
    )

    # Combine test sets (now balanced 50/50)
    X_test = pd.concat([X_phishing_test, X_legitimate_test])
    y_test = pd.concat([y_phishing_test, y_legitimate_test])

    # Shuffle test set
    shuffle_idx = X_test.sample(frac=1, random_state=random_state).index
    X_test = X_test.loc[shuffle_idx]
    y_test = y_test.loc[shuffle_idx]

    # Combine remaining data for train/val split
    X_trainval = pd.concat([X_phishing_trainval, X_legitimate_trainval])
    y_trainval = pd.concat([y_phishing_trainval, y_legitimate_trainval])

    # Split train_val into train and validation (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval
    )

    # Log split sizes
    total = len(X)
    logger.info(f"\nFinal split sizes:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/total*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/total*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/total*100:.1f}%) - BALANCED")

    # Log class distributions
    logger.info(f"\nClass distribution:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()}")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()}")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()} (50/50 split)")

    # Verify test set is balanced
    test_balance = y_test.value_counts()
    if len(test_balance) == 2 and test_balance.iloc[0] == test_balance.iloc[1]:
        logger.info("✓ Test set is perfectly balanced!")
    else:
        logger.warning("⚠ Test set balance may be slightly off")

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_fully_balanced(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    balanced_test: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data with fully balanced (50/50) train, validation, and test sets.

    This function ensures ALL splits have equal numbers of phishing and non-phishing
    samples, which is useful for training on balanced data and preventing class imbalance issues.

    Args:
        X: Features
        y: Target
        val_size: Fraction for validation (approximate)
        test_size: Fraction for testing (approximate)
        random_state: Random seed
        balanced_test: Ignored (always creates balanced splits)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Creating FULLY BALANCED splits (50/50 for train, val, and test)...")

    # Separate by class
    phishing_mask = y == 1
    legitimate_mask = y == 0

    X_phishing = X[phishing_mask]
    y_phishing = y[phishing_mask]
    X_legitimate = X[legitimate_mask]
    y_legitimate = y[legitimate_mask]

    logger.info(f"Total phishing samples: {len(X_phishing)}")
    logger.info(f"Total legitimate samples: {len(X_legitimate)}")

    # Calculate split sizes based on the MINORITY class
    min_class_size = min(len(X_phishing), len(X_legitimate))
    total_balanced_samples = min_class_size * 2  # Equal from each class

    # Calculate samples per split
    test_samples_per_class = int(min_class_size * test_size)
    val_samples_per_class = int(min_class_size * val_size)
    train_samples_per_class = min_class_size - test_samples_per_class - val_samples_per_class

    logger.info(f"\nBalanced split sizes (per class):")
    logger.info(f"  Train: {train_samples_per_class} samples per class")
    logger.info(f"  Val:   {val_samples_per_class} samples per class")
    logger.info(f"  Test:  {test_samples_per_class} samples per class")

    # Shuffle phishing samples
    shuffle_idx_phishing = X_phishing.sample(frac=1, random_state=random_state).index
    X_phishing_shuffled = X_phishing.loc[shuffle_idx_phishing].reset_index(drop=True)
    y_phishing_shuffled = y_phishing.loc[shuffle_idx_phishing].reset_index(drop=True)

    # Split phishing: first test_samples, then val_samples, then train_samples
    X_phishing_test = X_phishing_shuffled.iloc[:test_samples_per_class].copy()
    y_phishing_test = y_phishing_shuffled.iloc[:test_samples_per_class].copy()

    X_phishing_val = X_phishing_shuffled.iloc[test_samples_per_class:test_samples_per_class + val_samples_per_class].copy()
    y_phishing_val = y_phishing_shuffled.iloc[test_samples_per_class:test_samples_per_class + val_samples_per_class].copy()

    X_phishing_train = X_phishing_shuffled.iloc[test_samples_per_class + val_samples_per_class:test_samples_per_class + val_samples_per_class + train_samples_per_class].copy()
    y_phishing_train = y_phishing_shuffled.iloc[test_samples_per_class + val_samples_per_class:test_samples_per_class + val_samples_per_class + train_samples_per_class].copy()

    # Shuffle legitimate samples
    shuffle_idx_legitimate = X_legitimate.sample(frac=1, random_state=random_state).index
    X_legitimate_shuffled = X_legitimate.loc[shuffle_idx_legitimate].reset_index(drop=True)
    y_legitimate_shuffled = y_legitimate.loc[shuffle_idx_legitimate].reset_index(drop=True)

    # Split legitimate: first test_samples, then val_samples, then train_samples
    X_legitimate_test = X_legitimate_shuffled.iloc[:test_samples_per_class].copy()
    y_legitimate_test = y_legitimate_shuffled.iloc[:test_samples_per_class].copy()

    X_legitimate_val = X_legitimate_shuffled.iloc[test_samples_per_class:test_samples_per_class + val_samples_per_class].copy()
    y_legitimate_val = y_legitimate_shuffled.iloc[test_samples_per_class:test_samples_per_class + val_samples_per_class].copy()

    X_legitimate_train = X_legitimate_shuffled.iloc[test_samples_per_class + val_samples_per_class:test_samples_per_class + val_samples_per_class + train_samples_per_class].copy()
    y_legitimate_train = y_legitimate_shuffled.iloc[test_samples_per_class + val_samples_per_class:test_samples_per_class + val_samples_per_class + train_samples_per_class].copy()

    # Combine and shuffle each split
    # Train set
    X_train = pd.concat([X_phishing_train, X_legitimate_train])
    y_train = pd.concat([y_phishing_train, y_legitimate_train])
    shuffle_idx = X_train.sample(frac=1, random_state=random_state).index
    X_train = X_train.loc[shuffle_idx]
    y_train = y_train.loc[shuffle_idx]

    # Validation set
    X_val = pd.concat([X_phishing_val, X_legitimate_val])
    y_val = pd.concat([y_phishing_val, y_legitimate_val])
    shuffle_idx = X_val.sample(frac=1, random_state=random_state + 1).index
    X_val = X_val.loc[shuffle_idx]
    y_val = y_val.loc[shuffle_idx]

    # Test set
    X_test = pd.concat([X_phishing_test, X_legitimate_test])
    y_test = pd.concat([y_phishing_test, y_legitimate_test])
    shuffle_idx = X_test.sample(frac=1, random_state=random_state + 2).index
    X_test = X_test.loc[shuffle_idx]
    y_test = y_test.loc[shuffle_idx]

    # Log split sizes
    total = len(X)
    logger.info(f"\nFinal BALANCED split sizes:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/total*100:.1f}%) - BALANCED 50/50")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/total*100:.1f}%) - BALANCED 50/50")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/total*100:.1f}%) - BALANCED 50/50")

    # Log class distributions
    logger.info(f"\nClass distribution:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()} (50/50)")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()} (50/50)")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()} (50/50)")

    # Verify all sets are balanced
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        balance = y_split.value_counts()
        if len(balance) == 2 and balance.iloc[0] == balance.iloc[1]:
            logger.info(f"✓ {name} set is perfectly balanced!")
        else:
            logger.warning(f"⚠ {name} set balance may be slightly off: {balance.to_dict()}")

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
