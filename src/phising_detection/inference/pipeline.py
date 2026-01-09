"""
Phishing detection inference pipeline.

This module provides a complete inference pipeline for phishing detection including:
- Model loading from Hopsworks Model Registry
- Feature preprocessing
- Prediction with confidence scores
- Optional end-to-end URL scanning and prediction
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import joblib

from phising_detection.utils.hopsworks_utils import connect_to_hopsworks
from phising_detection.models.model_utils.data_prep import FEATURE_COLUMNS, CONTINUOUS_FEATURES
from phising_detection.features.urlscan_features import extract_features
from phising_detection.utils.urlscan import URLScanClient

logger = logging.getLogger(__name__)


class PhishingDetectionPipeline:
    """
    End-to-end inference pipeline for phishing detection.

    This class handles:
    - Loading trained models from Hopsworks
    - Preprocessing extracted features
    - Running inference
    - Optionally scanning URLs with URLScan.io
    """

    def __init__(
        self,
        model_name: str = "phishing_detector",
        model_version: Optional[int] = None,
        urlscan_api_key: Optional[str] = None
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_name: Name of the model in Hopsworks Model Registry
            model_version: Specific version to load (None = latest)
            urlscan_api_key: URLScan.io API key (optional, only needed for URL scanning)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.urlscan_client = None

        if urlscan_api_key:
            self.urlscan_client = URLScanClient(api_key=urlscan_api_key)

    def load_model_from_hopsworks(self) -> None:
        """
        Load model, scaler, and feature names from Hopsworks Model Registry.

        Raises:
            Exception: If model loading fails
        """
        logger.info(f"Loading model '{self.model_name}' from Hopsworks Model Registry...")

        # Connect to Hopsworks
        project = connect_to_hopsworks()
        mr = project.get_model_registry()

        # Get model from registry
        if self.model_version:
            model_registry = mr.get_model(self.model_name, version=self.model_version)
        else:
            model_registry = mr.get_model(self.model_name)

        logger.info(f"Found model: {self.model_name} version {model_registry.version}")

        # Download model artifacts to a temporary directory
        model_dir = model_registry.download()
        logger.info(f"Model artifacts downloaded to: {model_dir}")

        # Load model
        model_path = os.path.join(model_dir, "model.pkl")
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.txt")
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"Feature names loaded: {self.feature_names}")

        logger.info("Model pipeline initialized successfully!")

    def preprocess_features(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess extracted features to match model input format.

        Args:
            features_dict: Dictionary of extracted features

        Returns:
            Preprocessed DataFrame ready for model inference

        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None or self.scaler is None or self.feature_names is None:
            raise ValueError("Model not loaded. Call load_model_from_hopsworks() first.")

        # Create DataFrame with a single row
        df = pd.DataFrame([features_dict])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' missing, filling with default value")
                df[feature] = 0

        # Select and order features to match training
        df = df[self.feature_names]

        # Handle missing values (fill with median approximation)
        df = df.fillna(df.median())

        # Apply scaling to continuous features
        df_scaled = df.copy()
        df_scaled[CONTINUOUS_FEATURES] = self.scaler.transform(df[CONTINUOUS_FEATURES])

        return df_scaled

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on extracted features.

        Args:
            features_dict: Dictionary of extracted features

        Returns:
            Dictionary containing prediction results:
            - prediction: "PHISHING" or "LEGITIMATE"
            - confidence: Confidence score (0-1)
            - phishing_probability: Probability of phishing (0-1)
            - legitimate_probability: Probability of legitimate (0-1)
            - is_phishing: Boolean flag

        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_hopsworks() first.")

        # Preprocess features
        X = self.preprocess_features(features_dict)

        # Run inference
        logger.info("Running model inference...")
        prediction_proba = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]

        # prediction: 0 = legitimate, 1 = phishing
        is_phishing = bool(prediction)
        confidence = prediction_proba[1] if is_phishing else prediction_proba[0]

        result = {
            "prediction": "PHISHING" if is_phishing else "LEGITIMATE",
            "confidence": float(confidence),
            "phishing_probability": float(prediction_proba[1]),
            "legitimate_probability": float(prediction_proba[0]),
            "is_phishing": is_phishing
        }

        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")

        return result

    def predict_url(self, url: str) -> Dict[str, Any]:
        """
        End-to-end prediction: scan URL and predict if it's phishing.

        Args:
            url: URL to analyze

        Returns:
            Dictionary containing:
            - prediction: "PHISHING" or "LEGITIMATE"
            - confidence: Confidence score (0-1)
            - phishing_probability: Probability of phishing (0-1)
            - legitimate_probability: Probability of legitimate (0-1)
            - is_phishing: Boolean flag
            - features: Dictionary of extracted features
            - scan_uuid: URLScan UUID (if available)
            - error: Error message (if any)

        Raises:
            ValueError: If URLScan client is not initialized
        """
        if self.urlscan_client is None:
            raise ValueError("URLScan client not initialized. Provide urlscan_api_key in constructor.")

        try:
            # Step 1: Submit URL to URLScan.io and wait for results
            logger.info(f"Scanning URL: {url}")
            scan_result = self.urlscan_client.submit_and_wait(url)

            if not scan_result:
                return {
                    "error": "Failed to get scan results from URLScan.io",
                    "prediction": "ERROR",
                    "confidence": 0.0
                }

            # Step 2: Extract features from scan results
            logger.info("Extracting features from scan results...")
            features = extract_features(scan_result)
            logger.info(f"Extracted features: {features}")

            # Step 3: Run inference
            prediction_result = self.predict(features)

            # Add additional information
            prediction_result["features"] = features
            prediction_result["scan_uuid"] = scan_result.get("task", {}).get("uuid", "N/A")

            return prediction_result

        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return {
                "error": str(e),
                "prediction": "ERROR",
                "confidence": 0.0
            }

    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.scaler is not None and self.feature_names is not None
