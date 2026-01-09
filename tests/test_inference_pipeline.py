"""Tests for inference pipeline."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.phising_detection.inference.pipeline import PhishingDetectionPipeline


class TestPhishingDetectionPipeline:
    """Tests for PhishingDetectionPipeline class."""

    def test_init_without_urlscan(self):
        """Test initialization without URLScan API key."""
        pipeline = PhishingDetectionPipeline(
            model_name="test_model",
            model_version=1
        )
        assert pipeline.model_name == "test_model"
        assert pipeline.model_version == 1
        assert pipeline.model is None
        assert pipeline.scaler is None
        assert pipeline.feature_names is None
        assert pipeline.urlscan_client is None

    def test_init_with_urlscan(self):
        """Test initialization with URLScan API key."""
        with patch('src.phising_detection.inference.pipeline.URLScanClient') as mock_client:
            pipeline = PhishingDetectionPipeline(
                model_name="test_model",
                urlscan_api_key="test_key"
            )
            mock_client.assert_called_once_with(api_key="test_key")
            assert pipeline.urlscan_client is not None

    def test_is_loaded_false(self):
        """Test is_loaded returns False when model not loaded."""
        pipeline = PhishingDetectionPipeline()
        assert pipeline.is_loaded() is False

    def test_is_loaded_true(self):
        """Test is_loaded returns True when model is loaded."""
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.scaler = Mock()
        pipeline.feature_names = ['feature1', 'feature2']
        assert pipeline.is_loaded() is True

    @patch('src.phising_detection.inference.pipeline.connect_to_hopsworks')
    @patch('src.phising_detection.inference.pipeline.joblib.load')
    def test_load_model_from_hopsworks(self, mock_joblib_load, mock_connect):
        """Test loading model from Hopsworks."""
        # Setup mocks
        mock_project = Mock()
        mock_mr = Mock()
        mock_model_registry = Mock()
        mock_model_registry.version = 1
        mock_model_registry.download.return_value = "/tmp/model_dir"

        mock_project.get_model_registry.return_value = mock_mr
        mock_mr.get_model.return_value = mock_model_registry
        mock_connect.return_value = mock_project

        # Mock model and scaler
        mock_model = Mock()
        mock_scaler = Mock()
        mock_joblib_load.side_effect = [mock_model, mock_scaler]

        # Create temporary feature_names file
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_names_path = os.path.join(tmpdir, "feature_names.txt")
            with open(feature_names_path, 'w') as f:
                f.write("feature1\nfeature2\nfeature3")

            # Mock download to return our temp dir
            mock_model_registry.download.return_value = tmpdir

            # Test
            pipeline = PhishingDetectionPipeline(model_name="test_model")
            pipeline.load_model_from_hopsworks()

            # Assertions
            assert pipeline.model == mock_model
            assert pipeline.scaler == mock_scaler
            assert pipeline.feature_names == ['feature1', 'feature2', 'feature3']
            mock_connect.assert_called_once()
            mock_mr.get_model.assert_called_once_with("test_model")

    @patch('src.phising_detection.inference.pipeline.connect_to_hopsworks')
    @patch('src.phising_detection.inference.pipeline.joblib.load')
    def test_load_model_with_version(self, mock_joblib_load, mock_connect):
        """Test loading specific model version from Hopsworks."""
        # Setup mocks
        mock_project = Mock()
        mock_mr = Mock()
        mock_model_registry = Mock()
        mock_model_registry.version = 2
        mock_model_registry.download.return_value = "/tmp/model_dir"

        mock_project.get_model_registry.return_value = mock_mr
        mock_mr.get_model.return_value = mock_model_registry
        mock_connect.return_value = mock_project

        mock_joblib_load.side_effect = [Mock(), Mock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_names_path = os.path.join(tmpdir, "feature_names.txt")
            with open(feature_names_path, 'w') as f:
                f.write("feature1")

            mock_model_registry.download.return_value = tmpdir

            # Test with specific version
            pipeline = PhishingDetectionPipeline(model_name="test_model", model_version=2)
            pipeline.load_model_from_hopsworks()

            # Should request version 2
            mock_mr.get_model.assert_called_once_with("test_model", version=2)

    def test_preprocess_features_not_loaded(self):
        """Test preprocessing fails when model not loaded."""
        pipeline = PhishingDetectionPipeline()
        features = {'feature1': 10, 'feature2': 20}

        with pytest.raises(ValueError) as exc_info:
            pipeline.preprocess_features(features)
        assert "Model not loaded" in str(exc_info.value)

    def test_preprocess_features_success(self):
        """Test successful feature preprocessing."""
        # Setup pipeline with mock components
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.feature_names = [
            'domain_age_days',
            'secure_percentage',
            'has_umbrella_rank',
            'umbrella_rank',
            'has_tls',
            'tls_valid_days',
            'url_length',
            'subdomain_count'
        ]

        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.5, 0.8, 5000, 365, 25, 1]])
        pipeline.scaler = mock_scaler

        # Test features
        features = {
            'domain_age_days': 3000,
            'secure_percentage': 95.0,
            'has_umbrella_rank': 1,
            'umbrella_rank': 5000,
            'has_tls': 1,
            'tls_valid_days': 365,
            'url_length': 25,
            'subdomain_count': 1
        }

        result = pipeline.preprocess_features(features)

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == pipeline.feature_names
        assert len(result) == 1
        mock_scaler.transform.assert_called_once()

    def test_preprocess_features_with_missing_features(self):
        """Test preprocessing handles missing features."""
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.feature_names = [
            'domain_age_days', 'secure_percentage', 'has_umbrella_rank',
            'umbrella_rank', 'has_tls', 'tls_valid_days', 'url_length', 'subdomain_count'
        ]

        mock_scaler = Mock()
        # Mock the scaler to return the same shape as input continuous features (6 features)
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        pipeline.scaler = mock_scaler

        # Only provide some features (missing subdomain_count)
        features = {
            'domain_age_days': 3000,
            'secure_percentage': 95.0,
            'has_umbrella_rank': 1,
            'umbrella_rank': 5000,
            'has_tls': 1,
            'tls_valid_days': 365,
            'url_length': 25
        }

        result = pipeline.preprocess_features(features)

        # Should add missing subdomain_count and all features should be present
        assert 'subdomain_count' in result.columns
        assert len(result.columns) == 8  # All 8 features should be present
        assert list(result.columns) == pipeline.feature_names

    def test_predict_not_loaded(self):
        """Test prediction fails when model not loaded."""
        pipeline = PhishingDetectionPipeline()
        features = {'feature1': 10}

        with pytest.raises(ValueError) as exc_info:
            pipeline.predict(features)
        assert "Model not loaded" in str(exc_info.value)

    def test_predict_phishing(self):
        """Test prediction for phishing URL."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.feature_names = [
            'domain_age_days', 'secure_percentage', 'has_umbrella_rank',
            'umbrella_rank', 'has_tls', 'tls_valid_days', 'url_length', 'subdomain_count'
        ]

        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% phishing
        mock_model.predict.return_value = np.array([1])  # Phishing
        pipeline.model = mock_model

        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        pipeline.scaler = mock_scaler

        # Test features
        features = {
            'domain_age_days': 10,
            'secure_percentage': 50.0,
            'has_umbrella_rank': 0,
            'umbrella_rank': 999999,
            'has_tls': 0,
            'tls_valid_days': 0,
            'url_length': 150,
            'subdomain_count': 5
        }

        result = pipeline.predict(features)

        # Assertions
        assert result['prediction'] == "PHISHING"
        assert result['is_phishing'] is True
        assert result['confidence'] == 0.8
        assert result['phishing_probability'] == 0.8
        assert result['legitimate_probability'] == 0.2

    def test_predict_legitimate(self):
        """Test prediction for legitimate URL."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.feature_names = [
            'domain_age_days', 'secure_percentage', 'has_umbrella_rank',
            'umbrella_rank', 'has_tls', 'tls_valid_days', 'url_length', 'subdomain_count'
        ]

        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])  # 90% legitimate
        mock_model.predict.return_value = np.array([0])  # Legitimate
        pipeline.model = mock_model

        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        pipeline.scaler = mock_scaler

        # Test features
        features = {
            'domain_age_days': 3000,
            'secure_percentage': 95.0,
            'has_umbrella_rank': 1,
            'umbrella_rank': 5000,
            'has_tls': 1,
            'tls_valid_days': 365,
            'url_length': 25,
            'subdomain_count': 1
        }

        result = pipeline.predict(features)

        # Assertions
        assert result['prediction'] == "LEGITIMATE"
        assert result['is_phishing'] is False
        assert result['confidence'] == 0.9
        assert result['phishing_probability'] == 0.1
        assert result['legitimate_probability'] == 0.9

    def test_predict_url_without_urlscan(self):
        """Test predict_url fails without URLScan client."""
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.scaler = Mock()
        pipeline.feature_names = ['feature1']

        with pytest.raises(ValueError) as exc_info:
            pipeline.predict_url("https://example.com")
        assert "URLScan client not initialized" in str(exc_info.value)

    @patch('src.phising_detection.inference.pipeline.extract_features')
    def test_predict_url_success(self, mock_extract_features):
        """Test successful end-to-end URL prediction."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.feature_names = [
            'domain_age_days', 'secure_percentage', 'has_umbrella_rank',
            'umbrella_rank', 'has_tls', 'tls_valid_days', 'url_length', 'subdomain_count'
        ]

        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_model.predict.return_value = np.array([0])
        pipeline.model = mock_model

        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        pipeline.scaler = mock_scaler

        # Mock URLScan client
        mock_urlscan_client = Mock()
        mock_scan_result = {
            'task': {'uuid': 'test-uuid-123'},
            'page': {'domainAgeDays': 3000},
            'stats': {'securePercentage': 95}
        }
        mock_urlscan_client.submit_and_wait.return_value = mock_scan_result
        pipeline.urlscan_client = mock_urlscan_client

        # Mock extracted features
        extracted_features = {
            'domain_age_days': 3000,
            'secure_percentage': 95.0,
            'has_umbrella_rank': 1,
            'umbrella_rank': 5000,
            'has_tls': 1,
            'tls_valid_days': 365,
            'url_length': 25,
            'subdomain_count': 1
        }
        mock_extract_features.return_value = extracted_features

        # Test
        result = pipeline.predict_url("https://example.com")

        # Assertions
        assert result['prediction'] == "LEGITIMATE"
        assert result['confidence'] == 0.7
        assert result['features'] == extracted_features
        assert result['scan_uuid'] == 'test-uuid-123'
        mock_urlscan_client.submit_and_wait.assert_called_once_with("https://example.com")

    def test_predict_url_scan_fails(self):
        """Test predict_url handles URLScan failure."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.scaler = Mock()
        pipeline.feature_names = ['feature1']

        # Mock URLScan client that returns None
        mock_urlscan_client = Mock()
        mock_urlscan_client.submit_and_wait.return_value = None
        pipeline.urlscan_client = mock_urlscan_client

        # Test
        result = pipeline.predict_url("https://example.com")

        # Should return error
        assert 'error' in result
        assert result['prediction'] == "ERROR"
        assert result['confidence'] == 0.0

    @patch('src.phising_detection.inference.pipeline.extract_features')
    def test_predict_url_exception_handling(self, mock_extract_features):
        """Test predict_url handles exceptions gracefully."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.model = Mock()
        pipeline.scaler = Mock()
        pipeline.feature_names = ['feature1']

        # Mock URLScan client
        mock_urlscan_client = Mock()
        mock_urlscan_client.submit_and_wait.return_value = {'task': {}}
        pipeline.urlscan_client = mock_urlscan_client

        # Mock extract_features to raise exception
        mock_extract_features.side_effect = Exception("Test error")

        # Test
        result = pipeline.predict_url("https://example.com")

        # Should return error
        assert 'error' in result
        assert result['prediction'] == "ERROR"
        assert "Test error" in result['error']

    def test_predict_numerical_stability(self):
        """Test prediction handles edge cases in probabilities."""
        # Setup pipeline
        pipeline = PhishingDetectionPipeline()
        pipeline.feature_names = [
            'domain_age_days', 'secure_percentage', 'has_umbrella_rank',
            'umbrella_rank', 'has_tls', 'tls_valid_days', 'url_length', 'subdomain_count'
        ]

        # Mock model with extreme probabilities
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.999999, 0.000001]])
        mock_model.predict.return_value = np.array([0])
        pipeline.model = mock_model

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        pipeline.scaler = mock_scaler

        features = {
            'domain_age_days': 3000,
            'secure_percentage': 95.0,
            'has_umbrella_rank': 1,
            'umbrella_rank': 5000,
            'has_tls': 1,
            'tls_valid_days': 365,
            'url_length': 25,
            'subdomain_count': 1
        }
        result = pipeline.predict(features)

        # Should handle extreme values correctly
        assert result['confidence'] > 0.99
        assert result['prediction'] == "LEGITIMATE"
        assert result['phishing_probability'] < 0.01
