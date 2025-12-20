"""Tests for data loading functionality."""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phising_detection.data import load_phishing_urls


class TestLoadPhishingUrls:
    """Tests for load_phishing_urls function."""

    def test_load_phishing_urls_basic(self, temp_phishing_file, sample_phishing_urls):
        """Test basic loading of phishing URLs."""
        df = load_phishing_urls(temp_phishing_file)

        # Check DataFrame shape
        assert len(df) == len(sample_phishing_urls)
        assert df.shape[1] == 3  # url_id, url, is_phishing

        # Check column names
        assert list(df.columns) == ['url_id', 'url', 'is_phishing']

        # Check data types
        assert df['url_id'].dtype == 'int64'
        assert df['url'].dtype == 'object'
        assert df['is_phishing'].dtype == 'int64'

    def test_load_phishing_urls_content(self, temp_phishing_file, sample_phishing_urls):
        """Test that URLs are loaded correctly."""
        df = load_phishing_urls(temp_phishing_file)

        # Check URLs match
        assert df['url'].tolist() == sample_phishing_urls

        # Check url_ids are sequential
        assert df['url_id'].tolist() == list(range(len(sample_phishing_urls)))

        # Check all are labeled as phishing
        assert all(df['is_phishing'] == 1)

    def test_load_phishing_urls_with_label_false(self, temp_phishing_file):
        """Test loading URLs without phishing label."""
        df = load_phishing_urls(temp_phishing_file, is_phishing=False)

        # Check that is_phishing is 0
        assert all(df['is_phishing'] == 0)

    def test_load_phishing_urls_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_phishing_urls("non_existent_file.txt")

        assert "File not found" in str(exc_info.value)

    def test_load_phishing_urls_empty_file(self, temp_empty_file):
        """Test loading from empty file."""
        df = load_phishing_urls(temp_empty_file)

        # Should return empty DataFrame with correct columns
        assert len(df) == 0
        assert list(df.columns) == ['url_id', 'url', 'is_phishing']

    def test_load_phishing_urls_with_blank_lines(
        self, temp_file_with_blank_lines, sample_phishing_urls
    ):
        """Test that blank lines are filtered out."""
        df = load_phishing_urls(temp_file_with_blank_lines)

        # Should only have 4 URLs (blank lines removed)
        assert len(df) == 4

        # Check that only non-empty URLs are present
        assert sample_phishing_urls[0] in df['url'].values
        assert sample_phishing_urls[1] in df['url'].values
        assert sample_phishing_urls[2] in df['url'].values
        assert sample_phishing_urls[3] in df['url'].values

    def test_load_phishing_urls_pathlib_path(self, temp_phishing_file):
        """Test that function accepts pathlib.Path objects."""
        path = Path(temp_phishing_file)
        df = load_phishing_urls(path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_phishing_urls_string_path(self, temp_phishing_file):
        """Test that function accepts string paths."""
        df = load_phishing_urls(str(temp_phishing_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_phishing_urls_returns_dataframe(self, temp_phishing_file):
        """Test that function returns a pandas DataFrame."""
        result = load_phishing_urls(temp_phishing_file)

        assert isinstance(result, pd.DataFrame)

    def test_load_phishing_urls_url_id_uniqueness(self, temp_phishing_file):
        """Test that url_id values are unique."""
        df = load_phishing_urls(temp_phishing_file)

        assert df['url_id'].is_unique
        assert len(df['url_id'].unique()) == len(df)
