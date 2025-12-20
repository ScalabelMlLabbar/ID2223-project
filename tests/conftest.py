"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_phishing_urls():
    """Sample phishing URLs for testing."""
    return [
        "http://fake-bank-login.malicious.com/signin",
        "http://paypal-verify.scam.net/account",
        "https://amazon-security.fake.org/update",
        "http://apple-id-locked.phishing.com/unlock",
        "ftp://suspicious-link.bad.net/file.html"
    ]


@pytest.fixture
def sample_legitimate_urls():
    """Sample legitimate URLs for testing."""
    return [
        "https://www.google.com",
        "https://github.com/user/repo",
        "https://www.wikipedia.org",
        "https://www.python.org"
    ]


@pytest.fixture
def temp_phishing_file(sample_phishing_urls, tmp_path):
    """Create a temporary file with phishing URLs."""
    file_path = tmp_path / "test_phishing.txt"
    file_path.write_text("\n".join(sample_phishing_urls))
    return file_path


@pytest.fixture
def temp_legitimate_file(sample_legitimate_urls, tmp_path):
    """Create a temporary file with legitimate URLs."""
    file_path = tmp_path / "test_legitimate.txt"
    file_path.write_text("\n".join(sample_legitimate_urls))
    return file_path


@pytest.fixture
def temp_empty_file(tmp_path):
    """Create a temporary empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path


@pytest.fixture
def temp_file_with_blank_lines(sample_phishing_urls, tmp_path):
    """Create a temporary file with blank lines."""
    file_path = tmp_path / "test_with_blanks.txt"
    urls_with_blanks = [
        sample_phishing_urls[0],
        "",
        sample_phishing_urls[1],
        "   ",
        sample_phishing_urls[2],
        "\n",
        sample_phishing_urls[3]
    ]
    file_path.write_text("\n".join(urls_with_blanks))
    return file_path
