"""Tests for URLScan.io API client."""

import pytest
import responses
from unittest.mock import patch
import sys
from pathlib import Path

from src.phising_detection.utils.urlscan import URLScanClient, URLScanError


class TestURLScanClient:
    """Tests for URLScanClient class."""

    def test_init_with_api_key(self):
        """Test initialization with API key provided."""
        client = URLScanClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("URLSCAN_API_KEY", "env_key")
        client = URLScanClient()
        assert client.api_key == "env_key"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("URLSCAN_API_KEY", raising=False)
        with pytest.raises(URLScanError) as exc_info:
            URLScanClient()
        assert "No API key provided" in str(exc_info.value)

    @responses.activate
    def test_submit_url_success(self):
        """Test successful URL submission."""
        # Mock API response
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            json={
                "uuid": "test-uuid-123",
                "result": "https://urlscan.io/result/test-uuid-123/",
                "api": "https://urlscan.io/api/v1/result/test-uuid-123/"
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        result = client.submit_url("https://example.com")

        assert result["uuid"] == "test-uuid-123"
        assert "result" in result
        assert len(responses.calls) == 1
        assert responses.calls[0].request.url == "https://urlscan.io/api/v1/scan/"

    @responses.activate
    def test_submit_url_with_tags(self):
        """Test URL submission with tags."""
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            json={"uuid": "test-uuid", "result": "url"},
            status=200
        )

        client = URLScanClient(api_key="test_key")
        client.submit_url(
            "https://example.com",
            visibility="unlisted",
            tags=["phishing", "test"]
        )

        request_body = responses.calls[0].request.body
        assert b"phishing" in request_body
        assert b"test" in request_body

    @responses.activate
    def test_submit_url_rate_limit(self):
        """Test rate limit error handling."""
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            status=429
        )

        client = URLScanClient(api_key="test_key")
        with pytest.raises(URLScanError) as exc_info:
            client.submit_url("https://example.com")
        assert "Rate limit exceeded" in str(exc_info.value)

    @responses.activate
    def test_submit_url_bad_request(self):
        """Test bad request error handling."""
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            body="Invalid URL",
            status=400
        )

        client = URLScanClient(api_key="test_key")
        with pytest.raises(URLScanError) as exc_info:
            client.submit_url("not-a-valid-url")
        assert "Bad request" in str(exc_info.value)

    @responses.activate
    def test_get_result_success(self):
        """Test successful result retrieval."""
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            json={
                "page": {
                    "url": "https://example.com",
                    "title": "Example Domain"
                },
                "verdicts": {
                    "overall": {
                        "score": 0,
                        "malicious": False
                    }
                },
                "task": {"uuid": "test-uuid"}
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        result = client.get_result("test-uuid")

        assert result["page"]["url"] == "https://example.com"
        assert result["verdicts"]["overall"]["malicious"] is False

    @responses.activate
    def test_get_result_not_found(self):
        """Test result not found error."""
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            status=404
        )

        client = URLScanClient(api_key="test_key")
        with pytest.raises(URLScanError) as exc_info:
            client.get_result("test-uuid")
        assert "not found or not ready" in str(exc_info.value)

    @responses.activate
    def test_search_success(self):
        """Test search functionality."""
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/search/",
            json={
                "total": 2,
                "results": [
                    {"task": {"url": "https://example.com"}},
                    {"task": {"url": "https://example.org"}}
                ]
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        results = client.search(query="domain:example.com", size=10)

        assert results["total"] == 2
        assert len(results["results"]) == 2

    @responses.activate
    def test_get_verdict_malicious(self):
        """Test verdict extraction for malicious URL."""
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            json={
                "verdicts": {
                    "overall": {
                        "score": 100,
                        "malicious": True
                    }
                }
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        verdict = client.get_verdict("test-uuid")

        assert verdict == "malicious"

    @responses.activate
    def test_get_verdict_safe(self):
        """Test verdict extraction for safe URL."""
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            json={
                "verdicts": {
                    "overall": {
                        "score": 0,
                        "malicious": False
                    }
                }
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        verdict = client.get_verdict("test-uuid")

        assert verdict == "safe"

    @responses.activate
    def test_submit_and_wait_success(self):
        """Test submit and wait for results."""
        # Mock submission
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            json={
                "uuid": "test-uuid",
                "result": "https://urlscan.io/result/test-uuid/"
            },
            status=200
        )

        # Mock result retrieval
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            json={
                "page": {"title": "Example"},
                "task": {"uuid": "test-uuid"}
            },
            status=200
        )

        client = URLScanClient(api_key="test_key")
        result = client.submit_and_wait(
            "https://example.com",
            max_wait=10,
            poll_interval=1
        )

        assert result["page"]["title"] == "Example"

    @responses.activate
    def test_submit_and_wait_timeout(self):
        """Test timeout in submit_and_wait."""
        # Mock submission
        responses.add(
            responses.POST,
            "https://urlscan.io/api/v1/scan/",
            json={"uuid": "test-uuid"},
            status=200
        )

        # Mock result always returning 404
        responses.add(
            responses.GET,
            "https://urlscan.io/api/v1/result/test-uuid/",
            status=404
        )

        client = URLScanClient(api_key="test_key")
        with pytest.raises(URLScanError) as exc_info:
            client.submit_and_wait(
                "https://example.com",
                max_wait=3,
                poll_interval=1
            )
        assert "Timeout" in str(exc_info.value)
