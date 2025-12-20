"""URLScan.io API client for URL analysis."""

import os
import time
from typing import Dict, Any, Optional

import requests


class URLScanError(Exception):
    """Custom exception for URLScan API errors."""
    pass


class URLScanClient:
    """Client for interacting with URLScan.io API."""

    BASE_URL = "https://urlscan.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize URLScan client.

        Args:
            api_key: URLScan.io API key. If not provided, will try to get from
                    URLSCAN_API_KEY environment variable.

        Raises:
            URLScanError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("URLSCAN_API_KEY")
        if not self.api_key:
            raise URLScanError(
                "No API key provided. Set URLSCAN_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "API-Key": self.api_key,
            "Content-Type": "application/json"
        })

    def submit_url(
        self,
        url: str,
        visibility: str = "public",
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Submit a URL for scanning.

        Args:
            url: The URL to scan
            visibility: Visibility of the scan ('public', 'unlisted', or 'private')
            tags: Optional list of tags for categorization

        Returns:
            Dictionary containing scan submission response with 'uuid' and 'api' fields

        Raises:
            URLScanError: If submission fails
        """
        endpoint = f"{self.BASE_URL}/scan/"

        payload = {
            "url": url,
            "visibility": visibility
        }

        if tags:
            payload["tags"] = tags

        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise URLScanError("Rate limit exceeded. Please wait before retrying.")
            elif response.status_code == 400:
                raise URLScanError(f"Bad request: {response.text}")
            else:
                raise URLScanError(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise URLScanError(f"Request failed: {e}")

    def get_result(self, uuid: str) -> Dict[str, Any]:
        """
        Get scan results by UUID.

        Args:
            uuid: The scan UUID returned from submit_url

        Returns:
            Dictionary containing scan results

        Raises:
            URLScanError: If retrieval fails
        """
        endpoint = f"{self.BASE_URL}/result/{uuid}/"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise URLScanError(
                    f"Scan not found or not ready yet. UUID: {uuid}"
                )
            else:
                raise URLScanError(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise URLScanError(f"Request failed: {e}")

    def submit_and_wait(
        self,
        url: str,
        visibility: str = "public",
        tags: Optional[list] = None,
        max_wait: int = 60,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Submit a URL and wait for results.

        Args:
            url: The URL to scan
            visibility: Visibility of the scan
            tags: Optional list of tags
            max_wait: Maximum time to wait for results (seconds)
            poll_interval: Time between polling attempts (seconds)

        Returns:
            Dictionary containing scan results

        Raises:
            URLScanError: If submission or retrieval fails, or timeout occurs
        """
        # Submit URL
        submission = self.submit_url(url, visibility, tags)
        uuid = submission.get("uuid")

        if not uuid:
            raise URLScanError("No UUID returned from submission")

        # Wait for results
        elapsed = 0
        while elapsed < max_wait:
            try:
                time.sleep(poll_interval)
                elapsed += poll_interval

                result = self.get_result(uuid)
                return result

            except URLScanError as e:
                if "not found or not ready" in str(e):
                    # Scan not ready yet, continue waiting
                    continue
                else:
                    # Other error, raise it
                    raise

        raise URLScanError(
            f"Timeout waiting for scan results. UUID: {uuid}. "
            f"You can retrieve results later using get_result('{uuid}')"
        )

    def search(self, query: str, size: int = 100) -> Dict[str, Any]:
        """
        Search URLScan.io database.

        Args:
            query: Search query (e.g., 'domain:example.com')
            size: Number of results to return (max 10000)

        Returns:
            Dictionary containing search results

        Raises:
            URLScanError: If search fails
        """
        endpoint = f"{self.BASE_URL}/search/"

        params = {
            "q": query,
            "size": min(size, 10000)
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            raise URLScanError(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise URLScanError(f"Request failed: {e}")

    def get_verdict(self, uuid: str) -> Optional[str]:
        """
        Get the verdict (malicious/safe) for a scan.

        Args:
            uuid: The scan UUID

        Returns:
            Verdict string ('malicious', 'safe', or None if not available)

        Raises:
            URLScanError: If retrieval fails
        """
        result = self.get_result(uuid)

        # Extract verdict from results
        verdicts = result.get("verdicts", {})
        overall = verdicts.get("overall", {})

        if overall.get("malicious", False):
            return "malicious"
        elif overall.get("score", 0) == 0:
            return "safe"
        else:
            return None
