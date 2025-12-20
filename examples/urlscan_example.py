"""Example usage of URLScan.io API client."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phising_detection.api import URLScanClient, URLScanError


def main():
    """Demonstrate URLScan.io API usage."""

    try:
        # Initialize client (reads API key from URLSCAN_API_KEY env var)
        client = URLScanClient()

        # Example URL to scan
        test_url = "https://example.com"

        print(f"Submitting URL for scanning: {test_url}")

        # Option 1: Submit and get UUID for later retrieval
        submission = client.submit_url(
            url=test_url,
            visibility="public",
            tags=["example", "test"]
        )

        uuid = submission["uuid"]
        result_url = submission["result"]

        print(f"\n✓ Scan submitted successfully!")
        print(f"  UUID: {uuid}")
        print(f"  Results URL: {result_url}")
        print(f"\nYou can retrieve results later using:")
        print(f"  client.get_result('{uuid}')")

        # Option 2: Submit and wait for results (uncomment to use)
        # print("\n\nAlternatively, submit and wait for results:")
        # result = client.submit_and_wait(
        #     url=test_url,
        #     visibility="public",
        #     max_wait=60,
        #     poll_interval=5
        # )
        # print(f"Scan completed! Page title: {result.get('page', {}).get('title')}")

        # Option 3: Search for existing scans
        print("\n\nSearching for existing scans of this domain:")
        search_results = client.search(query="domain:example.com", size=5)

        total = search_results.get("total", 0)
        print(f"Found {total} existing scans")

        if search_results.get("results"):
            print("\nRecent scans:")
            for i, scan in enumerate(search_results["results"][:3], 1):
                task = scan.get("task", {})
                print(f"  {i}. {task.get('url')} - {task.get('time')}")

    except URLScanError as e:
        print(f"Error: {e}")
        print("\nMake sure you have set URLSCAN_API_KEY environment variable.")
        print("Get your API key from: https://urlscan.io/user/signup")
        sys.exit(1)


if __name__ == "__main__":
    main()
