# URLScan.io API Integration

This module provides a Python client for interacting with the URLScan.io API to analyze URLs for phishing detection.

## Setup

1. **Get an API key:**
   - Sign up at https://urlscan.io/user/signup
   - Get your API key from your account settings

2. **Set environment variable:**
   ```bash
   export URLSCAN_API_KEY="your_api_key_here"
   ```

   Or create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

## Basic Usage

### Initialize Client

```python
from phising_detection.api import URLScanClient

# Using environment variable
client = URLScanClient()

# Or pass API key directly
client = URLScanClient(api_key="your_api_key")
```

### Submit a URL for Scanning

```python
# Submit URL
result = client.submit_url(
    url="https://suspicious-site.com",
    visibility="public",  # or "unlisted" or "private"
    tags=["phishing", "test"]
)

uuid = result["uuid"]
print(f"Scan UUID: {uuid}")
```

### Retrieve Results

```python
# Get results by UUID
scan_result = client.get_result(uuid)

# Access scan data
page_title = scan_result["page"]["title"]
screenshot = scan_result["task"]["screenshotURL"]
```

### Submit and Wait for Results

```python
# Submit and automatically wait for completion
result = client.submit_and_wait(
    url="https://example.com",
    max_wait=60,  # seconds
    poll_interval=5  # seconds between checks
)
```

### Get Verdict

```python
# Get simple verdict (malicious/safe)
verdict = client.get_verdict(uuid)
print(f"Verdict: {verdict}")  # "malicious" or "safe"
```

### Search Existing Scans

```python
# Search for scans by domain
results = client.search(
    query="domain:example.com",
    size=10
)

for scan in results["results"]:
    print(scan["task"]["url"])
```

## API Response Examples

### Submission Response

```json
{
  "uuid": "abc123...",
  "result": "https://urlscan.io/result/abc123.../",
  "api": "https://urlscan.io/api/v1/result/abc123.../"
}
```

### Result Response

```json
{
  "page": {
    "url": "https://example.com",
    "title": "Example Domain",
    "status": "200"
  },
  "verdicts": {
    "overall": {
      "score": 0,
      "malicious": false
    }
  },
  "task": {
    "uuid": "abc123...",
    "time": "2024-01-01T12:00:00.000Z",
    "screenshotURL": "https://..."
  }
}
```

## Error Handling

```python
from phising_detection.api import URLScanClient, URLScanError

try:
    client = URLScanClient()
    result = client.submit_url("https://example.com")
except URLScanError as e:
    print(f"Error: {e}")
```

## Rate Limits

- Free tier: 50 submissions per day
- Paid tier: Higher limits available
- The client handles rate limit errors automatically

## Integration with Phishing Detection

```python
from phising_detection.api import URLScanClient
from phising_detection.data import load_phishing_urls

# Load your phishing URLs
df = load_phishing_urls()

# Analyze URLs
client = URLScanClient()

for idx, row in df.head(10).iterrows():  # Sample first 10
    try:
        result = client.submit_and_wait(row['url'])
        verdict = client.get_verdict(result['task']['uuid'])
        print(f"{row['url']}: {verdict}")
    except URLScanError as e:
        print(f"Error scanning {row['url']}: {e}")
```

## API Documentation

Full API documentation: https://urlscan.io/docs/api/
