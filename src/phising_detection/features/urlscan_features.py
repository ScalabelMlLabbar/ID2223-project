"""Feature extraction from URLScan.io results."""

from typing import Dict, Any, Optional
import pandas as pd


def extract_domain_age(result: Dict[str, Any]) -> Optional[int]:
    """
    Extract domain age in days from URLScan result.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Domain age in days, or None if not available
    """
    try:
        return result.get("page", {}).get("domainAgeDays")
    except (KeyError, TypeError):
        return None


def extract_secure_percentage(result: Dict[str, Any]) -> Optional[float]:
    """
    Extract percentage of secure requests from URLScan result.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Percentage of secure requests (0-100), or None if not available
    """
    try:
        return result.get("stats", {}).get("securePercentage")
    except (KeyError, TypeError):
        return None


def extract_umbrella_rank(result: Dict[str, Any]) -> Optional[int]:
    """
    Extract Cisco Umbrella popularity rank from URLScan result.
    Lower rank = more popular/legitimate site.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Umbrella rank, or None if not available (unranked sites)
    """
    try:
        return result.get("page", {}).get("umbrellaRank")
    except (KeyError, TypeError):
        return None


def extract_tls_valid_days(result: Dict[str, Any]) -> Optional[int]:
    """
    Extract TLS certificate validity period in days from URLScan result.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Number of days the TLS certificate is valid for, or None if not available
    """
    try:
        return result.get("page", {}).get("tlsValidDays")
    except (KeyError, TypeError):
        return None


def extract_url_length(result: Dict[str, Any]) -> Optional[int]:
    """
    Extract URL length from URLScan result.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Length of the URL, or None if not available
    """
    try:
        url = result.get("task", {}).get("url")
        return len(url) if url else None
    except (KeyError, TypeError):
        return None


def extract_subdomain_count(result: Dict[str, Any]) -> Optional[int]:
    """
    Extract number of subdomains from URLScan result.
    Example: www.example.com has 1 subdomain, example.com has 0.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Number of subdomains, or None if not available
    """
    try:
        domain = result.get("page", {}).get("domain")
        if not domain:
            return None

        # Count dots and subtract 1 for TLD (e.g., example.com has 1 dot = 0 subdomains)
        # www.example.com has 2 dots = 1 subdomain
        parts = domain.split(".")
        # Assuming TLD is last part and domain is second-to-last
        # subdomain count = total parts - 2 (domain + TLD)
        subdomain_count = max(0, len(parts) - 2)
        return subdomain_count
    except (KeyError, TypeError, AttributeError):
        return None


def extract_features(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all available features from URLScan result.

    Args:
        result: URLScan.io API result dictionary

    Returns:
        Dictionary of extracted features
    """
    # Extract umbrella rank and create two features from it
    umbrella_rank = extract_umbrella_rank(result)
    has_umbrella_rank = 1 if umbrella_rank is not None else 0
    umbrella_rank_filled = umbrella_rank if umbrella_rank is not None else 999999

    # Extract TLS validity and create two features from it
    tls_valid_days = extract_tls_valid_days(result)
    has_tls = 1 if tls_valid_days is not None else 0
    tls_valid_days_filled = tls_valid_days if tls_valid_days is not None else 0

    features = {
        "domain_age_days": extract_domain_age(result),
        "secure_percentage": extract_secure_percentage(result),
        "has_umbrella_rank": has_umbrella_rank,
        "umbrella_rank": umbrella_rank_filled,
        "has_tls": has_tls,
        "tls_valid_days": tls_valid_days_filled,
        "url_length": extract_url_length(result),
        "subdomain_count": extract_subdomain_count(result),
    }

    return features


def extract_features_to_dataframe(results: list[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract features from multiple URLScan results into a DataFrame.

    Args:
        results: List of URLScan.io API result dictionaries

    Returns:
        DataFrame with extracted features
    """
    features_list = []

    for result in results:
        features = extract_features(result)
        # Add URL and UUID for reference
        # Use original_url if available (preserves submitted URL), otherwise use task URL
        features["url"] = result.get("original_url") or result.get("task", {}).get("url")
        features["scan_uuid"] = result.get("task", {}).get("uuid")
        features_list.append(features)

    return pd.DataFrame(features_list)