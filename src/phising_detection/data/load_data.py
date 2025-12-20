"""Data loading utilities for phishing detection."""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_phishing_urls(
    file_path: Union[str, Path] = None,
    is_phishing: bool = True,
) -> pd.DataFrame:
    """
    Load phishing URLs from a text file into a DataFrame.

    Args:
        file_path: Path to the file containing phishing URLs (one per line).
                   If None, uses the default phishing-links-ACTIVE.txt in the data directory.
        is_phishing: Whether to add a label column indicating phishing (default True)

    Returns:
        DataFrame with columns: url_id, url, and optionally is_phishing
    """
    if file_path is None:
        # Default to the file in the same directory as this module
        file_path = Path(__file__).parent / "phishing-links-ACTIVE.txt"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read URLs from file
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Create DataFrame
    df = pd.DataFrame({
        'url_id': range(len(urls)),
        'url': urls,
        'is_phishing': int(is_phishing)
    })



    return df


