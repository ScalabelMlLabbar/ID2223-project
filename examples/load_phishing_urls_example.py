"""Example script showing how to load phishing URLs into a DataFrame."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phising_detection.data import load_phishing_urls

# Load phishing URLs from the file
df = load_phishing_urls("../src/phising_detection/data/data_files/phishing-links-ACTIVE.txt")

# Display basic information
print(f"Loaded {len(df)} phishing URLs")
print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Display summary statistics
print("\nDataset info:")
print(df.info())

# Check for duplicates
duplicates = df['url'].duplicated().sum()
print(f"\nNumber of duplicate URLs: {duplicates}")

# Save to CSV if needed
# df.to_csv("data/phishing_urls.csv", index=False)
