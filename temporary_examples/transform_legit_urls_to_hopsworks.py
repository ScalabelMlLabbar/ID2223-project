"""Script to retrieve legitimate URLs from Hopsworks, transform them to match phishing URL format, and upload."""

import sys
from pathlib import Path
import json
import argparse

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phising_detection.utils.hopsworks_utils import (
    connect_to_hopsworks,
    upload_dataframe_to_feature_group,
    get_or_create_feature_group
)


import hopsworks

# Connect and read feature group
project = hopsworks.login()
fs = project.get_feature_store(name='simbe200_featurestore')
fg = fs.get_feature_group('scan_progress_legit_urls', version=4)
df = fg.read()

print("Original feature group data:")
print(df.head(5))
print(f"\nDataFrame shape: {df.shape}")

# Extract all URLs from the JSON-serialized 'urls' column
all_urls = []
for _, row in df.iterrows():
    # Deserialize the JSON string to get the list of URLs
    urls_list = json.loads(row['urls'])
    all_urls.extend(urls_list)

print(f"\nTotal URLs extracted: {len(all_urls)}")

# Create new DataFrame matching phishing URL format
legit_urls_df = pd.DataFrame({
    'url_id': range(len(all_urls)),
    'url': all_urls,
    'is_phishing': 0  # 0 for legitimate URLs
})

get_or_create_feature_group(project, 'legit_urls_before_scan', version=1)
upload_dataframe_to_feature_group(
    project=project,
    df=legit_urls_df,
    feature_group_name='legit_urls_before_scan',
    version=1,
    description='Legitimate URLs formatted for phishing detection',
    primary_key=['url_id'],
    event_time=None,
    online_enabled=False
)

