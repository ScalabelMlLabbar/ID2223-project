import sys
from pathlib import Path

import hopsworks
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phising_detection.utils.hopsworks_utils import (
    connect_to_hopsworks,
    get_or_create_feature_group,
    upload_dataframe_to_feature_group
)


def upload_legit_urlcsv_to_hopsworks():


    # Load the legitimate URLs dataset
    legit_urls_df = pd.read_csv("../src/phising_detection/data/data_files/legitimate-urls-extracted.csv",)

    # Connect to Hopsworks project
    project = connect_to_hopsworks()

    # Define feature group parameters
    feature_group_name = "scan_progress_legit_urls"
    version = 1
    description = "Feature group for legitimate URLs dataset"
    primary_key = ["domain"]
    event_time = None
    online_enabled = False

    # Upload the DataFrame to Hopsworks feature group
    upload_dataframe_to_feature_group(
        project=project,
        df=legit_urls_df,
        feature_group_name=feature_group_name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=online_enabled
    )

if __name__ == "__main__":
    upload_legit_urlcsv_to_hopsworks()