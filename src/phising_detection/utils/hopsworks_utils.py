"""Utilities for connecting to and interacting with Hopsworks Feature Store."""

import os
import logging
from typing import Optional
import pandas as pd
import hopsworks
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def connect_to_hopsworks(api_key: Optional[str] = None, project_name: Optional[str] = None):
    """
    Connect to Hopsworks.

    Args:
        api_key: Hopsworks API key. If None, reads from HOPSWORKS_API_KEY env variable.
        project_name: Hopsworks project name. If None, reads from HOPSWORKS_PROJECT_NAME env variable.

    Returns:
        Hopsworks project object

    Raises:
        ValueError: If API key or project name is not provided
    """
    # Get API key from parameter or environment
    api_key = api_key or os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "Hopsworks API key not provided. "
            "Set HOPSWORKS_API_KEY environment variable or pass api_key parameter."
        )

    # Get project name from parameter or environment
    project_name = project_name or os.getenv("HOPSWORKS_PROJECT")

    logger.info(f"Connecting to Hopsworks project: {project_name or 'default'}")

    # Login to Hopsworks
    try:
        if project_name:
            project = hopsworks.login(
                api_key_value=api_key,
                project=project_name,
                engine="python"  # Use Python engine (serverless, no cert download)
            )
        else:
            project = hopsworks.login(
                api_key_value=api_key,
                engine="python"
            )
    except Exception as e:
        logger.error(f"Failed to connect to Hopsworks: {e}")
        logger.info("Trying to connect without specifying project...")
        project = hopsworks.login(
            api_key_value=api_key,
            engine="python"
        )

    logger.info(f"Successfully connected to Hopsworks project: {project.name}")
    return project


def get_or_create_feature_group(
    project,
    name: str,
    version: int = 1,
    description: str = "",
    primary_key: list = None,
    event_time: Optional[str] = None,
    online_enabled: bool = False
):
    """
    Get existing feature group or create new one if it doesn't exist.

    Args:
        project: Hopsworks project object
        name: Feature group name
        version: Feature group version
        description: Description of the feature group
        primary_key: List of column names to use as primary key
        event_time: Column name to use as event time
        online_enabled: Whether to enable online feature serving

    Returns:
        Feature group object
    """
    fs = project.get_feature_store()

    try:
        # Try to get existing feature group
        fg = fs.get_feature_group(name=name, version=version)
        logger.info(f"Retrieved existing feature group: {name} (version {version})")
        return fg
    except Exception:
        # Feature group doesn't exist, will need to create it
        logger.info(f"Feature group {name} (version {version}) not found, will create on first insert")
        return None


def upload_dataframe_to_feature_group(
    project,
    df: pd.DataFrame,
    feature_group_name: str,
    version: int = 1,
    description: str = "",
    primary_key: list = None,
    event_time: Optional[str] = None,
    online_enabled: bool = False,
    write_options: dict = None
):
    """
    Upload a DataFrame to a Hopsworks feature group.

    Args:
        project: Hopsworks project object
        df: Pandas DataFrame to upload
        feature_group_name: Name of the feature group
        version: Feature group version
        description: Description of the feature group
        primary_key: List of column names to use as primary key
        event_time: Column name to use as event time
        online_enabled: Whether to enable online feature serving
        write_options: Additional write options (e.g., {"wait_for_job": False})

    Returns:
        Feature group object
    """
    fs = project.get_feature_store()

    logger.info(f"Uploading DataFrame to feature group: {feature_group_name} (version {version})")
    logger.info(f"DataFrame shape: {df.shape}")

    # Create or get feature group
    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        description=description,
        primary_key=primary_key or [],
        event_time=event_time,
        online_enabled=online_enabled
    )

    # Insert data
    write_options = write_options or {"wait_for_job": True}
    fg.insert(df, write_options=write_options)

    logger.info(f"Successfully uploaded {len(df)} rows to {feature_group_name}")
    return fg


def read_feature_group(
    project,
    feature_group_name: str,
    version: int = 1,
    online: bool = False
) -> pd.DataFrame:
    """
    Read data from a Hopsworks feature group.

    Args:
        project: Hopsworks project object
        feature_group_name: Name of the feature group
        version: Feature group version
        online: Whether to read from online feature store

    Returns:
        Pandas DataFrame with feature group data
    """
    fs = project.get_feature_store()

    logger.info(f"Reading feature group: {feature_group_name} (version {version})")

    fg = fs.get_feature_group(name=feature_group_name, version=version)

    if online:
        df = fg.read(online=True)
    else:
        df = fg.read()

    logger.info(f"Read {len(df)} rows from {feature_group_name}")
    return df


def create_feature_view(
    project,
    name: str,
    version: int = 1,
    description: str = "",
    query=None,
    labels: list = None
):
    """
    Create a feature view for training datasets.

    Args:
        project: Hopsworks project object
        name: Feature view name
        version: Feature view version
        description: Description of the feature view
        query: Query object to define feature selection
        labels: List of label column names

    Returns:
        Feature view object
    """
    fs = project.get_feature_store()

    logger.info(f"Creating feature view: {name} (version {version})")

    fv = fs.create_feature_view(
        name=name,
        version=version,
        description=description,
        query=query,
        labels=labels or []
    )

    logger.info(f"Successfully created feature view: {name}")
    return fv
