"""
Batch URL scanning script that:
1. Loads 2 feature groups from Hopsworks (phishing and legitimate URLs)
2. Creates balanced dataset with equal amounts from both
3. Scans URLs in batches of 200 with URLScan
4. Extracts features from scan results
5. Uploads results to Hopsworks after each batch
6. Repeats until all URLs are scanned
"""

import sys
import os
import logging
import time
from typing import List, Dict, Any
import pandas as pd

# Add src folder to path
src_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(src_folder)

from api.urlscan import URLScanClient, URLScanError
from features.urlscan_features import extract_features_to_dataframe
from utils import hopsworks_utils as hw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_balance_feature_groups(
    project,
    fg1_name: str,
    fg1_version: int,
    fg2_name: str,
    fg2_version: int,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Load two feature groups and create balanced dataset with equal samples.

    Args:
        project: Hopsworks project object
        fg1_name: Name of first feature group (e.g., phishing URLs)
        fg1_version: Version of first feature group
        fg2_name: Name of second feature group (e.g., legitimate URLs)
        fg2_version: Version of second feature group
        sample_size: Number of samples from each group (if None, uses minimum)

    Returns:
        Balanced DataFrame with equal samples from both groups
    """
    logger.info(f"Loading feature group: {fg1_name} v{fg1_version}")
    df1 = hw.read_feature_group(project, fg1_name, fg1_version)

    logger.info(f"Loading feature group: {fg2_name} v{fg2_version}")
    df2 = hw.read_feature_group(project, fg2_name, fg2_version)

    logger.info(f"Feature group 1 size: {len(df1)}")
    logger.info(f"Feature group 2 size: {len(df2)}")

    # Determine sample size
    if sample_size is None:
        sample_size = min(len(df1), len(df2))
    else:
        sample_size = min(sample_size, len(df1), len(df2))

    logger.info(f"Sampling {sample_size} records from each feature group")

    # Sample equal amounts from each, important this randomness can affect performens of network, upsameling would be better if we had the resources.
    df1_sample = df1.sample(n=sample_size + int(0.33*sample_size), random_state=42) #to acount for offline pages in phising dataset
    df2_sample = df2.sample(n=sample_size, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([df1_sample, df2_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Created balanced dataset with {len(balanced_df)} total URLs")
    return balanced_df


def get_already_scanned_urls(
    project,
    feature_group_name: str,
    version: int
) -> set:
    """
    Retrieve URLs that have already been scanned from output feature group.

    Args:
        project: Hopsworks project object
        feature_group_name: Name of output feature group
        version: Feature group version

    Returns:
        Set of URLs that have already been scanned (empty set if FG doesn't exist)
    """
    try:
        logger.info(f"Checking for existing scans in {feature_group_name} v{version}")
        existing_df = hw.read_feature_group(project, feature_group_name, version)

        if 'url' in existing_df.columns:
            scanned_urls = set(existing_df['url'].dropna().unique())
            logger.info(f"Found {len(scanned_urls)} already scanned URLs")
            return scanned_urls
        else:
            logger.warning(f"Feature group exists but no 'url' column found")
            return set()

    except Exception as e:
        logger.info(f"Output feature group not found or error reading it: {e}")
        logger.info("Will scan all URLs")
        return set()


def filter_already_scanned(
    df: pd.DataFrame,
    scanned_urls: set,
    url_column: str = None
) -> pd.DataFrame:
    """
    Filter out URLs that have already been scanned.

    Args:
        df: DataFrame with URLs to scan
        scanned_urls: Set of already scanned URLs
        url_column: Name of URL column (auto-detected if None)

    Returns:
        Filtered DataFrame with only unscanned URLs
    """
    if not scanned_urls:
        logger.info("No previously scanned URLs to filter")
        return df

    # Auto-detect URL column
    if url_column is None:
        url_column = 'phishing_url' if 'phishing_url' in df.columns else 'url'

    original_count = len(df)
    filtered_df = df[~df[url_column].isin(scanned_urls)].reset_index(drop=True)
    filtered_count = len(filtered_df)
    skipped_count = original_count - filtered_count

    logger.info(f"Filtered out {skipped_count} already scanned URLs")
    logger.info(f"Remaining URLs to scan: {filtered_count}")

    return filtered_df


def submit_url_batch(
    client: URLScanClient,
    urls: List[str],
    visibility: str = "public",
    delay_between_submissions: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Submit a batch of URLs for scanning (without waiting for results).

    Args:
        client: URLScan client instance
        urls: List of URLs to scan
        visibility: Scan visibility setting
        delay_between_submissions: Delay in seconds between submissions to respect rate limits

    Returns:
        List of submission dictionaries with 'url', 'uuid', and 'api' fields
    """
    submissions = []

    for i, url in enumerate(urls, 1):
        logger.info(f"Submitting URL {i}/{len(urls)}: {url}")

        try:
            submission = client.submit_url(url=url, visibility=visibility)
            # Add the original URL to the submission data
            submission['url'] = url
            submissions.append(submission)
            logger.info(f"Successfully submitted: {url} (UUID: {submission.get('uuid')})")

        except URLScanError as e:
            logger.error(f"Failed to submit {url}: {e}")
            # Continue with next URL
            continue

        # Rate limiting: wait between submissions
        if i < len(urls):
            time.sleep(delay_between_submissions)

    logger.info(f"Submitted {len(submissions)}/{len(urls)} URLs successfully")
    return submissions


def retrieve_scan_results(
    client: URLScanClient,
    submissions: List[Dict[str, Any]],
    max_wait: int = 300,
    poll_interval: int = 10,
    initial_wait: int = 30
) -> List[Dict[str, Any]]:
    """
    Retrieve results for submitted scans.

    Args:
        client: URLScan client instance
        submissions: List of submission dictionaries from submit_url_batch
        max_wait: Maximum time to wait for each scan (seconds)
        poll_interval: Time between polling attempts (seconds)
        initial_wait: Time to wait before first poll attempt (seconds)

    Returns:
        List of scan results (successful retrievals only)
    """
    logger.info(f"Waiting {initial_wait} seconds for scans to complete...")
    time.sleep(initial_wait)

    results = []
    pending_submissions = submissions.copy()

    start_time = time.time()

    while pending_submissions and (time.time() - start_time) < max_wait:
        still_pending = []

        for submission in pending_submissions:
            uuid = submission.get('uuid')
            url = submission.get('url')

            try:
                result = client.get_result(uuid)
                # Preserve the original submitted URL for proper matching later
                result['original_url'] = url
                results.append(result)
                logger.info(f"Retrieved result for {url} (UUID: {uuid})")

            except URLScanError as e:
                if "not found or not ready" in str(e):
                    # Scan not ready yet, keep in pending list
                    still_pending.append(submission)
                else:
                    # Other error, log and skip
                    logger.error(f"Failed to retrieve result for {url} (UUID: {uuid}): {e}")

        pending_submissions = still_pending

        if pending_submissions:
            logger.info(f"Still waiting for {len(pending_submissions)} scans. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

    if pending_submissions:
        logger.warning(f"Timeout: {len(pending_submissions)} scans did not complete in time")
        for submission in pending_submissions:
            logger.warning(f"  - {submission.get('url')} (UUID: {submission.get('uuid')})")

    logger.info(f"Successfully retrieved {len(results)}/{len(submissions)} scan results")
    return results


def process_and_upload_batch(
    project,
    scan_results: List[Dict[str, Any]],
    original_df: pd.DataFrame,
    feature_group_name: str,
    version: int,
    primary_key: List[str]
):
    """
    Extract features from scan results and upload to Hopsworks.

    Args:
        project: Hopsworks project object
        scan_results: List of URLScan result dictionaries
        original_df: Original DataFrame with URL metadata (is_phishing, etc.)
        feature_group_name: Name of output feature group
        version: Feature group version
        primary_key: Primary key columns for feature group
    """
    if not scan_results:
        logger.warning("No scan results to process")
        return

    logger.info(f"Extracting features from {len(scan_results)} scan results")
    features_df = extract_features_to_dataframe(scan_results)

    # Merge with original data to get labels (is_phishing)
    # Assuming original_df has 'url' or 'phishing_url' column
    url_col = 'phishing_url' if 'phishing_url' in original_df.columns else 'url'

    # Merge on URL to add is_phishing label
    features_df = features_df.merge(
        original_df[[url_col, 'is_phishing']],
        left_on='url',
        right_on=url_col,
        how='left'
    )

    # Drop duplicate url column if exists
    if url_col != 'url' and url_col in features_df.columns:
        features_df = features_df.drop(columns=[url_col])

    # Check for NaN values in is_phishing and log warnings
    nan_count = features_df['is_phishing'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count}/{len(features_df)} records with NaN is_phishing values")
        logger.warning("This indicates URL mismatch between submitted and retrieved URLs")
        # Show some examples of URLs that didn't match
        nan_urls = features_df[features_df['is_phishing'].isna()]['url'].head(5).tolist()
        logger.warning(f"Example URLs with no match: {nan_urls}")

    # Drop rows with NaN is_phishing to avoid data quality issues
    before_drop = len(features_df)
    features_df = features_df.dropna(subset=['is_phishing'])
    after_drop = len(features_df)

    if before_drop != after_drop:
        logger.warning(f"Dropped {before_drop - after_drop} rows with missing is_phishing labels")

    if len(features_df) == 0:
        logger.error("No valid records to upload after dropping NaN values")
        return

    logger.info(f"Uploading {len(features_df)} records to Hopsworks")

    hw.upload_dataframe_to_feature_group(
        project=project,
        df=features_df,
        feature_group_name=feature_group_name,
        version=version,
        description="URLScan features extracted from phishing and legitimate URLs",
        primary_key=primary_key,
        online_enabled=True,
        write_options={"wait_for_job": True}
    )

    logger.info("Successfully uploaded batch to Hopsworks")


def main(
    fg1_name: str = "phishing_urls",
    fg1_version: int = 2,
    fg2_name: str = "legitimate_urls",
    fg2_version: int = 1,
    output_fg_name: str = "urlscan_features",
    output_version: int = 1,
    batch_size: int = 200,
    sample_size: int = None
):
    """
    Main orchestration function.

    Args:
        fg1_name: Name of first feature group
        fg1_version: Version of first feature group
        fg2_name: Name of second feature group
        fg2_version: Version of second feature group
        output_fg_name: Name of output feature group
        output_version: Version of output feature group
        batch_size: Number of URLs to scan per batch
        sample_size: Number of samples from each input group (None = all)
    """
    logger.info("=" * 80)
    logger.info("Starting batch URL scanning pipeline")
    logger.info("=" * 80)

    # Connect to Hopsworks
    logger.info("Connecting to Hopsworks...")
    project = hw.connect_to_hopsworks()

    # Initialize URLScan client
    logger.info("Initializing URLScan client...")
    urlscan_client = URLScanClient()

    # Load and balance feature groups
    logger.info("Loading and balancing feature groups...")
    balanced_df = load_and_balance_feature_groups(
        project=project,
        fg1_name=fg1_name,
        fg1_version=fg1_version,
        fg2_name=fg2_name,
        fg2_version=fg2_version,
        sample_size=sample_size
    )

    # Check for already scanned URLs
    logger.info("Checking for already scanned URLs...")
    scanned_urls = get_already_scanned_urls(
        project=project,
        feature_group_name=output_fg_name,
        version=output_version
    )

    # Filter out already scanned URLs
    balanced_df = filter_already_scanned(
        df=balanced_df,
        scanned_urls=scanned_urls
    )

    # Check if there are any URLs left to scan
    if len(balanced_df) == 0:
        logger.info("All URLs have already been scanned. Nothing to do!")
        return

    # Determine URL column name
    url_col = 'phishing_url' if 'phishing_url' in balanced_df.columns else 'url'
    all_urls = balanced_df[url_col].tolist()

    total_urls = len(all_urls)
    total_batches = (total_urls + batch_size - 1) // batch_size

    logger.info(f"Total URLs to scan: {total_urls}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total batches: {total_batches}")

    # Process in batches
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_urls)

        logger.info("=" * 80)
        logger.info(f"Processing batch {batch_num + 1}/{total_batches}")
        logger.info(f"URLs {start_idx + 1} to {end_idx} of {total_urls}")
        logger.info("=" * 80)

        # Get batch of URLs
        batch_urls = all_urls[start_idx:end_idx]
        batch_df = balanced_df.iloc[start_idx:end_idx]

        # Phase 1: Submit all URLs for scanning
        logger.info(f"Submitting {len(batch_urls)} URLs for scanning...")
        submissions = submit_url_batch(
            client=urlscan_client,
            urls=batch_urls,
            visibility="public",
            delay_between_submissions=1.0  # 1 second between submissions
        )

        # Phase 2: Retrieve scan results
        if submissions:
            logger.info(f"Retrieving results for {len(submissions)} submitted scans...")
            scan_results = retrieve_scan_results(
                client=urlscan_client,
                submissions=submissions,
                max_wait=300,  # 5 minutes total wait time
                poll_interval=10,  # Check every 10 seconds
                initial_wait=30  # Wait 30 seconds before first check
            )
        else:
            scan_results = []
            logger.warning("No URLs were successfully submitted")

        # Process and upload results
        if scan_results:
            process_and_upload_batch(
                project=project,
                scan_results=scan_results,
                original_df=batch_df,
                feature_group_name=output_fg_name,
                version=output_version,
                primary_key=["scan_uuid"]
            )
        else:
            logger.warning(f"No successful scans in batch {batch_num + 1}, skipping upload")

        # Wait between batches to respect rate limits
        if batch_num < total_batches - 1:
            wait_time = 10
            logger.info(f"Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)

    logger.info("=" * 80)
    logger.info("Batch URL scanning pipeline completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Example usage - adjust parameters as needed
    main(
        fg1_name="phishing_urls",
        fg1_version=2,
        fg2_name="legit_urls_before_scan",
        fg2_version=1,
        output_fg_name="urlscan_features",
        output_version=1,
        batch_size=300,
        sample_size=None  # Set to None to use all available data
    )