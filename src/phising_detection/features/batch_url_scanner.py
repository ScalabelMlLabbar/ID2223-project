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
import argparse
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
    version: int,
    attempted_fg_name: str = None,
    attempted_fg_version: int = 1
) -> set:
    """
    Retrieve URLs that have already been scanned or attempted from feature groups.

    Args:
        project: Hopsworks project object
        feature_group_name: Name of output feature group (successful scans)
        version: Feature group version
        attempted_fg_name: Name of attempted scans tracking feature group (optional)
        attempted_fg_version: Version of attempted scans feature group

    Returns:
        Set of URLs that have already been scanned or attempted (empty set if FG doesn't exist)
    """
    all_attempted_urls = set()

    # Check successful scans
    try:
        logger.info(f"Checking for successful scans in {feature_group_name} v{version}")
        existing_df = hw.read_feature_group(project, feature_group_name, version)

        if 'url' in existing_df.columns:
            scanned_urls = set(existing_df['url'].dropna().unique())
            logger.info(f"Found {len(scanned_urls)} successfully scanned URLs")
            all_attempted_urls.update(scanned_urls)
        else:
            logger.warning(f"Feature group exists but no 'url' column found")

    except Exception as e:
        logger.info(f"Output feature group not found or error reading it: {e}")

    # Check attempted scans (including failed ones)
    if attempted_fg_name:
        try:
            logger.info(f"Checking for attempted scans in {attempted_fg_name} v{attempted_fg_version}")
            attempted_df = hw.read_feature_group(project, attempted_fg_name, attempted_fg_version)

            if 'url' in attempted_df.columns:
                attempted_urls = set(attempted_df['url'].dropna().unique())
                logger.info(f"Found {len(attempted_urls)} attempted URLs (including failures)")
                all_attempted_urls.update(attempted_urls)

        except Exception as e:
            logger.info(f"Attempted scans feature group not found: {e}")

    logger.info(f"Total URLs to skip (successful + attempted): {len(all_attempted_urls)}")
    return all_attempted_urls


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


def record_attempted_scans(
    project,
    urls: List[str],
    statuses: List[str],
    uuids: List[str] = None,
    feature_group_name: str = "attempted_scans",
    version: int = 1
):
    """
    Record attempted scans (both successful and failed) to prevent re-trying.

    Args:
        project: Hopsworks project object
        urls: List of URLs that were attempted
        statuses: List of status strings ('submitted', 'success', 'failed', 'timeout')
        uuids: Optional list of scan UUIDs
        feature_group_name: Name of tracking feature group
        version: Feature group version
    """
    if not urls:
        return

    import datetime

    # Create DataFrame of attempted scans
    attempted_df = pd.DataFrame({
        'url': urls,
        'status': statuses,
        'timestamp': [datetime.datetime.now()] * len(urls)
    })

    logger.info(f"Recording {len(attempted_df)} attempted scans")

    try:
        hw.upload_dataframe_to_feature_group(
            project=project,
            df=attempted_df,
            feature_group_name=feature_group_name,
            version=version,
            description="Tracking of all attempted URL scans (successful and failed)",
            primary_key=["url"],
            online_enabled=False,
            write_options={"wait_for_job": False}  # Don't wait, just record async
        )
        logger.info(f"Recorded attempted scans to {feature_group_name}")
    except Exception as e:
        logger.warning(f"Failed to record attempted scans: {e}")


def submit_url_batch(
    client: URLScanClient,
    urls: List[str],
    visibility: str = "public",
    delay_between_submissions: float = 1.0
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Submit a batch of URLs for scanning (without waiting for results).

    Args:
        client: URLScan client instance
        urls: List of URLs to scan
        visibility: Scan visibility setting
        delay_between_submissions: Delay in seconds between submissions to respect rate limits

    Returns:
        Tuple of (submissions list, permanent_failures list)
        - submissions: List of submission dicts with 'url', 'uuid', 'api'
        - permanent_failures: List of {'url', 'error'} for non-retryable failures
    """
    submissions = []
    permanent_failures = []

    for i, url in enumerate(urls, 1):
        logger.info(f"Submitting URL {i}/{len(urls)}: {url}")

        try:
            submission = client.submit_url(url=url, visibility=visibility)
            # Add the original URL to the submission data
            submission['url'] = url
            submissions.append(submission)
            logger.info(f"Successfully submitted: {url} (UUID: {submission.get('uuid')})")

        except URLScanError as e:
            error_msg = str(e).lower()
            # Check if this is a permanent failure or temporary (rate limit)
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit hit for {url} - will retry later")
                # Don't add to permanent failures - this can be retried
            elif "bad request" in error_msg or "invalid" in error_msg:
                logger.error(f"Permanent failure for {url}: {e}")
                permanent_failures.append({'url': url, 'error': str(e)})
            else:
                logger.error(f"Failed to submit {url}: {e}")
                # Unknown error - don't record as permanent for safety
            continue

        # Rate limiting: wait between submissions
        if i < len(urls):
            time.sleep(delay_between_submissions)

    logger.info(f"Submitted {len(submissions)}/{len(urls)} URLs successfully")
    if permanent_failures:
        logger.info(f"Permanent failures: {len(permanent_failures)}")
    return submissions, permanent_failures


def retrieve_scan_results(
    client: URLScanClient,
    submissions: List[Dict[str, Any]],
    max_wait: int = 300,
    poll_interval: int = 10,
    initial_wait: int = 30
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Retrieve results for submitted scans.

    Args:
        client: URLScan client instance
        submissions: List of submission dictionaries from submit_url_batch
        max_wait: Maximum time to wait for each scan (seconds)
        poll_interval: Time between polling attempts (seconds)
        initial_wait: Time to wait before first poll attempt (seconds)

    Returns:
        Tuple of (results list, permanent_failures list)
        - results: List of scan results (successful retrievals only)
        - permanent_failures: List of {'url', 'error'} for non-retryable failures (excludes timeouts)
    """
    logger.info(f"Waiting {initial_wait} seconds for scans to complete...")
    time.sleep(initial_wait)

    results = []
    permanent_failures = []
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
                error_msg = str(e).lower()
                if "not found or not ready" in error_msg:
                    # Scan not ready yet, keep in pending list
                    still_pending.append(submission)
                elif "dns" in error_msg or "domain" in error_msg or "unreachable" in error_msg:
                    # Permanent DNS/domain failures - won't work on retry
                    logger.error(f"Permanent failure for {url} (UUID: {uuid}): {e}")
                    permanent_failures.append({'url': url, 'error': str(e)})
                else:
                    # Other error - log but don't record as permanent for safety
                    logger.error(f"Failed to retrieve result for {url} (UUID: {uuid}): {e}")

        pending_submissions = still_pending

        if pending_submissions:
            logger.info(f"Still waiting for {len(pending_submissions)} scans. Waiting {poll_interval}s...")
            time.sleep(poll_interval)

    # Timeouts are NOT permanent failures - scans might just be slow
    if pending_submissions:
        logger.warning(f"Timeout: {len(pending_submissions)} scans did not complete in time (will retry later)")
        for submission in pending_submissions:
            logger.warning(f"  - {submission.get('url')} (UUID: {submission.get('uuid')})")

    logger.info(f"Successfully retrieved {len(results)}/{len(submissions)} scan results")
    if permanent_failures:
        logger.info(f"Permanent failures: {len(permanent_failures)}")
    return results, permanent_failures


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

    # Check for already scanned URLs (including failed attempts)
    logger.info("Checking for already scanned URLs...")
    scanned_urls = get_already_scanned_urls(
        project=project,
        feature_group_name=output_fg_name,
        version=output_version,
        attempted_fg_name="attempted_scans",  # Track failed scans too
        attempted_fg_version=1
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
        submissions, submission_failures = submit_url_batch(
            client=urlscan_client,
            urls=batch_urls,
            visibility="public",
            delay_between_submissions=1.0  # 1 second between submissions
        )

        # Phase 2: Retrieve scan results
        if submissions:
            logger.info(f"Retrieving results for {len(submissions)} submitted scans...")
            scan_results, retrieval_failures = retrieve_scan_results(
                client=urlscan_client,
                submissions=submissions,
                max_wait=300,  # 5 minutes total wait time
                poll_interval=10,  # Check every 10 seconds
                initial_wait=30  # Wait 30 seconds before first check
            )
        else:
            scan_results = []
            retrieval_failures = []
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

        # Record ONLY successful scans and permanent failures (not timeouts or rate limits)
        successful_urls = {result.get('original_url') or result.get('task', {}).get('url')
                          for result in scan_results}

        attempted_urls = []
        attempted_statuses = []
        attempted_uuids = []

        # Record successful scans
        for result in scan_results:
            url = result.get('original_url') or result.get('task', {}).get('url')
            uuid = result.get('task', {}).get('uuid')
            attempted_urls.append(url)
            attempted_statuses.append('success')
            attempted_uuids.append(uuid)

        # Record permanent failures from submission (invalid URLs, etc.)
        for failure in submission_failures:
            attempted_urls.append(failure['url'])
            attempted_statuses.append('failed_permanent')
            attempted_uuids.append(None)

        # Record permanent failures from retrieval (DNS errors, etc.)
        for failure in retrieval_failures:
            attempted_urls.append(failure['url'])
            attempted_statuses.append('failed_permanent')
            attempted_uuids.append(None)

        # Only record if we have something to record
        if attempted_urls:
            record_attempted_scans(
                project=project,
                urls=attempted_urls,
                statuses=attempted_statuses,
                uuids=attempted_uuids,
                feature_group_name="attempted_scans",
                version=1
            )

        # Wait between batches to respect rate limits
        if batch_num < total_batches - 1:
            wait_time = 10
            logger.info(f"Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)

    logger.info("=" * 80)
    logger.info("Batch URL scanning pipeline completed!")
    logger.info("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch URL scanning pipeline for phishing detection"
    )

    parser.add_argument(
        "--fg1-name",
        type=str,
        default="phishing_urls",
        help="Name of first feature group (default: phishing_urls)"
    )
    parser.add_argument(
        "--fg1-version",
        type=int,
        default=2,
        help="Version of first feature group (default: 2)"
    )
    parser.add_argument(
        "--fg2-name",
        type=str,
        default="legitimate_urls",
        help="Name of second feature group (default: legitimate_urls)"
    )
    parser.add_argument(
        "--fg2-version",
        type=int,
        default=1,
        help="Version of second feature group (default: 1)"
    )
    parser.add_argument(
        "--output-fg-name",
        type=str,
        default="urlscan_features",
        help="Name of output feature group (default: urlscan_features)"
    )
    parser.add_argument(
        "--output-version",
        type=int,
        default=1,
        help="Version of output feature group (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of URLs to scan per batch (default: 200)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples from each input group (default: None = use all)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        fg1_name=args.fg1_name,
        fg1_version=args.fg1_version,
        fg2_name=args.fg2_name,
        fg2_version=args.fg2_version,
        output_fg_name=args.output_fg_name,
        output_version=args.output_version,
        batch_size=args.batch_size,
        sample_size=args.sample_size
    )