"""Example script to extract URLs from legitimate domains using sitemaps."""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from phising_detection.data.sitemap_parser import get_urls_from_sitemap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_already_processed_domains(csv_file: Path) -> set:
    """
    Get set of domains that have already been processed.

    Args:
        csv_file: Path to the CSV file with processed results

    Returns:
        Set of domain names that have been processed
    """
    if not csv_file.exists():
        return set()

    try:
        df = pd.read_csv(csv_file)
        if 'domain' in df.columns:
            return set(df['domain'].unique())
    except Exception as e:
        logger.warning(f"Error reading existing CSV: {e}")

    return set()


def save_batch_to_csv(batch_data: list, csv_file: Path, mode: str = 'a'):
    """
    Save a batch of domain data to CSV.

    Args:
        batch_data: List of dictionaries with keys: domain, urls, is_phishing
        csv_file: Path to CSV file
        mode: File mode ('w' for write/overwrite, 'a' for append)
    """
    if not batch_data:
        return

    df = pd.DataFrame(batch_data)

    # Convert URL lists to JSON strings for CSV storage
    import json
    df['urls'] = df['urls'].apply(json.dumps)

    # Write header only if creating new file
    header = not csv_file.exists() or mode == 'w'

    df.to_csv(csv_file, mode=mode, header=header, index=False)
    logger.info(f"Saved batch of {len(batch_data)} domains to {csv_file}")


def main():
    """Extract URLs from legitimate domains and save to CSV incrementally."""

    # Path to legitimate domains file
    domains_file = Path(__file__).parent.parent / "src" / "phising_detection" / "data" / "data_files" / "legitimate-urls.txt"

    # Output CSV file for extracted URLs
    output_csv = Path(__file__).parent.parent / "src" / "phising_detection" / "data" / "data_files" / "legitimate-urls-extracted.csv"

    # Read domains from file
    logger.info(f"Reading domains from {domains_file}")
    with open(domains_file, 'r', encoding='utf-8') as f:
        all_domains = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(all_domains)} total domains")

    # Check which domains have already been processed
    processed_domains = get_already_processed_domains(output_csv)
    logger.info(f"Already processed: {len(processed_domains)} domains")

    # Filter out already processed domains
    domains_to_process = [d for d in all_domains if d not in processed_domains]
    logger.info(f"Remaining to process: {len(domains_to_process)} domains")

    if not domains_to_process:
        logger.info("All domains have already been processed!")
        return

    # Configuration
    max_urls_per_domain = 10
    batch_size = 50  # Save every 50 domains
    timeout = 10
    delay_between_domains = 0.5

    # Process domains in batches
    total_processed = 0
    total_urls_extracted = 0
    domains_with_urls = 0
    batch_data = []

    for i, domain in enumerate(domains_to_process):
        logger.info(f"Processing {i+1}/{len(domains_to_process)}: {domain}")

        try:
            urls = get_urls_from_sitemap(
                domain,
                max_urls=max_urls_per_domain,
                timeout=timeout
            )

            logger.info(f"  Found {len(urls)} URLs from {domain}")

            # Add domain to batch data (even if no URLs found)
            batch_data.append({
                'domain': domain,
                'urls': urls,  # Store as list (will be converted to JSON in save function)
                'is_phishing': 0
            })

            total_urls_extracted += len(urls)
            if urls:
                domains_with_urls += 1
            total_processed += 1

            # Save batch every N domains
            if (i + 1) % batch_size == 0:
                save_batch_to_csv(batch_data, output_csv, mode='a')
                logger.info(f"Checkpoint: Processed {total_processed} domains, {domains_with_urls} with URLs, extracted {total_urls_extracted} total URLs")
                batch_data = []

            # Small delay to be polite
            import time
            time.sleep(delay_between_domains)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Saving current batch...")
            if batch_data:
                save_batch_to_csv(batch_data, output_csv, mode='a')
            logger.info(f"Saved progress. Processed {total_processed} domains so far.")
            return
        except Exception as e:
            logger.error(f"Error processing {domain}: {e}")
            # Still save the domain with empty URL list
            batch_data.append({
                'domain': domain,
                'urls': [],
                'is_phishing': 0
            })
            total_processed += 1
            continue

    # Save any remaining data
    if batch_data:
        save_batch_to_csv(batch_data, output_csv, mode='a')

    # Print final summary
    logger.info("\n=== Final Summary ===")
    logger.info(f"Domains processed this run: {total_processed}")
    logger.info(f"Domains with URLs this run: {domains_with_urls}")
    logger.info(f"Total URLs extracted this run: {total_urls_extracted}")
    if total_processed > 0:
        logger.info(f"Average URLs per domain: {total_urls_extracted / total_processed:.1f}")

    # Show overall statistics from CSV
    if output_csv.exists():
        import json
        df = pd.read_csv(output_csv)

        # Parse URL lists from JSON
        df['urls_parsed'] = df['urls'].apply(json.loads)
        df['url_count'] = df['urls_parsed'].apply(len)

        total_urls = df['url_count'].sum()
        domains_with_urls_total = (df['url_count'] > 0).sum()

        logger.info(f"\nOverall statistics:")
        logger.info(f"Total domains processed: {len(df)}")
        logger.info(f"Domains with URLs: {domains_with_urls_total}")
        logger.info(f"Domains without URLs: {len(df) - domains_with_urls_total}")
        logger.info(f"Total URLs: {total_urls}")
        logger.info(f"Average URLs per domain (with URLs): {total_urls / domains_with_urls_total:.1f}" if domains_with_urls_total > 0 else "Average URLs per domain: 0")

        logger.info(f"\nSample domains and their URLs:")
        for _, row in df.head(10).iterrows():
            url_list = json.loads(row['urls'])
            url_count = len(url_list)
            if url_count > 0:
                logger.info(f"  {row['domain']}: {url_count} URLs")
                for url in url_list[:3]:  # Show first 3 URLs
                    logger.info(f"    - {url}")
                if url_count > 3:
                    logger.info(f"    ... and {url_count - 3} more")
            else:
                logger.info(f"  {row['domain']}: No URLs found")


if __name__ == "__main__":
    main()