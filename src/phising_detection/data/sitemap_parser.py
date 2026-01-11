"""Sitemap parsing utilities for extracting URLs from domains."""

import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Set
from urllib.parse import urljoin
import time
import logging

logger = logging.getLogger(__name__)


def get_urls_from_sitemap(
    domain: str,
    max_urls: Optional[int] = None,
    timeout: int = 10,
    max_depth: int = 2
) -> List[str]:
    """
    Extract URLs from a domain's sitemap.xml.

    Args:
        domain: Domain name (e.g., 'google.com' or 'www.google.com')
        max_urls: Maximum number of URLs to return (None for all)
        timeout: Request timeout in seconds
        max_depth: Maximum depth for nested sitemaps (sitemap index files)

    Returns:
        List of URLs found in the sitemap
    """
    urls: Set[str] = set()

    # Try common sitemap locations
    sitemap_urls = _get_sitemap_urls(domain)

    for sitemap_url in sitemap_urls:
        try:
            logger.info(f"Fetching sitemap: {sitemap_url}")
            response = requests.get(sitemap_url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; URLCollector/1.0)'
            })

            if response.status_code == 200:
                extracted = _parse_sitemap(
                    response.content,
                    max_urls - len(urls) if max_urls else None,
                    timeout,
                    max_depth
                )
                urls.update(extracted)
                logger.info(f"Found {len(extracted)} URLs from {sitemap_url}")

                if max_urls and len(urls) >= max_urls:
                    break
            else:
                logger.debug(f"Failed to fetch {sitemap_url}: {response.status_code}")

        except requests.RequestException as e:
            logger.debug(f"Error fetching {sitemap_url}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error parsing {sitemap_url}: {e}")
            continue

    result = list(urls)
    if max_urls:
        result = result[:max_urls]

    return result


def _get_sitemap_urls(domain: str) -> List[str]:
    """
    Generate possible sitemap URLs for a domain.

    Args:
        domain: Domain name

    Returns:
        List of potential sitemap URLs to try
    """
    # Remove any protocol if present
    domain = domain.replace('http://', '').replace('https://', '').rstrip('/')

    # Try both with and without www
    domains_to_try = [domain]
    if not domain.startswith('www.'):
        domains_to_try.append(f'www.{domain}')
    else:
        domains_to_try.append(domain.replace('www.', '', 1))

    sitemap_urls = []
    for d in domains_to_try:
        # Try HTTPS first, then HTTP
        sitemap_urls.extend([
            f'https://{d}/sitemap.xml',
            f'https://{d}/sitemap_index.xml',
            f'https://{d}/sitemap',
            f'http://{d}/sitemap.xml',
        ])

    return sitemap_urls


def _parse_sitemap(
    content: bytes,
    max_urls: Optional[int] = None,
    timeout: int = 10,
    max_depth: int = 2,
    current_depth: int = 0,
    max_sitemaps_per_index: int = 5
) -> Set[str]:
    """
    Parse sitemap XML content and extract URLs.

    Handles both regular sitemaps and sitemap index files.

    Args:
        content: XML content as bytes
        max_urls: Maximum URLs to extract
        timeout: Request timeout for nested sitemaps
        max_depth: Maximum recursion depth for sitemap indexes
        current_depth: Current recursion depth

    Returns:
        Set of URLs found in the sitemap
    """
    urls: Set[str] = set()

    try:
        root = ET.fromstring(content)

        # Define XML namespaces
        namespaces = {
            'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'image': 'http://www.google.com/schemas/sitemap-image/1.1',
            'news': 'http://www.google.com/schemas/sitemap-news/0.9'
        }

        # Check if this is a sitemap index (contains references to other sitemaps)
        sitemap_refs = root.findall('.//sm:sitemap/sm:loc', namespaces)

        if sitemap_refs and current_depth < max_depth:
            # This is a sitemap index - fetch referenced sitemaps
            logger.info(f"Found sitemap index with {len(sitemap_refs)} sitemaps")
            
            if len(sitemap_refs) > max_sitemaps_per_index:
                logger.info(f"Limiting sitemaps to {max_sitemaps_per_index} sitemaps of {len(sitemap_refs)} possible")

            for index, sitemap_loc in enumerate(sitemap_refs):
                if index >= max_sitemaps_per_index:
                    break

                if max_urls and len(urls) >= max_urls:
                    return urls

                sitemap_url = sitemap_loc.text
                if sitemap_url:
                    try:
                        response = requests.get(sitemap_url, timeout=timeout, headers={
                            'User-Agent': 'Mozilla/5.0 (compatible; URLCollector/1.0)'
                        })
                        # For nested sitemaps
                        if response.status_code == 200:
                            nested_urls = _parse_sitemap(
                                response.content,
                                max_urls - len(urls) if max_urls else None,
                                timeout,
                                max_depth,
                                current_depth + 1,
                                max_sitemaps_per_index
                            )
                            urls.update(nested_urls)
                            time.sleep(0.1)  # Small delay to be polite
                    except Exception as e:
                        logger.debug(f"Error fetching nested sitemap {sitemap_url}: {e}")
                        continue

        # Extract regular URL entries
        url_entries = root.findall('.//sm:url/sm:loc', namespaces)

        for loc in url_entries:
            if max_urls and len(urls) >= max_urls:
                break
            if loc.text:
                urls.add(loc.text)

    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
    except Exception as e:
        logger.warning(f"Error parsing sitemap: {e}")

    return urls


def extract_urls_from_domains(
    domains: List[str],
    max_urls_per_domain: int = 10,
    timeout: int = 10,
    delay_between_domains: float = 0.5
) -> dict:
    """
    Extract URLs from multiple domains using their sitemaps.

    Args:
        domains: List of domain names
        max_urls_per_domain: Maximum URLs to extract per domain
        timeout: Request timeout in seconds
        delay_between_domains: Delay in seconds between domain requests

    Returns:
        Dictionary mapping domain to list of extracted URLs
    """
    results = {}

    for i, domain in enumerate(domains):
        logger.info(f"Processing domain {i+1}/{len(domains)}: {domain}")

        urls = get_urls_from_sitemap(
            domain,
            max_urls=max_urls_per_domain,
            timeout=timeout
        )

        results[domain] = urls

        if i < len(domains) - 1:  # Don't sleep after the last domain
            time.sleep(delay_between_domains)

    return results