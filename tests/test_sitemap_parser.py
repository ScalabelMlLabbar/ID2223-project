"""Tests for sitemap parser module."""

import pytest
import responses
from phising_detection.data.sitemap_parser import (
    get_urls_from_sitemap,
    _get_sitemap_urls,
    _parse_sitemap,
    extract_urls_from_domains
)


class TestGetSitemapUrls:
    """Tests for _get_sitemap_urls function."""

    def test_basic_domain(self):
        """Test sitemap URL generation for basic domain."""
        urls = _get_sitemap_urls("example.com")

        assert "https://example.com/sitemap.xml" in urls
        assert "https://example.com/sitemap_index.xml" in urls
        assert "https://example.com/sitemap" in urls
        assert "http://example.com/sitemap.xml" in urls

    def test_domain_with_www(self):
        """Test sitemap URL generation for domain with www."""
        urls = _get_sitemap_urls("www.example.com")

        # Should try both with and without www
        assert any("www.example.com" in url for url in urls)
        assert any("example.com/sitemap" in url and "www" not in url for url in urls)

    def test_domain_without_www(self):
        """Test sitemap URL generation adds www variant."""
        urls = _get_sitemap_urls("example.com")

        assert any("www.example.com" in url for url in urls)


class TestParseSitemap:
    """Tests for _parse_sitemap function."""

    def test_parse_simple_sitemap(self):
        """Test parsing a simple sitemap with URLs."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/page1</loc>
    </url>
    <url>
        <loc>https://example.com/page2</loc>
    </url>
    <url>
        <loc>https://example.com/page3</loc>
    </url>
</urlset>"""

        urls = _parse_sitemap(sitemap_xml)

        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls

    def test_parse_empty_sitemap(self):
        """Test parsing an empty sitemap."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>"""

        urls = _parse_sitemap(sitemap_xml)
        assert len(urls) == 0

    def test_parse_sitemap_with_max_urls(self):
        """Test parsing sitemap with max_urls limit."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page1</loc></url>
    <url><loc>https://example.com/page2</loc></url>
    <url><loc>https://example.com/page3</loc></url>
    <url><loc>https://example.com/page4</loc></url>
    <url><loc>https://example.com/page5</loc></url>
</urlset>"""

        urls = _parse_sitemap(sitemap_xml, max_urls=3)
        assert len(urls) == 3

    def test_parse_invalid_xml(self):
        """Test parsing invalid XML returns empty set."""
        invalid_xml = b"This is not XML"

        urls = _parse_sitemap(invalid_xml)
        assert len(urls) == 0

    @responses.activate
    def test_parse_sitemap_index(self):
        """Test parsing a sitemap index that references other sitemaps."""
        sitemap_index_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap>
        <loc>https://example.com/sitemap1.xml</loc>
    </sitemap>
    <sitemap>
        <loc>https://example.com/sitemap2.xml</loc>
    </sitemap>
</sitemapindex>"""

        sitemap1_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page1</loc></url>
    <url><loc>https://example.com/page2</loc></url>
</urlset>"""

        sitemap2_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page3</loc></url>
</urlset>"""

        # Mock the nested sitemap requests
        responses.add(
            responses.GET,
            "https://example.com/sitemap1.xml",
            body=sitemap1_xml,
            status=200
        )
        responses.add(
            responses.GET,
            "https://example.com/sitemap2.xml",
            body=sitemap2_xml,
            status=200
        )

        urls = _parse_sitemap(sitemap_index_xml, max_depth=2)

        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls


class TestGetUrlsFromSitemap:
    """Tests for get_urls_from_sitemap function."""

    @responses.activate
    def test_successful_sitemap_fetch(self):
        """Test successful sitemap fetching."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page1</loc></url>
    <url><loc>https://example.com/page2</loc></url>
</urlset>"""

        # Mock the sitemap request
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=sitemap_xml,
            status=200
        )

        urls = get_urls_from_sitemap("example.com", timeout=5)

        assert len(urls) >= 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    @responses.activate
    def test_sitemap_not_found(self):
        """Test handling when sitemap is not found."""
        # Mock 404 responses for all sitemap URLs
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://example.com/sitemap_index.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://example.com/sitemap",
            status=404
        )
        responses.add(
            responses.GET,
            "http://example.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap_index.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap",
            status=404
        )
        responses.add(
            responses.GET,
            "http://www.example.com/sitemap.xml",
            status=404
        )

        urls = get_urls_from_sitemap("example.com", timeout=5)

        assert len(urls) == 0

    @responses.activate
    def test_max_urls_limit(self):
        """Test that max_urls limit is respected."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page1</loc></url>
    <url><loc>https://example.com/page2</loc></url>
    <url><loc>https://example.com/page3</loc></url>
    <url><loc>https://example.com/page4</loc></url>
    <url><loc>https://example.com/page5</loc></url>
</urlset>"""

        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=sitemap_xml,
            status=200
        )

        urls = get_urls_from_sitemap("example.com", max_urls=3, timeout=5)

        assert len(urls) <= 3

    @responses.activate
    def test_timeout_handling(self):
        """Test that timeout is handled gracefully."""
        from requests.exceptions import Timeout

        # Mock timeout
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "https://example.com/sitemap_index.xml",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "https://example.com/sitemap",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "http://example.com/sitemap.xml",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap.xml",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap_index.xml",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "https://www.example.com/sitemap",
            body=Timeout()
        )
        responses.add(
            responses.GET,
            "http://www.example.com/sitemap.xml",
            body=Timeout()
        )

        # Should not raise exception, just return empty list
        urls = get_urls_from_sitemap("example.com", timeout=1)
        assert len(urls) == 0


class TestExtractUrlsFromDomains:
    """Tests for extract_urls_from_domains function."""

    @responses.activate
    def test_extract_from_multiple_domains(self):
        """Test extracting URLs from multiple domains."""
        sitemap1_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example1.com/page1</loc></url>
    <url><loc>https://example1.com/page2</loc></url>
</urlset>"""

        sitemap2_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example2.com/pageA</loc></url>
</urlset>"""

        # Mock sitemaps for both domains
        responses.add(
            responses.GET,
            "https://example1.com/sitemap.xml",
            body=sitemap1_xml,
            status=200
        )
        responses.add(
            responses.GET,
            "https://example2.com/sitemap.xml",
            body=sitemap2_xml,
            status=200
        )

        domains = ["example1.com", "example2.com"]
        results = extract_urls_from_domains(
            domains,
            max_urls_per_domain=10,
            timeout=5,
            delay_between_domains=0
        )

        assert "example1.com" in results
        assert "example2.com" in results
        assert len(results["example1.com"]) == 2
        assert len(results["example2.com"]) == 1
        assert "https://example1.com/page1" in results["example1.com"]
        assert "https://example2.com/pageA" in results["example2.com"]

    @responses.activate
    def test_extract_with_failures(self):
        """Test extraction continues even if some domains fail."""
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example1.com/page1</loc></url>
</urlset>"""

        # Mock success for first domain
        responses.add(
            responses.GET,
            "https://example1.com/sitemap.xml",
            body=sitemap_xml,
            status=200
        )

        # Mock failures for second domain
        responses.add(
            responses.GET,
            "https://example2.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://example2.com/sitemap_index.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://example2.com/sitemap",
            status=404
        )
        responses.add(
            responses.GET,
            "http://example2.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example2.com/sitemap.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example2.com/sitemap_index.xml",
            status=404
        )
        responses.add(
            responses.GET,
            "https://www.example2.com/sitemap",
            status=404
        )
        responses.add(
            responses.GET,
            "http://www.example2.com/sitemap.xml",
            status=404
        )

        domains = ["example1.com", "example2.com"]
        results = extract_urls_from_domains(
            domains,
            max_urls_per_domain=10,
            timeout=5,
            delay_between_domains=0
        )

        # Should have results for both, but example2 should be empty
        assert "example1.com" in results
        assert "example2.com" in results
        assert len(results["example1.com"]) == 1
        assert len(results["example2.com"]) == 0
