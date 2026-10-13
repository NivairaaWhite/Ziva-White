"""SSRF guard and allow-list matching. No network: only IP literals and malformed URLs are used."""
import pytest

from zivawhite.connectors.web_fetch_connector import WebFetchConnector, check_url
from zivawhite.data.sources import is_allow_listed


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost:8000/v1/health",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "http://10.0.0.5/admin",
    "http://192.168.1.1/",
    "http://[::1]/",
    "http://0.0.0.0/",
])
def test_private_and_loopback_addresses_are_refused(url, monkeypatch):
    monkeypatch.delenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", raising=False)
    assert check_url(url) is not None


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "javascript:alert(1)", "http://"])
def test_non_http_urls_are_refused(url):
    assert check_url(url) is not None


def test_public_ip_literal_is_allowed(monkeypatch):
    monkeypatch.delenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", raising=False)
    assert check_url("https://93.184.216.34/") is None


def test_private_fetch_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", "1")
    assert check_url("http://192.168.1.1/") is None


def test_fetch_connector_reports_a_refusal_instead_of_fetching(monkeypatch):
    monkeypatch.delenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", raising=False)
    result = WebFetchConnector().call("http://169.254.169.254/latest/meta-data/")
    assert not result.ok and "non-public" in result.error


def test_allow_list_matches_on_hostname_not_netloc():
    assert is_allow_listed("https://www.gutenberg.org/files/1/1.txt")
    assert is_allow_listed("https://www.gutenberg.org:443/files/1/1.txt")   # explicit port was refused before
    assert not is_allow_listed("https://gutenberg.org.evil.example/x")      # lookalike suffix
    assert not is_allow_listed("https://evil.example/gutenberg.org")
    assert not is_allow_listed("not a url")
