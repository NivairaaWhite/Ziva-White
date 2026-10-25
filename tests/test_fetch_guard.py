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


# ------------------------------------------------------------------ robots.txt gets the SSRF treatment too (review item)
from zivawhite.connectors import web_fetch_connector as wf  # noqa: E402

PUBLIC = "93.184.216.34"          # a public IP literal: check_url resolves literals without any network


class FakeResp:
    def __init__(self, status=200, body=b"", location=None, encoding="utf-8"):
        self.status_code, self._body, self.encoding = status, body, encoding
        self.headers = {"Location": location} if location else {}
        self.is_redirect = bool(location) and 300 <= status < 400
        self.text = body.decode("utf-8", "replace")

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise wf.requests.HTTPError(str(self.status_code))


def serve(monkeypatch, routes):
    """Route table: url -> FakeResp. Records every URL actually requested."""
    asked = []

    def fake_get(url, headers=None, timeout=None, allow_redirects=True, stream=False):
        asked.append(url)
        resp = routes.get(url, FakeResp(404))
        while allow_redirects and resp.is_redirect:           # what the real requests.get does by default
            url = wf.urljoin(url, resp.headers["Location"])
            asked.append(url)
            resp = routes.get(url, FakeResp(404))
        return resp
    monkeypatch.setattr(wf.requests, "get", fake_get)
    monkeypatch.setattr(wf.time, "sleep", lambda s: None)
    monkeypatch.delenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", raising=False)
    return asked


def test_a_robots_txt_that_redirects_to_an_internal_address_is_never_requested(monkeypatch):
    """A public site answers /robots.txt with a 302 to the cloud-metadata address. Before the fix requests followed it
    (a blind request into the internal network); now the hop is checked and refused, and the page is not fetched."""
    asked = serve(monkeypatch, {f"http://{PUBLIC}/robots.txt": FakeResp(302, location="http://169.254.169.254/latest/meta-data/")})
    result = WebFetchConnector().call(f"http://{PUBLIC}/page")
    assert not result.ok and "robots" in result.error
    assert asked == [f"http://{PUBLIC}/robots.txt"]                 # nothing else was ever requested


def test_a_robots_redirect_to_another_public_host_is_followed(monkeypatch):
    other = "93.184.216.35"
    asked = serve(monkeypatch, {
        f"http://{PUBLIC}/robots.txt": FakeResp(301, location=f"http://{other}/robots.txt"),
        f"http://{other}/robots.txt": FakeResp(200, b"User-agent: *\nAllow: /\n"),
        f"http://{PUBLIC}/page": FakeResp(200, b"<html><body>hello world</body></html>"),
    })
    result = WebFetchConnector().call(f"http://{PUBLIC}/page")
    assert result.ok and result.text == "hello world" and f"http://{other}/robots.txt" in asked


def test_a_page_that_redirects_reports_the_url_that_actually_answered(monkeypatch):
    final = "93.184.216.36"
    serve(monkeypatch, {
        f"http://{PUBLIC}/robots.txt": FakeResp(404), f"http://{final}/robots.txt": FakeResp(404),
        f"http://{PUBLIC}/short": FakeResp(302, location=f"http://{final}/article"),
        f"http://{final}/article": FakeResp(200, b"<html><body>the article</body></html>"),
    })
    result = WebFetchConnector().call(f"http://{PUBLIC}/short")
    assert result.ok and result.meta["url"] == f"http://{final}/article"
    assert result.meta["requested_url"] == f"http://{PUBLIC}/short"


def test_robots_txt_that_disallows_is_still_honoured(monkeypatch):
    serve(monkeypatch, {f"http://{PUBLIC}/robots.txt": FakeResp(200, b"User-agent: *\nDisallow: /private\n"),
                        f"http://{PUBLIC}/private/x": FakeResp(200, b"<html>secret</html>")})
    assert "robots" in WebFetchConnector().call(f"http://{PUBLIC}/private/x").error


def test_an_oversized_robots_txt_is_read_only_up_to_the_cap(monkeypatch):
    big = b"User-agent: *\nAllow: /\n" + b"# padding\n" * 200_000        # ~2 MB
    serve(monkeypatch, {f"http://{PUBLIC}/robots.txt": FakeResp(200, big),
                        f"http://{PUBLIC}/p": FakeResp(200, b"<html><body>ok</body></html>")})
    assert WebFetchConnector().call(f"http://{PUBLIC}/p").ok


# ------------------------------------------------------------------ the user agent is honest
def test_the_user_agent_names_the_bot_and_advertises_a_contact_only_if_you_set_one(monkeypatch):
    from zivawhite import __version__
    monkeypatch.delenv("ZIVAWHITE_BOT_CONTACT", raising=False)
    ua = wf.user_agent()
    assert ua.startswith(f"ZivaWhiteBot/{__version__}") and "invalid" not in ua and "+" not in ua
    monkeypatch.setenv("ZIVAWHITE_BOT_CONTACT", "https://me.example/bot")
    assert "+https://me.example/bot" in wf.user_agent()
