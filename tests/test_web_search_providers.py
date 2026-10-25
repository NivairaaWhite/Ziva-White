"""Each search provider has its own request shape, auth and response shape (review item): drive every adapter with a
fake HTTP layer, and check the API key never leaks into an error message."""
import pytest
import requests

from zivawhite.connectors import web_search_connector as ws
from zivawhite.connectors.web_search_connector import WebSearchConnector, detect_provider, scrub

KEY = "sk-live-0123456789abcdef"


class Resp:
    def __init__(self, data=None, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Client Error for url: https://x/?key={KEY}")   # real requests embeds the URL
            err.response = self
            raise err

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def patch(monkeypatch, resp):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params or {}, headers=headers or {})
        if isinstance(resp, Exception):
            raise resp
        return resp
    monkeypatch.setattr(ws.requests, "get", fake_get)
    return seen


def conn(provider, **kw):
    return WebSearchConnector(searxng_url="", provider=provider, api_key=KEY, **kw)


@pytest.mark.parametrize("url,expected", [
    ("https://api.search.brave.com/res/v1/web/search", "brave"), ("https://serpapi.com/search.json", "serpapi"),
    ("https://www.googleapis.com/customsearch/v1", "google_cse"), ("https://my-proxy.example/search", "generic"), ("", "generic")])
def test_provider_is_recognised_from_the_url(url, expected):
    assert detect_provider(url) == expected
    assert detect_provider(url, "serpapi") == "serpapi" and detect_provider(url, "nonsense") == "generic"


def test_brave_sends_the_key_in_a_header_not_the_url_and_reads_web_results(monkeypatch):
    seen = patch(monkeypatch, Resp({"web": {"results": [{"title": "T", "description": "D", "url": "https://a.example"}]}}))
    r = conn("brave").call("hello", max_results=50)
    assert r.ok and "T: D  [https://a.example]" in r.text and r.meta["urls"] == ["https://a.example"]
    assert seen["headers"]["X-Subscription-Token"] == KEY and KEY not in str(seen["params"])
    assert seen["params"]["count"] == 20                              # Brave's own maximum


def test_serpapi_shape(monkeypatch):
    seen = patch(monkeypatch, Resp({"organic_results": [{"title": "T", "snippet": "S", "link": "https://b.example"}]}))
    r = conn("serpapi").call("hello")
    assert r.ok and "T: S  [https://b.example]" in r.text
    assert seen["params"]["api_key"] == KEY and seen["params"]["engine"] == "google" and seen["url"].startswith("https://serpapi.com")


def test_google_cse_needs_its_engine_id_and_reads_items(monkeypatch):
    assert not conn("google_cse").available()                         # no cx -> not configured
    seen = patch(monkeypatch, Resp({"items": [{"title": "T", "snippet": "S", "link": "https://c.example"}]}))
    c = conn("google_cse", cx="engine-1")
    assert c.available()
    r = c.call("hello", max_results=50)
    assert r.ok and "https://c.example" in r.text
    assert seen["params"]["cx"] == "engine-1" and seen["params"]["key"] == KEY and seen["params"]["num"] == 10


def test_generic_proxy_gets_a_bearer_header_and_the_old_results_shape(monkeypatch):
    seen = patch(monkeypatch, Resp({"results": [{"title": "T", "snippet": "S", "url": "https://d.example"}]}))
    r = WebSearchConnector(searxng_url="", provider="generic", api_url="https://proxy.example/s", api_key=KEY).call("q")
    assert r.ok and "https://d.example" in r.text
    assert seen["headers"]["Authorization"] == f"Bearer {KEY}" and "key" not in seen["params"]


def test_an_answer_in_the_wrong_shape_is_no_results_not_a_crash(monkeypatch):
    patch(monkeypatch, Resp({"unexpected": True}))
    r = conn("brave").call("hello")
    assert r.ok and r.text == "no results" and r.meta["n_results"] == 0


@pytest.mark.parametrize("failure", [Resp(status=403), requests.ConnectionError(f"max retries exceeded url: /?key={KEY}"), Resp(None)])
def test_failures_never_carry_the_api_key_or_a_url(monkeypatch, failure):
    patch(monkeypatch, failure)
    r = conn("serpapi").call("hello")
    assert not r.ok and KEY not in r.error and "http" not in r.error.lower().replace("http 403", "")
    assert r.error                                                    # and it still says something useful


def test_http_errors_report_the_status_code(monkeypatch):
    patch(monkeypatch, Resp(status=429))
    assert conn("brave").call("q").error == "HTTP 429 from the search provider"


def test_searxng_still_wins_when_configured(monkeypatch):
    seen = patch(monkeypatch, Resp({"results": [{"title": "T", "content": "C", "url": "https://e.example"}]}))
    r = WebSearchConnector(searxng_url="http://localhost:8888/", provider="brave", api_key=KEY).call("q")
    assert r.ok and "T: C  [https://e.example]" in r.text and seen["url"] == "http://localhost:8888/search"


def test_nothing_configured_is_unavailable(monkeypatch):
    for v in ("SEARXNG_URL", "SEARCH_API_URL", "SEARCH_API_KEY", "SEARCH_API_PROVIDER", "SEARCH_API_CX"):
        monkeypatch.delenv(v, raising=False)
    c = WebSearchConnector()
    assert not c.available() and c.call("q").error == "no search backend configured"


def test_a_known_provider_needs_no_url_just_a_key(monkeypatch):
    for v in ("SEARXNG_URL", "SEARCH_API_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SEARCH_API_PROVIDER", "brave")
    monkeypatch.setenv("SEARCH_API_KEY", KEY)
    assert WebSearchConnector().available()


def test_scrub():
    assert scrub(f"bad url ?key={KEY}", KEY) == "bad url ?key=***"
    assert scrub("short", "ab") == "short"                           # a trivially short 'secret' would mangle ordinary text
