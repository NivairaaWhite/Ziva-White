"""
Web search backend.

Primary: a self-hosted SearxNG instance (SEARXNG_URL). SearxNG fans a
query out to dozens of real engines and hands back JSON -- no
per-query billing and no third-party rate limit, because it's your
own infrastructure. This is the "search without being limited" path:
`docker run -d -p 8888:8888 searxng/searxng` and point SEARXNG_URL at
it (see docs/ROUTER.md).

Fallback: a commercial search API, for when standing up SearxNG isn't an
option. Every provider answers in its own JSON shape and authenticates its
own way, so each one has an adapter here -- you name it in
SEARCH_API_PROVIDER (or it is recognised from SEARCH_API_URL):

    brave        Brave Search API. Key in the X-Subscription-Token header. The recommended choice.
    serpapi      SerpAPI (Google results). The key travels as an `api_key` query parameter -- SerpAPI offers no header.
    google_cse   Google Custom Search JSON API (needs SEARCH_API_CX too). Key as a query parameter. Google has closed
                 it to new customers and switches it off on 2027-01-01, so this is for existing customers only.
    generic      A proxy you run yourself that returns {"results": [{"title", "url", "snippet"}]}; the key is sent as
                 `Authorization: Bearer <key>`.

(Bing's Search API is not here: Microsoft shut it down on 2025-08-11.)

A provider that wants the key in the URL puts it in a place logs like to copy. This connector never prints or
returns a URL it sent a key in: failures are reported as "HTTP 403 from the search provider", and any trace of the
key is scrubbed from whatever message is left.

Either way, this connector fetches result *snippets*, not full
copyrighted page bodies -- full-page reading (with robots.txt
enforcement) is a separate, explicit step in web_fetch_connector.py.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from .base import Connector, ConnectorResult

Hit = Tuple[str, str, str]                      # (title, snippet, url)

PROVIDERS = ("brave", "serpapi", "google_cse", "generic")
_DEFAULT_URL = {
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "serpapi": "https://serpapi.com/search.json",
    "google_cse": "https://www.googleapis.com/customsearch/v1",
}


def detect_provider(url: str, explicit: str = "") -> str:
    """The adapter for this configuration: SEARCH_API_PROVIDER if set, else recognised from the URL's host."""
    explicit = (explicit or "").strip().lower()
    if explicit:
        return explicit if explicit in PROVIDERS else "generic"
    host = (urlparse(url or "").hostname or "").lower()
    if host.endswith("brave.com"):
        return "brave"
    if host.endswith("serpapi.com"):
        return "serpapi"
    if host.endswith("googleapis.com") and "customsearch" in (url or ""):
        return "google_cse"
    return "generic"


# ---- one adapter per provider: how to ask, and how to read the answer ---------------------------------------
def _hits(items: Any, title: str, snippet: str, url: str) -> List[Hit]:
    out: List[Hit] = []
    for r in items if isinstance(items, list) else []:
        if isinstance(r, dict):
            out.append((str(r.get(title) or ""), str(r.get(snippet) or ""), str(r.get(url) or "")))
    return out


def _request_brave(query: str, n: int, key: str, cx: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    return ({"X-Subscription-Token": key, "Accept": "application/json"}, {"q": query, "count": min(n, 20)})


def _request_serpapi(query: str, n: int, key: str, cx: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    return ({}, {"engine": "google", "q": query, "num": n, "api_key": key})


def _request_google_cse(query: str, n: int, key: str, cx: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    return ({}, {"key": key, "cx": cx, "q": query, "num": min(n, 10)})


def _request_generic(query: str, n: int, key: str, cx: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    return ({"Authorization": f"Bearer {key}"}, {"q": query, "count": n})


def _read_brave(data: Any) -> List[Hit]:
    return _hits(((data or {}).get("web") or {}).get("results"), "title", "description", "url")


def _read_serpapi(data: Any) -> List[Hit]:
    return _hits((data or {}).get("organic_results"), "title", "snippet", "link")


def _read_google_cse(data: Any) -> List[Hit]:
    return _hits((data or {}).get("items"), "title", "snippet", "link")


def _read_generic(data: Any) -> List[Hit]:
    return _hits((data or {}).get("results"), "title", "snippet", "url")


_ADAPTERS: Dict[str, Tuple[Callable, Callable]] = {
    "brave": (_request_brave, _read_brave),
    "serpapi": (_request_serpapi, _read_serpapi),
    "google_cse": (_request_google_cse, _read_google_cse),
    "generic": (_request_generic, _read_generic),
}


def scrub(text: str, *secrets: str) -> str:
    """Remove every secret from a message before it is stored, logged or shown."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


class WebSearchConnector(Connector):
    name = "web_search"

    def __init__(self, searxng_url: Optional[str] = None, timeout: float = 10.0, *, provider: Optional[str] = None,
                 api_url: Optional[str] = None, api_key: Optional[str] = None, cx: Optional[str] = None):
        self.searxng_url = (searxng_url or os.getenv("SEARXNG_URL", "")).rstrip("/")
        self.fallback_key = api_key if api_key is not None else os.getenv("SEARCH_API_KEY", "")
        self.cx = cx if cx is not None else os.getenv("SEARCH_API_CX", "")
        url = api_url if api_url is not None else os.getenv("SEARCH_API_URL", "")
        self.provider = detect_provider(url, provider if provider is not None else os.getenv("SEARCH_API_PROVIDER", ""))
        self.fallback_url = url or _DEFAULT_URL.get(self.provider, "")
        self.timeout = timeout

    def _fallback_ready(self) -> bool:
        return bool(self.fallback_url and self.fallback_key) and (self.provider != "google_cse" or bool(self.cx))

    def available(self) -> bool:
        return bool(self.searxng_url) or self._fallback_ready()

    def call(self, prompt: str, max_results: int = 5, **kw) -> ConnectorResult:
        if self.searxng_url:
            return self._searxng(prompt, max_results)
        if self._fallback_ready():  # same condition available() uses
            return self._fallback(prompt, max_results)
        return ConnectorResult(text="", source=self.name, ok=False, error="no search backend configured")

    def _fail(self, error: str) -> ConnectorResult:
        return ConnectorResult(text="", source=self.name, ok=False, error=scrub(error, self.fallback_key, self.cx))

    def _describe(self, exc: Exception, who: str) -> str:
        """A failure message that can never carry a URL: a request URL may contain the API key."""
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            return f"HTTP {exc.response.status_code} from {who}"
        if isinstance(exc, requests.RequestException):
            return f"{type(exc).__name__} reaching {who}"
        if isinstance(exc, ValueError):
            return f"{who} returned something that is not JSON"
        return f"{type(exc).__name__} while searching"

    def _searxng(self, query: str, max_results: int) -> ConnectorResult:
        try:
            resp = requests.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:max_results]
            return self._format([(r.get("title", ""), r.get("content", ""), r.get("url", "")) for r in results])
        except Exception as e:  # noqa: BLE001
            return self._fail(self._describe(e, "SearxNG"))

    def _fallback(self, query: str, max_results: int) -> ConnectorResult:
        build_request, read_response = _ADAPTERS[self.provider]
        headers, params = build_request(query, max_results, self.fallback_key, self.cx)
        try:
            resp = requests.get(self.fallback_url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return self._format(read_response(resp.json())[:max_results])
        except Exception as e:  # noqa: BLE001
            return self._fail(self._describe(e, "the search provider"))

    @staticmethod
    def _format(hits: List[Hit]) -> ConnectorResult:
        lines = [f"- {title}: {body}  [{url}]" for title, body, url in hits]
        return ConnectorResult(
            text="\n".join(lines) if lines else "no results",
            source="web_search",
            meta={"urls": [url for _, _, url in hits], "n_results": len(hits)},
        )
