"""
Web search backend.

Primary: a self-hosted SearxNG instance (SEARXNG_URL). SearxNG fans a
query out to dozens of real engines and hands back JSON -- no
per-query billing and no third-party rate limit, because it's your
own infrastructure. This is the "search without being limited" path:
`docker run -d -p 8888:8888 searxng/searxng` and point SEARXNG_URL at
it (see README).

Fallback: any commercial search API (Bing / Google CSE / SerpAPI) if
the caller sets SEARCH_API_KEY + SEARCH_API_URL, for when standing up
SearxNG isn't an option.

Either way, this connector fetches result *snippets*, not full
copyrighted page bodies -- full-page reading (with robots.txt
enforcement) is a separate, explicit step in web_fetch_connector.py.
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests

from .base import Connector, ConnectorResult


class WebSearchConnector(Connector):
    name = "web_search"

    def __init__(self, searxng_url: Optional[str] = None, timeout: float = 10.0):
        self.searxng_url = (searxng_url or os.getenv("SEARXNG_URL", "")).rstrip("/")
        self.fallback_url = os.getenv("SEARCH_API_URL", "")
        self.fallback_key = os.getenv("SEARCH_API_KEY", "")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.searxng_url) or bool(self.fallback_url and self.fallback_key)

    def call(self, prompt: str, max_results: int = 5, **kw) -> ConnectorResult:
        if self.searxng_url:
            return self._searxng(prompt, max_results)
        if self.fallback_url and self.fallback_key:  # same condition available() uses
            return self._fallback(prompt, max_results)
        return ConnectorResult(text="", source=self.name, ok=False, error="no search backend configured")

    def _searxng(self, query: str, max_results: int) -> ConnectorResult:
        try:
            resp = requests.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:max_results]
            return self._format(results, "url", "title", "content")
        except Exception as e:
            return ConnectorResult(text="", source=self.name, ok=False, error=str(e))

    def _fallback(self, query: str, max_results: int) -> ConnectorResult:
        try:
            resp = requests.get(
                self.fallback_url,
                params={"q": query, "count": max_results, "key": self.fallback_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:max_results]
            return self._format(results, "url", "title", "snippet")
        except Exception as e:
            return ConnectorResult(text="", source=self.name, ok=False, error=str(e))

    @staticmethod
    def _format(results: List[dict], url_k: str, title_k: str, body_k: str) -> ConnectorResult:
        lines, urls = [], []
        for r in results:
            title, body, url = r.get(title_k, ""), r.get(body_k, ""), r.get(url_k, "")
            lines.append(f"- {title}: {body}  [{url}]")
            urls.append(url)
        return ConnectorResult(
            text="\n".join(lines) if lines else "no results",
            source="web_search",
            meta={"urls": urls, "n_results": len(results)},
        )
