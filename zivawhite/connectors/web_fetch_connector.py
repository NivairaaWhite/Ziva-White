"""
Fetches and reads a single URL - the "surf the internet" step, used
after web_search_connector finds candidate pages.

Rules baked in on purpose, not negotiable via arguments:
  - only http(s) URLs, and only ones that resolve to PUBLIC addresses. A
    URL pointing at localhost, a private network or a cloud metadata
    address (169.254.169.254) is refused -- otherwise anyone who can reach
    the server could use it to read internal services (SSRF). Redirects are
    followed by hand (max 3) and every hop is re-checked. Set
    ZIVAWHITE_ALLOW_PRIVATE_FETCH=1 to turn this off for your own LAN.
    (Limitation: the name is resolved once for the check and again by the
    HTTP library, so a hostile DNS server could in theory answer differently
    the second time.)
  - robots.txt is always checked before fetching, per RFC 9309: a missing
    robots.txt (4xx) allows the fetch; a server error or an unreachable site
    does not. Disallowed URLs are refused rather than fetched anyway.
  - one request per host at a time, with a minimum interval between
    requests to the same host (_MIN_INTERVAL) -- polite by default.
  - a UA string that identifies the bot honestly instead of spoofing
    a browser.
  - downloads are capped (_MAX_BYTES) so one huge file can't eat memory.
This is what "without being limited" means in practice here: not
rate-limited by a paid API's quota, not "ignore site owners' wishes."
"""
from __future__ import annotations

import ipaddress
import os
import socket
import time
import urllib.robotparser as robotparser
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base import Connector, ConnectorResult

_USER_AGENT = "ZivaWhiteBot/0.1 (+https://example.invalid/bot; research/personal use)"
_MIN_INTERVAL = 2.0  # seconds between requests to the same host
_MAX_REDIRECTS = 3
_MAX_BYTES = 10_000_000


def _allow_private() -> bool:
    return os.getenv("ZIVAWHITE_ALLOW_PRIVATE_FETCH", "").lower() in ("1", "true", "yes")


def check_url(url: str) -> Optional[str]:
    """None if `url` is OK to fetch, otherwise a short reason it isn't."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return "malformed URL"
    if parsed.scheme not in ("http", "https"):
        return f"unsupported scheme {parsed.scheme!r} (only http/https)"
    if not host:
        return "URL has no host"
    if _allow_private():
        return None
    try:
        addresses = {info[4][0].split("%")[0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        return f"could not resolve {host}"
    for addr in addresses:
        if not ipaddress.ip_address(addr).is_global:
            return f"{host} resolves to a non-public address ({addr})"
    return None


class WebFetchConnector(Connector):
    name = "web_fetch"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._robots_cache: Dict[str, robotparser.RobotFileParser] = {}
        self._last_hit: Dict[str, float] = {}

    def available(self) -> bool:
        return True  # no API key needed, but robots.txt can still say no

    def _robots_allows(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(origin)
        if rp is None:
            rp = robotparser.RobotFileParser()
            try:
                # urllib's own rp.read() sends Python's default User-Agent (which many sites
                # block, turning "allow" into "deny everything") and has no timeout.
                resp = requests.get(f"{origin}/robots.txt", headers={"User-Agent": _USER_AGENT},
                                    timeout=self.timeout)
            except requests.RequestException:
                return False  # can't tell -> don't fetch; not cached, so a later call can retry
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                rp.modified()
            elif 400 <= resp.status_code < 500:
                rp.allow_all = True  # no robots.txt -> everything allowed
            else:
                return False  # server error -> assume disallowed, and retry next time
            self._robots_cache[origin] = rp
        try:
            return rp.can_fetch(_USER_AGENT, url)
        except Exception:
            return False

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_hit.get(host, 0.0)
        wait = _MIN_INTERVAL - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.time()

    def _get(self, url: str) -> requests.Response:
        """GET with manual, re-validated redirects. Raises ValueError with the
        reason when a hop is refused."""
        for _ in range(_MAX_REDIRECTS + 1):
            problem = check_url(url)
            if problem:
                raise ValueError(problem)
            if not self._robots_allows(url):
                raise ValueError("disallowed by robots.txt")
            self._throttle(url)
            resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=self.timeout,
                                allow_redirects=False, stream=True)
            location = resp.headers.get("Location")
            if resp.is_redirect and location:
                resp.close()
                url = urljoin(url, location)
                continue
            return resp
        raise ValueError(f"more than {_MAX_REDIRECTS} redirects")

    @staticmethod
    def _read_capped(resp: requests.Response) -> str:
        body = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) >= _MAX_BYTES:
                break
        resp.close()
        return bytes(body).decode(resp.encoding or "utf-8", errors="replace")

    def call(self, prompt: str, max_chars: int = 4000, **kw) -> ConnectorResult:
        url = prompt.strip()
        try:
            resp = self._get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(self._read_capped(resp), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return ConnectorResult(
                text=text[:max_chars],
                source=self.name,
                meta={"url": url, "truncated": len(text) > max_chars},
            )
        except Exception as e:  # includes policy refusals (SSRF guard, robots.txt, redirect loops)
            return ConnectorResult(text="", source=self.name, ok=False, error=str(e))
