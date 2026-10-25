"""
access.py - who may make this machine DO something over HTTP.

Answering a question is low risk; locking the screen or opening an app is not. The rules for the
control path of /v1/assistant, kept free of web-framework imports so they are unit-tested directly:

  1. The caller must be local (loopback, and not relayed by a reverse proxy) OR present the
     configured API key. If you put a proxy in front of this server, every request looks local to
     it -- so requests carrying proxy headers are never treated as local; use an API key.
  2. Cross-site protection: a browser always attaches an `Origin` header to a cross-site POST. If
     one is present it must be this server's own origin or one the operator allowed
     (ZIVAWHITE_CORS_ORIGINS). Another website open in your browser therefore cannot make your
     assistant do things, even though your browser can reach 127.0.0.1.

  3. DNS rebinding: a page on evil.example can make the browser re-resolve evil.example to 127.0.0.1, after
     which its requests reach this server carrying `Host: evil.example` AND `Origin: http://evil.example`.
     Those two agree, so rule 2 alone would call them "same origin". The Host header is therefore checked
     against an allow-list of names that can only mean this machine: localhost, any literal IP address (a
     rebinding attack needs a DNS name), and whatever you list in ZIVAWHITE_ALLOWED_HOSTS. The web server
     applies the same check to EVERY request (webguard.py), so such a page cannot read answers either.

All rules must hold. If any fails the text is still understood but the action is refused with
an explanation (see assistant.py), so nothing happens silently.
"""
from __future__ import annotations

import collections
import hmac
import ipaddress
import os
import re
import secrets
import threading
from typing import Mapping, Optional
from urllib.parse import urlparse

_PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "x-forwarded-host")
_LOOPBACK = re.compile(r"^(127\.\d+\.\d+\.\d+|::1|localhost|::ffff:127\.\d+\.\d+\.\d+)$")
_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def _h(headers: Mapping[str, str], name: str) -> str:
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return ""


def is_local_request(client_host: Optional[str], headers: Mapping[str, str]) -> bool:
    if any(_h(headers, n) for n in _PROXY_HEADERS):
        return False
    return bool(client_host and _LOOPBACK.match(client_host))


def has_valid_key(headers: Mapping[str, str], server_key: str) -> bool:
    if not server_key:
        return False
    auth = _h(headers, "authorization")
    if not auth.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth[len("Bearer "):].encode(), server_key.encode())


def hostname_of(host_header: str) -> str:
    """The bare, lower-case host name from a Host header value: no port, no brackets around an IPv6 address."""
    host = (host_header or "").strip().lower()
    if host.startswith("["):                       # [::1]:8000
        end = host.find("]")
        return host[1:end] if end != -1 else host
    if host.count(":") == 1:                       # name:port, or 127.0.0.1:8000
        return host.split(":", 1)[0]
    return host                                    # a bare name, or an unbracketed IPv6 literal


def _is_ip_literal(name: str) -> bool:
    try:
        ipaddress.ip_address(name.split("%", 1)[0])
        return True
    except ValueError:
        return False


def allowed_hosts_from_env() -> str:
    return os.getenv("ZIVAWHITE_ALLOWED_HOSTS", "")


def host_ok(host_header: str, extra_allowed: Optional[str] = None) -> bool:
    """Is this Host header one that can only be this machine (see rule 3 above)? An absent Host header is not a
    browser (HTTP/1.0 tools), so it passes. `extra_allowed` (default: $ZIVAWHITE_ALLOWED_HOSTS) is a comma-separated
    list of extra names such as `my-pc.local`; `*` switches the check off (not recommended)."""
    if not host_header or not host_header.strip():
        return True
    name = hostname_of(host_header)
    if name == "localhost" or _is_ip_literal(name):
        return True
    extra = [h.strip().lower() for h in (extra_allowed if extra_allowed is not None else allowed_hosts_from_env()).split(",")
             if h.strip()]
    return "*" in extra or name in [hostname_of(h) for h in extra]


def origin_ok(headers: Mapping[str, str], extra_allowed: Optional[str] = None) -> bool:
    origin = _h(headers, "origin")
    if not origin:
        return True                        # not a browser cross-site request (curl, the CLI, a native app)
    if origin == "null":
        return False                       # sandboxed iframes / file:// pages
    host = _h(headers, "host")
    try:
        # "Origin equals Host" only means "served by this very server" if Host itself can be trusted: under DNS
        # rebinding the attacker controls BOTH, and they match.
        if host and urlparse(origin).netloc == host and host_ok(host):
            return True
    except ValueError:
        return False
    allowed = (extra_allowed if extra_allowed is not None else os.getenv("ZIVAWHITE_CORS_ORIGINS", "")).strip()
    if allowed == "*":
        return False                       # "anyone may READ answers" is not "anyone may press buttons"
    if allowed and origin in [o.strip() for o in allowed.split(",") if o.strip()]:
        return True
    return bool(_LOCAL_ORIGIN.match(origin))


def control_allowed(client_host: Optional[str], headers: Mapping[str, str], server_key: str) -> bool:
    return ((is_local_request(client_host, headers) or has_valid_key(headers, server_key))
            and host_ok(_h(headers, "host")) and origin_ok(headers))


class SessionRegistry:
    """Server-issued conversation ids for HTTP callers.

    A pending "are you sure?" (lock, sleep, ...) is parked under a session id. If the CLIENT chose that id, anyone
    could name someone else's session -- including the predictable ones the voice loop and the CLI use ("voice",
    "cli") -- and answer "yes" to a question they were never asked. Here the id is a random token this server hands
    out; an id it did not issue is ignored and replaced, and the caller gets the new one back to use next time. The
    ids are also namespaced per transport (`http:<token>`), so no HTTP caller can ever produce "voice" or "cli".
    """

    def __init__(self, max_sessions: int = 2000):
        self._ids: "collections.OrderedDict[str, None]" = collections.OrderedDict()
        self._max = max(1, max_sessions)
        self._lock = threading.Lock()

    def resolve(self, presented: Optional[str]) -> str:
        """The caller's own session id if this registry issued it, otherwise a newly issued one."""
        with self._lock:
            if presented and presented in self._ids:
                self._ids.move_to_end(presented)
                return presented
            token = secrets.token_urlsafe(18)
            self._ids[token] = None
            while len(self._ids) > self._max:
                self._ids.popitem(last=False)             # forget the least recently used
            return token

    @staticmethod
    def key(token: str) -> str:
        """The key the assistant files this session under."""
        return f"http:{token}"
