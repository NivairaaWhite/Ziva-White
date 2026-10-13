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

Both rules must hold. If either fails the text is still understood but the action is refused with
an explanation (see assistant.py), so nothing happens silently.
"""
from __future__ import annotations

import hmac
import os
import re
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


def origin_ok(headers: Mapping[str, str], extra_allowed: Optional[str] = None) -> bool:
    origin = _h(headers, "origin")
    if not origin:
        return True                        # not a browser cross-site request (curl, the CLI, a native app)
    if origin == "null":
        return False                       # sandboxed iframes / file:// pages
    host = _h(headers, "host")
    try:
        if host and urlparse(origin).netloc == host:
            return True                    # the page was served by this very server
    except ValueError:
        return False
    allowed = (extra_allowed if extra_allowed is not None else os.getenv("ZIVAWHITE_CORS_ORIGINS", "")).strip()
    if allowed == "*":
        return False                       # "anyone may READ answers" is not "anyone may press buttons"
    if allowed and origin in [o.strip() for o in allowed.split(",") if o.strip()]:
        return True
    return bool(_LOCAL_ORIGIN.match(origin))


def control_allowed(client_host: Optional[str], headers: Mapping[str, str], server_key: str) -> bool:
    return (is_local_request(client_host, headers) or has_valid_key(headers, server_key)) and origin_ok(headers)
