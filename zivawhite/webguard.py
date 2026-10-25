"""
Request guards for the HTTP server, written as plain ASGI middleware (no FastAPI/Starlette import), so they are
small, order-independent and testable without a web framework installed.

  HostGuardMiddleware       refuses any request whose Host header is not this machine (DNS rebinding, see access.py)
  BodyLimitMiddleware       refuses request bodies over a size limit with 413, before the app parses them
  SecurityHeadersMiddleware adds nosniff / no-referrer / frame-deny to every response, and a strict
                            Content-Security-Policy to HTML pages (the web UI)

server.create_app() installs all three.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from typing import Awaitable, Callable, Dict, List, Optional

from .access import allowed_hosts_from_env, host_ok, hostname_of

Scope = Dict[str, object]
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]

DEFAULT_MAX_BODY_BYTES = 1_000_000          # ~1 MB: far more than any prompt a model with this context can use


def max_body_bytes() -> int:
    """ZIVAWHITE_MAX_BODY_BYTES, or 1 MB. 0 or a negative number turns the limit off."""
    try:
        return int(os.getenv("ZIVAWHITE_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES)))
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES


def _header(scope: Scope, name: bytes) -> Optional[str]:
    for k, v in scope.get("headers", []):  # type: ignore[union-attr]
        if k.lower() == name:
            return v.decode("latin-1")
    return None


async def _respond(send: Send, status: int, body: str, extra: Optional[List[tuple]] = None) -> None:
    payload = body.encode("utf-8")
    headers = [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(payload)).encode())]
    headers += extra or []
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


# --------------------------------------------------------------------------------------------- Host header
class HostGuardMiddleware:
    """400 for a Host header that is not localhost, an IP literal, or listed in ZIVAWHITE_ALLOWED_HOSTS."""

    def __init__(self, app, extra_allowed: Optional[str] = None):
        self.app = app
        self.extra_allowed = extra_allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            host = _header(scope, b"host")
            extra = self.extra_allowed if self.extra_allowed is not None else allowed_hosts_from_env()
            if not host_ok(host or "", extra):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                name = hostname_of(host or "")
                await _respond(send, 400,
                               f"Invalid Host header {name!r}. This server only answers to localhost and IP addresses by default "
                               f"(a defence against DNS rebinding). To reach it by another name, set "
                               f"ZIVAWHITE_ALLOWED_HOSTS={name} and restart.\n")
                return
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------------------------- body size
class BodyLimitMiddleware:
    """413 for a request body over `max_bytes`: checked up front from Content-Length, and while reading for
    bodies that arrive chunked with no length. The app never sees the oversized part."""

    def __init__(self, app, max_bytes: Optional[int] = None):
        self.app = app
        self.max_bytes = max_body_bytes() if max_bytes is None else max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._too_large(send)
            return

        state = {"received": 0, "exceeded": False, "started": False}

        async def counting_receive() -> dict:
            msg = await receive()
            if msg["type"] == "http.request" and not state["exceeded"]:
                state["received"] += len(msg.get("body", b""))
                if state["received"] > self.max_bytes:
                    state["exceeded"] = True
            if state["exceeded"]:
                return {"type": "http.disconnect"}          # the app stops reading; we answer for it below
            return msg

        async def guarded_send(msg: dict) -> None:
            if state["exceeded"]:
                return                                       # whatever the app says about a body it never got
            if msg["type"] == "http.response.start":
                state["started"] = True
            await send(msg)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not state["exceeded"]:
                raise
        if state["exceeded"] and not state["started"]:
            await self._too_large(send)

    async def _too_large(self, send: Send) -> None:
        await _respond(send, 413, f"Request body too large (limit {self.max_bytes:,} bytes).\n")


# --------------------------------------------------------------------------------------------- headers
_BASE_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cross-origin-opener-policy", b"same-origin"),
]

_INLINE_BLOCK = re.compile(r"<(script|style)(?:\s[^>]*)?>(.*?)</\1\s*>", re.S | re.I)


def csp_for_page(html: str) -> str:
    """A Content-Security-Policy for one self-contained HTML page: scripts and styles may run only if their text is
    exactly what is in the page right now (hash-allow-listed), nothing may be loaded from other sites, and the
    page may talk only to the server it came from. An injected <script>, an inline handler or a remote include
    simply does not run. No 'unsafe-inline'."""
    hashes: Dict[str, List[str]] = {"script": [], "style": []}
    for tag, body in _INLINE_BLOCK.findall(html):
        digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii")
        hashes[tag.lower()].append(f"'sha256-{digest}'")
    script_src = " ".join(["'self'"] + hashes["script"]) if hashes["script"] else "'none'"
    style_src = " ".join(["'self'"] + hashes["style"]) if hashes["style"] else "'none'"
    return "; ".join([
        "default-src 'none'",
        f"script-src {script_src}",
        f"style-src {style_src}",
        "connect-src 'self'",
        "img-src 'self' data:",
        "media-src 'self' blob:",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


class SecurityHeadersMiddleware:
    """Adds the always-safe headers to every response, and `html_csp` to HTML responses. A header the app already
    set is left alone."""

    def __init__(self, app, html_csp: Optional[str] = None):
        self.app = app
        self.html_csp = html_csp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(msg: dict) -> None:
            if msg["type"] == "http.response.start":
                headers = list(msg.get("headers", []))
                present = {k.lower() for k, _ in headers}
                for k, v in _BASE_HEADERS:
                    if k not in present:
                        headers.append((k, v))
                is_html = any(k.lower() == b"content-type" and v.lower().startswith(b"text/html") for k, v in headers)
                if is_html and self.html_csp and b"content-security-policy" not in present:
                    headers.append((b"content-security-policy", self.html_csp.encode("latin-1")))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, send_with_headers)
