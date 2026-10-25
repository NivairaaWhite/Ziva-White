"""The plain-ASGI guards in webguard.py, driven without any web framework: Host allow-list (DNS rebinding),
request-body cap, and security headers / CSP."""
import asyncio
import os
import re

import pytest

from zivawhite.webguard import (BodyLimitMiddleware, HostGuardMiddleware, SecurityHeadersMiddleware, csp_for_page)

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "..", "zivawhite", "webui", "index.html")


def run(app, *, host="localhost:8000", method="POST", chunks=(b"",), extra_headers=(), scope_type="http"):
    """Drive an ASGI app once. Returns (status, {header: value}, body, sent_messages)."""
    headers = [(b"host", host.encode())] if host is not None else []
    headers += [(k.encode(), v.encode()) for k, v in extra_headers]
    scope = {"type": scope_type, "method": method, "path": "/", "headers": headers}
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
    sent = []

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    start = next((m for m in sent if m["type"] in ("http.response.start", "websocket.close")), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    hdrs = {k.decode().lower(): v.decode() for k, v in (start or {}).get("headers", [])}
    return (start or {}).get("status"), hdrs, body, sent


class Echo:
    """A tiny app: reads the whole body, answers 200 with its length. Remembers whether it was called."""

    def __init__(self, content_type=b"application/json"):
        self.calls, self.content_type = 0, content_type

    async def __call__(self, scope, receive, send):
        self.calls += 1
        n = 0
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            n += len(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", self.content_type)]})
        await send({"type": "http.response.body", "body": str(n).encode()})


# ------------------------------------------------------------------------------------------ Host header
@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1:8000", "[::1]:8000", "192.168.1.20:8000", "localhost"])
def test_hosts_that_can_only_mean_this_machine_pass(host):
    app = Echo()
    status, *_ = run(HostGuardMiddleware(app, extra_allowed=""), host=host)
    assert status == 200 and app.calls == 1


@pytest.mark.parametrize("host", ["evil.example:8000", "evil.example", "localhost.evil.example", "127.0.0.1.evil.example:8000"])
def test_a_rebound_hostname_is_refused_before_the_app_runs(host):
    app = Echo()
    status, _, body, _ = run(HostGuardMiddleware(app, extra_allowed=""), host=host)
    assert status == 400 and app.calls == 0
    assert b"ZIVAWHITE_ALLOWED_HOSTS" in body            # the refusal says how to fix a legitimate setup


def test_extra_hosts_come_from_the_setting_or_the_environment(monkeypatch):
    assert run(HostGuardMiddleware(Echo(), extra_allowed="my-pc.local"), host="my-pc.local:8000")[0] == 200
    monkeypatch.setenv("ZIVAWHITE_ALLOWED_HOSTS", "box.lan")
    assert run(HostGuardMiddleware(Echo()), host="box.lan")[0] == 200
    assert run(HostGuardMiddleware(Echo()), host="other.lan")[0] == 400


def test_a_websocket_with_a_bad_host_is_closed():
    app = Echo()
    _, _, _, sent = run(HostGuardMiddleware(app, extra_allowed=""), host="evil.example", scope_type="websocket")
    assert sent[0]["type"] == "websocket.close" and app.calls == 0


def test_a_request_with_no_host_header_is_not_a_browser_and_passes():
    assert run(HostGuardMiddleware(Echo(), extra_allowed=""), host=None)[0] == 200


# ------------------------------------------------------------------------------------------ body size
def test_a_declared_oversized_body_is_refused_without_calling_the_app():
    app = Echo()
    status, _, _, _ = run(BodyLimitMiddleware(app, max_bytes=100), chunks=(b"x" * 500,),
                          extra_headers=[("content-length", "500")])
    assert status == 413 and app.calls == 0


def test_a_chunked_body_with_no_length_is_cut_off_while_reading():
    app = Echo()
    status, _, body, sent = run(BodyLimitMiddleware(app, max_bytes=100), chunks=(b"x" * 60, b"x" * 60, b"x" * 60))
    assert status == 413 and b"too large" in body
    assert sum(1 for m in sent if m["type"] == "http.response.start") == 1      # exactly one answer


def test_bodies_within_the_limit_pass_untouched():
    app = Echo()
    status, _, body, _ = run(BodyLimitMiddleware(app, max_bytes=100), chunks=(b"x" * 40, b"x" * 40),
                             extra_headers=[("content-length", "80")])
    assert status == 200 and body == b"80"


def test_zero_turns_the_limit_off():
    status, _, body, _ = run(BodyLimitMiddleware(Echo(), max_bytes=0), chunks=(b"x" * 5000,))
    assert status == 200 and body == b"5000"


def test_the_limit_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_MAX_BODY_BYTES", "10")
    assert run(BodyLimitMiddleware(Echo()), chunks=(b"x" * 50,))[0] == 413
    monkeypatch.setenv("ZIVAWHITE_MAX_BODY_BYTES", "not-a-number")
    assert run(BodyLimitMiddleware(Echo()), chunks=(b"x" * 50,))[0] == 200      # falls back to the 1 MB default


# ------------------------------------------------------------------------------------------ headers and CSP
def test_every_response_gets_the_basic_hardening_headers_and_only_html_gets_a_csp():
    json_status, json_h, _, _ = run(SecurityHeadersMiddleware(Echo(b"application/json"), html_csp="default-src 'none'"))
    assert json_h["x-content-type-options"] == "nosniff" and json_h["x-frame-options"] == "DENY"
    assert json_h["referrer-policy"] == "no-referrer" and "content-security-policy" not in json_h
    _, html_h, _, _ = run(SecurityHeadersMiddleware(Echo(b"text/html; charset=utf-8"), html_csp="default-src 'none'"))
    assert html_h["content-security-policy"] == "default-src 'none'"


def test_a_header_the_app_set_itself_is_not_duplicated():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"X-Content-Type-Options", b"custom"), (b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b""})
    _, _, _, sent = run(SecurityHeadersMiddleware(app))
    names = [k.lower() for k, _ in sent[0]["headers"]]
    assert names.count(b"x-content-type-options") == 1


def test_csp_allows_only_the_pages_own_inline_blocks_and_never_unsafe_inline():
    page = "<html><head><style>a{color:red}</style></head><body><script>var x=1;</script></body></html>"
    csp = csp_for_page(page)
    assert "'unsafe-inline'" not in csp and "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert len(re.findall(r"'sha256-[A-Za-z0-9+/=]+'", csp)) == 2
    changed = csp_for_page(page.replace("var x=1;", "var x=2;"))
    assert changed != csp                                   # editing the script changes its hash: injected code can't match


def test_the_real_web_ui_is_covered_by_its_own_csp():
    html = open(INDEX, encoding="utf-8").read()
    csp = csp_for_page(html)
    assert len(re.findall(r"'sha256-", csp)) == 2           # exactly one <script> and one <style> block
    assert "'unsafe-inline'" not in csp and "connect-src 'self'" in csp
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), "an inline event handler would be blocked by the CSP"
    assert 'style="' not in html, "an inline style attribute would be blocked by the CSP"
