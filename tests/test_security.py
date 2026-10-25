import pytest
from fastapi import HTTPException

from zivawhite.security import RateLimiter, _client_key, make_rate_limit_dependency, require_api_key


class FakeClient:
    def __init__(self, host):
        self.host = host


class FakeRequest:
    """Minimal stand-in for fastapi.Request -- just enough for
    _client_key() and the rate-limit dependency to read from, without
    pulling in httpx/TestClient as a test-only dependency this project
    doesn't otherwise need."""
    def __init__(self, host="1.2.3.4", forwarded_for=None):
        self.client = FakeClient(host)
        self.headers = {"x-forwarded-for": forwarded_for} if forwarded_for else {}


# -- auth ---------------------------------------------------------------

def test_require_api_key_noop_when_unset(monkeypatch):
    monkeypatch.delenv("ZIVAWHITE_SERVER_API_KEY", raising=False)
    require_api_key(authorization=None)  # must not raise


def test_require_api_key_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        require_api_key(authorization=None)
    assert exc.value.status_code == 401


def test_require_api_key_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        require_api_key(authorization="Bearer wrong")
    assert exc.value.status_code == 401


def test_require_api_key_accepts_correct_token(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    require_api_key(authorization="Bearer secret")  # must not raise


# -- rate limiting --------------------------------------------------------

def test_rate_limiter_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(limit=2, window_s=60)
    limiter.check("client-a")
    limiter.check("client-a")
    with pytest.raises(HTTPException) as exc:
        limiter.check("client-a")
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(limit=1, window_s=60)
    limiter.check("client-a")
    limiter.check("client-b")  # different client, must not raise


def test_rate_limiter_window_expires(monkeypatch):
    import time as time_module
    limiter = RateLimiter(limit=1, window_s=0.05)
    limiter.check("client-a")
    time_module.sleep(0.1)
    limiter.check("client-a")  # window elapsed, must not raise


def test_client_key_ignores_forwarded_for_by_default(monkeypatch):
    # an attacker-supplied header must not let a client pick its own rate-limit identity
    monkeypatch.delenv("ZIVAWHITE_TRUST_PROXY", raising=False)
    req = FakeRequest(host="10.0.0.1", forwarded_for="203.0.113.9, 10.0.0.1")
    assert _client_key(req) == "10.0.0.1"


def test_client_key_uses_forwarded_for_behind_a_trusted_proxy(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_TRUST_PROXY", "1")
    req = FakeRequest(host="10.0.0.1", forwarded_for="203.0.113.9, 10.0.0.1")
    assert _client_key(req) == "203.0.113.9"


def test_rate_limiter_forgets_idle_clients():
    import time as time_module

    limiter = RateLimiter(limit=5, window_s=0.01)
    limiter._SWEEP_AT = 3
    for i in range(10):
        limiter.check(f"client-{i}")
    time_module.sleep(0.03)
    limiter.check("fresh")  # triggers the sweep: expired clients are dropped
    assert len(limiter._hits) < 10


def test_require_api_key_accepts_the_right_token(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    require_api_key(authorization="Bearer secret")  # must not raise


def test_client_key_falls_back_to_direct_host():
    req = FakeRequest(host="10.0.0.1")
    assert _client_key(req) == "10.0.0.1"


def test_make_rate_limit_dependency_disabled_with_zero_limit(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_RATE_LIMIT", "0/60")
    dep = make_rate_limit_dependency()
    req = FakeRequest()
    for _ in range(100):
        dep(req)  # must never raise when disabled


def test_make_rate_limit_dependency_enforces_configured_limit(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_RATE_LIMIT", "1/60")
    dep = make_rate_limit_dependency()
    req = FakeRequest()
    dep(req)
    with pytest.raises(HTTPException):
        dep(req)


# -- brute-force protection ---------------------------------------------------
from zivawhite.security import make_require_api_key  # noqa: E402


def _guess(dep, req, token):
    dep(req, authorization=f"Bearer {token}")


def test_repeated_wrong_keys_are_locked_out_with_429_even_for_a_later_correct_key(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    monkeypatch.setenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "3/60")
    dep, req = make_require_api_key(), FakeRequest(host="203.0.113.7")
    for i in range(3):
        with pytest.raises(HTTPException) as exc:
            _guess(dep, req, f"wrong{i}")
        assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        _guess(dep, req, "wrong3")
    assert exc.value.status_code == 429 and "Retry-After" in exc.value.headers
    with pytest.raises(HTTPException) as exc:          # locked out BEFORE the key is even looked at: a lucky guess can't win
        _guess(dep, req, "secret")
    assert exc.value.status_code == 429


def test_the_lockout_is_per_client_and_successes_never_count(monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    monkeypatch.setenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "2/60")
    dep = make_require_api_key()
    good = FakeRequest(host="198.51.100.1")
    for _ in range(50):
        _guess(dep, good, "secret")                    # polling with the right key is never throttled by this
    bad, other = FakeRequest(host="203.0.113.7"), FakeRequest(host="203.0.113.8")
    for i in range(2):
        with pytest.raises(HTTPException):
            _guess(dep, bad, f"x{i}")
    with pytest.raises(HTTPException) as exc:
        _guess(dep, bad, "x")
    assert exc.value.status_code == 429
    _guess(dep, other, "secret")                       # a different client is unaffected
    _guess(dep, good, "secret")


def test_lockout_is_a_noop_without_a_configured_key_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("ZIVAWHITE_SERVER_API_KEY", raising=False)
    dep = make_require_api_key()
    for _ in range(30):
        dep(FakeRequest(), authorization=None)
    monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", "secret")
    monkeypatch.setenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "0/60")
    off = make_require_api_key()
    for _ in range(30):
        with pytest.raises(HTTPException) as exc:
            _guess(off, FakeRequest(), "nope")
        assert exc.value.status_code == 401            # never escalates to 429 when disabled
