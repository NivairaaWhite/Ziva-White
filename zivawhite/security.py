"""
security.py - the "production ready, not just a local demo" pieces for
`zivawhite.app serve`: optional bearer-token auth and a basic
per-client rate limiter. Both default to OFF (matching the rest of this
project's "degrades gracefully to whatever's configured" philosophy) so
`launch.py` keeps working with zero setup on localhost --
you turn them on for anything reachable beyond your own machine.

Deliberately dependency-free (no slowapi, no redis) -- an in-memory
limiter is the right amount of engineering for a single-process app like
this one; if you're running multiple worker processes behind a load
balancer, put a real rate limiter (or your cloud provider's) in front
instead, since in-memory state doesn't share across processes.
"""
from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import Header, HTTPException, Request


# -- auth ---------------------------------------------------------------

def get_server_api_key() -> str:
    return os.getenv("ZIVAWHITE_SERVER_API_KEY", "")


def _check_bearer(authorization: Optional[str], expected: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token -- set the Authorization header")
    # constant-time compare: a plain != leaks, via response timing, how many leading
    # characters of a guess were right
    if not hmac.compare_digest(authorization[len("Bearer "):].encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: if ZIVAWHITE_SERVER_API_KEY is set, every
    request must carry `Authorization: Bearer <that key>` or gets a 401.
    If it's unset (the default), this is a no-op -- auth is opt-in, not
    forced on you the moment you clone the repo.

    This bare form has no memory of who failed. The server uses make_require_api_key(), which is the same
    check plus a lock-out for a client that keeps guessing."""
    expected = get_server_api_key()
    if not expected:
        return
    _check_bearer(authorization, expected)


def _parse_limit(raw: str, default: Tuple[int, float]) -> Tuple[int, float]:
    try:
        limit_s, window_s = raw.split("/")
        return int(limit_s), float(window_s)
    except ValueError:
        return default


# -- rate limiting --------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window` seconds per
    client key (IP by default). Good enough to blunt casual abuse/runaway clients
    on a single-process deployment; not a substitute for a real edge rate limiter
    under real load."""

    _SWEEP_AT = 10_000  # distinct keys held before expired ones are purged

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def _sweep(self, now: float) -> None:
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] > self.window_s]:
            del self._hits[key]

    def raise_if_full(self, key: str) -> None:
        """429 if `key` already used up its allowance in the window. Records nothing."""
        now = time.monotonic()
        if len(self._hits) > self._SWEEP_AT:
            self._sweep(now)  # keeps memory bounded if a client cycles through keys
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = max(0.0, self.window_s - (now - hits[0]))
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                 headers={"Retry-After": str(int(retry_after) + 1)})

    def record(self, key: str) -> None:
        self._hits[key].append(time.monotonic())

    def check(self, key: str) -> None:
        """Count one request for `key`, or 429 if it is over the limit."""
        self.raise_if_full(key)
        self.record(key)


def _client_key(request: Request) -> str:
    # X-Forwarded-For is attacker-controlled unless a reverse proxy you run sets it.
    # Trusting it unconditionally let any client dodge the limiter by sending a fresh
    # fake IP with every request, so it's only honored when you say a proxy is in front.
    if os.getenv("ZIVAWHITE_TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def make_rate_limit_dependency():
    """Reads ZIVAWHITE_RATE_LIMIT="<requests>/<seconds>" (default
    "60/60", i.e. 60 requests/minute) once at server startup and returns
    a FastAPI dependency enforcing it per client. Set it to "0/1" (or
    anything with limit<=0) to disable."""
    limit, window = _parse_limit(os.getenv("ZIVAWHITE_RATE_LIMIT", "60/60"), (60, 60.0))
    limiter = RateLimiter(limit, window) if limit > 0 else None

    def dependency(request: Request) -> None:
        if limiter is not None:
            limiter.check(_client_key(request))

    return dependency


def make_require_api_key():
    """The API-key dependency the server uses: require_api_key plus a lock-out for guessers.

    A client that has failed authentication ZIVAWHITE_AUTH_FAIL_LIMIT times ("<failures>/<seconds>", default
    "10/60", 0 disables) is answered 429 -- before its key is even looked at -- until the window passes, so a
    key cannot be brute-forced at network speed. Only FAILURES count: a client with the right key is never
    throttled by this, and the UI's polling is unaffected. Each call builds its own counter, so every app (and every
    test) starts clean."""
    limit, window = _parse_limit(os.getenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "10/60"), (10, 60.0))
    failures = RateLimiter(limit, window) if limit > 0 else None

    def dependency(request: Request, authorization: Optional[str] = Header(default=None)) -> None:
        expected = get_server_api_key()
        if not expected:
            return
        who = _client_key(request)
        if failures is not None:
            failures.raise_if_full(who)
        try:
            _check_bearer(authorization, expected)
        except HTTPException:
            if failures is not None:
                failures.record(who)
            raise

    return dependency
