<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
The HTTP API, its hardening, and what is deliberately not limited.

# HTTP API

```bash
python -m zivawhite.app serve --port 8000     # API + web UI only; `run` adds the background jobs. Binds 127.0.0.1
```

```
GET  /v1/health                                    runtime, which model file is loaded, web search configured?
GET  /v1/version                                   app version, whether auth is required
POST /v1/ask       {"query": "...", "route": "auto"}            routed answer (identity/local/web); never touches the device
POST /v1/generate  {"prompt": "...", "max_new_tokens": 200, "raw": false}   local model only, new text only
POST /v1/assistant {"text": "...", "session": "<id from the last reply>", "route": "auto"}   the full assistant: identity, DEVICE CONTROL, model
GET  /v1/assistant/info                            names, wake words (for the web UI)
GET  /v1/assistant/events?since=N                  background notifications (timers)
POST /v1/voice/stop                                stop speaking (local/authenticated callers)
GET  /v1/datasets                                  folder status + what the learner is doing
GET  /v1/status                                    background jobs, model, learning state
```

`/v1/assistant` is the only endpoint that can act on the machine, and only for loopback callers or ones
presenting the API key whose `Origin` isn't a foreign site (see "Device control"). Everything else about
auth below applies to all `/v1` endpoints except `/v1/health`, `/v1/version` and `/v1/assistant/info`.

`route` is one of `auto`, `local`, `web`; anything else is a 422 (it used to be silently treated as `web`).
`session` is the conversation id the server returned in the previous reply: omit it for a new conversation and
keep the one you are given. An id the server did not issue is ignored and a new one is returned.

No `image_base64`/`image_media_type` fields any more — see "Removed:
the AI-API wrapper" above for why multimodal input didn't survive that
removal, and "Removed: the artificial limits on the model" below for
why `query`/`prompt`/`max_new_tokens` no longer have arbitrary caps.

### Production hardening — opt-in, off by default

Auth and rate limiting default to off/loose, so `launch.py` and
local testing keep working with zero setup. Turn each on via `.env`
(see `.env.example`) before putting this somewhere other than your own
machine can reach it — see `security.py` for the implementation. These
protect the **server** (who can call it, how often) — a different
thing from the model-capacity caps removed below, and unaffected by
that removal:

- **Auth** — set `ZIVAWHITE_SERVER_API_KEY` and the `/v1` endpoints (ask, generate, assistant,
  events, datasets, status) require `Authorization: Bearer <that key>`, 401
  otherwise. `/v1/health` and `/v1/version` stay open (so a monitoring
  probe doesn't need a key), and `/v1/version` reports
  `"auth_required": true/false` so the bundled webui knows to show its
  API-key field automatically.
- **Rate limiting** — `ZIVAWHITE_RATE_LIMIT="<requests>/<seconds>"`
  (default `60/60`), per client IP, 429 + `Retry-After` once exceeded.
  `X-Forwarded-For` is only honored with `ZIVAWHITE_TRUST_PROXY=1` (otherwise any client
  could dodge the limit by faking it). In-memory and
  single-process by design — put a real edge limiter in front if you're
  running multiple worker processes.
- **Wrong-key lock-out** — `ZIVAWHITE_AUTH_FAIL_LIMIT="<failures>/<seconds>"` (default `10/60`, `0` turns it off):
  a client that sends that many wrong keys is answered 429 (before its key is even looked at) until the window
  passes, so a key cannot be brute-forced at network speed. Only failures count, so a client with the right key
  (the web UI's polling included) is never throttled by it. The general rate limiter now runs *before* the key
  check, so failed logins on `ask`/`generate`/`assistant` are counted too.
- **Host check (DNS rebinding)** — every request's `Host` header must be `localhost`, an IP address, or listed in
  `ZIVAWHITE_ALLOWED_HOSTS` (comma-separated, e.g. `my-pc.local`); anything else is a 400 that names the setting to
  change. See [DEVICE_CONTROL.md](DEVICE_CONTROL.md).
- **Request size** — bodies over `limits.max_body_bytes` (default 1 MB; env `ZIVAWHITE_MAX_BODY_BYTES`, `0` = off) are
  refused with 413 before they are parsed, including chunked bodies that declare no length.
- **Generation limits** — `limits.max_new_tokens` (default: the model's context window, `block_size`) is the most
  tokens one request may ask for; more is a 400. A request that cannot get the model within `limits.lock_timeout_s`
  (default 30 s) is a 503 with `Retry-After`, so one long generation can no longer pile requests up in the threadpool
  until `/v1/health` stops answering. Streaming replies release the model as soon as it has finished writing: a
  slow or stalled client reading the stream no longer keeps everyone else waiting.
- **Response headers** — every response carries `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`; the web UI page also gets a strict `Content-Security-Policy` (scripts and styles
  only if their exact text is in the page, connections only to the server itself, no framing).
- **Binding / CORS** — the server listens on `127.0.0.1` unless you pass
  `--host 0.0.0.0` (set `ZIVAWHITE_SERVER_API_KEY` first; it warns if you don't).
  Browser calls from other sites are refused unless they come from localhost;
  `ZIVAWHITE_CORS_ORIGINS` takes a comma-separated list, or `*`.
- **Structured errors everywhere** — a validation failure, a missing
  route, an expected error (no checkpoint loaded, rate limited,
  unauthorized), and a genuinely unexpected exception all come back as
  `{"error": "..."}` with the right HTTP status, never a raw stack
  trace to the client (the real exception still goes to the server's
  own log).
- **Request logging** — every request logged with
  method/path/status/latency.

### Removed: the artificial limits on the model

(The three server-side protections above are a separate thing. What was removed below were arbitrary ceilings on what
the *model* could be asked; what exists now protects the *server* from one request starving the rest. The default
token cap is the model's own context window, not a number someone picked.)

`AskBody`/`GenBody` in `app.py` used to cap `query`/`prompt` at 8000
characters, `max_new_tokens` at 2000, and an image at ~11MB decoded.
None of those were real constraints of the model — they were arbitrary
API-layer ceilings. All three are gone: `min_length=1`/`ge=1` remain
(an empty query or a request for zero tokens is a no-op, not a
capacity limit), but there's no longer an upper bound beyond what the
model's own `block_size` and available memory actually allow. Verified
directly: a query of 18,000 characters and a `max_new_tokens=50000`
request are both accepted now where they'd previously have been
rejected at the API layer.
