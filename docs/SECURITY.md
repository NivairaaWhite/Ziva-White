# Security notes

Back to the [README](../README.md). This is what protects what, in one place. The detail is in the linked pages.

**The model of the threat.** By default the server listens on `127.0.0.1` and needs no key: it assumes only you can
reach it. The risks are then (1) other web pages open in *your* browser, and (2) anything you expose beyond your
own machine.

| Protection | Defends against | Where |
|---|---|---|
| Device control only from loopback or with the API key, never with a foreign `Origin` | a web page pressing buttons on your machine | [DEVICE_CONTROL.md](DEVICE_CONTROL.md) |
| `Host` header allow-list on every request | DNS rebinding (a page re-pointing its domain at `127.0.0.1` to look "same-origin") | [DEVICE_CONTROL.md](DEVICE_CONTROL.md) |
| Conversation ids issued by the server, namespaced per transport | answering someone else's "are you sure?" | [DEVICE_CONTROL.md](DEVICE_CONTROL.md) |
| The confirmation is re-checked against the caller's rights | completing an action another caller started | [DEVICE_CONTROL.md](DEVICE_CONTROL.md) |
| Optional API key, rate limit, wrong-key lock-out | network callers, key guessing | [HTTP_API.md](HTTP_API.md) |
| Body-size limit, token cap, busy timeout | one request starving the rest | [HTTP_API.md](HTTP_API.md) |
| `nosniff`, frame-deny, hash-based Content-Security-Policy on the web UI | injected scripts, framing | [HTTP_API.md](HTTP_API.md) |
| Checkpoints load with `weights_only=True`; pickle only if `ZIVAWHITE_ALLOW_PICKLE=1` | a downloaded `.pt` file running code | below |
| URL checks on every fetch hop, including `robots.txt` | the fetcher being bounced to an internal address (SSRF) | [ROUTER.md](ROUTER.md) |
| Search API keys scrubbed from error messages | keys leaking into logs and answers | [ROUTER.md](ROUTER.md) |
| Commands are argument lists, never shell strings; a fixed grammar, never model output | injection through text | [DEVICE_CONTROL.md](DEVICE_CONTROL.md) |

## Model files

Training writes `.pt` checkpoints. They are read with torch's `weights_only` loader, which can only produce
tensors, numbers, strings and containers of them, so a file that was shared with you cannot execute code when
loaded. A file that needs full pickle is refused with an explanation. The one thing that used to need it, the
int8 file from `compress`, is now stored as plain tensors (int8 weights plus a scale each) and loads safely.
`ZIVAWHITE_ALLOW_PICKLE=1` re-enables pickle for a file you made yourself; leave it off otherwise.

## What is *not* protected

* **The confirmation is not an access check.** "Are you sure?" guards against mishearing. Who may control the device
  is decided before it, and checked again after it.
* **A local process on your machine** is "local". Anything that can make requests from `127.0.0.1` can use the
  device-control endpoint when no key is set. Set `ZIVAWHITE_SERVER_API_KEY` if other users or untrusted programs
  share the machine.
* **The rate limiter and lock-out are in memory, per process.** Behind several workers or a reverse proxy, put a real
  limiter at the edge and set `ZIVAWHITE_TRUST_PROXY=1` only if the proxy sets `X-Forwarded-For` honestly.
* **Fetching a hostname resolves it twice** (once to check, once to connect), so a hostile DNS server can still race
  the check. Don't point the fetcher at hostile networks from a machine that can reach sensitive internal ones.
