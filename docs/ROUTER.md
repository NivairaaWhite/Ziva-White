<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
How a question is routed between the local model and live web, and what the web may be used for.

# The router — local model + live web, nothing else

`router.py` decides per query:

- a URL in the query, or freshness language ("today", "latest", "right now",
  a year, "weather", "live score", "current price/president/...", "who is the
  current...") → **web**: search, optionally fetch a page, then have the local
  model (if it's been instruction-tuned) write the answer grounded in what was
  actually retrieved. If the web returns nothing (no search backend configured,
  offline), the local model answers instead of a dead end.
- otherwise, if a local checkpoint is loaded → **local**. The router also checks
  its own confidence (mean top-1 probability over the tokens it generated) and
  falls back to web search if that's below a floor and a search backend is
  configured. That number is a rough signal, not a measure of correctness.
- no local model → **web**, if a search backend is configured.
- nothing configured at all → still returns an honest answer (raw search
  context, or a plain statement that no backend is available) instead of crashing.

Force a route explicitly with `route="local"|"web"` instead of `"auto"` any time.
Chat-tuned checkpoints are prompted in their chat format and stop when the reply
ends; a base checkpoint just continues the text (the response says so).

## Removed: the AI-API wrapper

This project used to be, honestly, two things wearing one name: a
real local model, plus a wrapper that could delegate answer-writing to
Anthropic, Gemini, any OpenAI-compatible endpoint, or any other REST
API via a generic connector. That wrapper — `connectors/
anthropic_connector.py`, `gemini_connector.py`, `openai_connector.py`,
`generic_connector.py`, `tools.py`'s function-calling loop, and the
`"api"` route in `router.py` — has been removed. Every answer this
project produces now comes from its own trained weights. What's left
in `connectors/` is exactly two things, both read-only and neither an
AI model: `web_search_connector.py` and `web_fetch_connector.py`, used
for grounding a `web`-routed answer and for gathering text `autolearn`
trains on — never for having another company's model write the
answer.

What this costs you, plainly: no more automatic fallback to a strong
external model when the local one is weak or untrained, no more
"any REST API in five lines of `.env`," and no multimodal image input
(that only ever worked because an external API could see images — see
the capability table above). What it buys you: `ask()` never sends
your query to a third party, `.env` no longer needs to be a list of
other companies' API keys, and every answer's quality is a true
reflection of *this* model, not whichever backend happened to answer
that query.

### "Search the web without being limited"

Commercial search APIs cap you (Google's Custom Search API: 100 free queries/day,
then billed — and closed to new customers, switched off on 2027-01-01; Bing's
search API no longer exists, Microsoft shut it down on 2025-08-11) — that's a
real limit self-hosting routes around. **SearxNG**, a free, open-source metasearch engine, fans a
query out to dozens of real engines and hands back JSON. Point
`SEARXNG_URL` at your own instance and there's no per-query quota,
because it's your infrastructure:

```bash
docker run -d --name searxng -p 8888:8888 searxng/searxng
echo "SEARXNG_URL=http://localhost:8888" >> .env
```

`web_search_connector.py` uses it as the primary backend and falls
back to a commercial API only if you'd rather not run the container.
Every provider answers in its own shape and takes its key its own way, so there is one adapter each; name yours in
`SEARCH_API_PROVIDER` (or let it be recognised from `SEARCH_API_URL`):

| `SEARCH_API_PROVIDER` | Provider | Key goes in |
|---|---|---|
| `brave` (recommended) | Brave Search API | the `X-Subscription-Token` header |
| `serpapi` | SerpAPI (Google results) | an `api_key` URL parameter (their only option) |
| `google_cse` | Google Custom Search JSON API (also needs `SEARCH_API_CX`) | a `key` URL parameter (their only option) |
| `generic` | a proxy you run, answering `{"results": [{"title", "url", "snippet"}]}` | `Authorization: Bearer <key>` |

A provider that wants the key in the URL puts it where logs like to copy it, so this connector never prints or
returns a URL it sent a key in: a failure is reported as e.g. "HTTP 403 from the search provider", and any trace
of the key is scrubbed from what is left.

### Training itself on public data — the actual guardrail

"Train on online data that's allowed to be used by the public" is
implemented as a literal allow-list, not a vague intention:
`data/sources.py` is a `REGISTRY` of domains with a license note
each (Project Gutenberg — public domain; Wikipedia/Wikisource — CC
BY-SA; arXiv abstracts; specific `.gov` public-domain sources). This
is the one restriction in this project that stayed in place through
every other round of removing limits — it's a copyright/licensing
safeguard, not an arbitrary capability limit, and training on text
without a license that permits it isn't something this project will do
regardless of what else changes. Two independent gates check against
it before anything reaches a training shard:

1. The router's opportunistic loop: when a `web` route fetches a
   page from an allow-listed domain, that text (with URL + license)
   is appended to `data/queue/*.jsonl`. Nothing from an unlisted
   domain is ever queued, even though it was fine to *fetch and read
   once* to answer a question — reading and training are separate
   permissions on purpose.
2. The optional **web intake** in `autolearn.py` (`autolearn.web.enabled`, off by default). It reads
   YOUR topics (`_sources/topics.txt`) and URLs (`_sources/urls.txt`); none ship. It keeps only
   allow-listed result URLs, fetches them, and saves each page as an ordinary file under
   `datasets/text/web/`, with its URL, source and license recorded in
   `datasets/_sources/web_provenance.jsonl`. From there it is just another dataset file: visible,
   deletable, and trained through the same pipeline. No AI model picks the topics or decides what is
   worth fetching (that would quietly bring the wrapper back). The license gate applies at fetch and
   again at export, and no setting turns it off. It does **not** apply to files you place in
   `datasets/` yourself: that data is yours to vouch for.

Add a source by adding an entry to `REGISTRY` with an honest license
note — don't add anything paywalled or ToS-restricted, since that
defeats the point of having the gate at all.
