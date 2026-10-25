"""
router.py - the routing brain around the local model.

ZivaWhite is ONE object, and everything it does is the local model
(ZivaGPT) plus read-only access to live web information. There is no
"wrapper" that delegates answer-writing to another company's AI model
-- there used to be one (Anthropic/OpenAI/Gemini/generic-REST
connectors), and it has been removed. Every answer this project
produces is written by its own trained weights, not routed out to
someone else's.

Every call to `.ask()` goes through the same routing brain:

    query -> decide route -> call backend -> answer

Routes:
  "local"   - answer from the local model only (fast, private, works
              fully offline, quality bounded by how much you've
              trained it).
  "web"     - do a live web search (+ optional page fetch) first, then
              have the local model write the answer grounded in those
              results (if it's been instruction-tuned; otherwise the
              raw retrieved text is returned honestly instead of
              having an untuned base model mangle it -- see _run_web).
  "auto"    (default) - the router decides per-query using cheap
              heuristics: explicit URLs and freshness-indicating
              language ("today", "latest", "current", a year, "who is
              the current ...", prices/scores/weather) push toward
              "web"; everything else tries "local". If no local model
              is loaded, "auto" falls back to "web" so there's still
              an honest attempt at an answer instead of a dead end. The
              reverse also holds: if an auto-routed web query retrieves
              nothing (no search backend configured, offline, no hits),
              the local model answers instead. An explicit route="web"
              reports the failure rather than quietly switching.

There is no vision/image input -- that capability only ever existed
because an API connector could see images; the local ZivaGPT model is
a byte-level text transformer with no vision capability at all, so
removing the API wrapper removes multimodal input as a real
consequence, not an oversight.

Self-training loop: whenever a "web" route pulls in text from a
source registered in data/sources.py as publicly reusable, that text
(with full provenance) is appended to data/queue/*.jsonl. Running
`python -m zivawhite.app ingest --commit` folds the queue into the
training shards, so the model can be re-trained on what it looked up
without a human manually curating a dataset. `python -m zivawhite.app
autolearn` runs that whole loop (pick a topic -> search -> queue ->
fold -> retrain) on its own, continuously -- see autolearn.py. No
source outside data/sources.py's registry is ever queued -- that is a
copyright/licensing gate, not a capability limit, and it is the one
restriction in this project that was deliberately kept rather than
removed: training on text without a license that permits it isn't
something this project will do regardless of what else changes.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .chat import generate_text
from .connectors import Connector, WebFetchConnector, WebSearchConnector
from .errors import ModelBusy
from .evaluate import repeated_ngram_rate
from .model import ZivaGPT
from .tokenizer import TokenizerMismatch, TokenizerNotFound, load_tokenizer, tokenizer_for_checkpoint

# Words that mean "this needs live information". Deliberately specific: bare
# "current" and "score" used to match "electric current" and "score a goal" and
# send physics/sports-practice questions to web search.
_FRESHNESS_PATTERNS = [
    r"\btoday\b", r"\bcurrently\b", r"\blatest\b", r"\bright now\b",
    r"\bcurrent (price|president|prime minister|ceo|champion|status|weather|time|temperature|version|events?)\b",
    r"\bthis (week|month|year)\b",
    r"\bweather\b", r"\b(live|final|latest|current) scores?\b", r"\bstock price\b", r"\bnews\b",
    r"\bwho is the (current|new)\b",
]
_URL_PATTERN = re.compile(r"https?://\S+")


def default_freshness_years(today: Optional[datetime.date] = None) -> Tuple[int, int]:
    """A year at or after last year reads as "recent": last year .. fourteen years ahead. It moves with the clock;
    a fixed 2025-2039 used to go stale. Override with router.freshness_years in the config."""
    year = (today or datetime.date.today()).year
    return year - 1, year + 14


def build_freshness_regex(years: Optional[Sequence[int]] = None, extra_patterns: Sequence[str] = ()) -> "re.Pattern[str]":
    """The 'this needs live information' pattern: the built-in phrases, a year window, and any extra regexes."""
    lo, hi = (int(years[0]), int(years[1])) if years else default_freshness_years()
    parts = list(_FRESHNESS_PATTERNS) + [r"\b(" + "|".join(str(y) for y in range(lo, hi + 1)) + r")\b"] + list(extra_patterns)
    return re.compile("|".join(parts), re.IGNORECASE)


_FRESHNESS_RE = build_freshness_regex()


@dataclass
class RouteDecision:
    route: str                 # "local" | "web"
    reason: str
    url: Optional[str] = None


@dataclass
class Answer:
    text: str
    route: str
    reason: str
    sources: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    latency_s: float = 0.0


class ZivaWhite:
    def __init__(
        self,
        model: Optional[ZivaGPT] = None,
        tokenizer=None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        connectors: Optional[Dict[str, Connector]] = None,
        confidence_floor: float = 0.35,
        queue_dir: str = "data/queue",
        tokenizer_dir: Optional[str] = None,
        lock_timeout_s: float = 30.0,
        max_new_tokens_cap: Optional[int] = None,
        freshness_years: Optional[Sequence[int]] = None,
        freshness_patterns: Sequence[str] = (),
        loop_threshold: float = 0.5,
    ):
        self.device = device
        self.model = model
        # Which tokenizer? One passed in is used as given. Otherwise the model's OWN tokenizer is found when its
        # checkpoint is loaded (load_local_model): the one saved next to it, then `tokenizer_dir` (the configured
        # one), then the bundled general-English one. Until a model is loaded this placeholder is all there is.
        self._tokenizer_pinned = tokenizer is not None
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer_note = "supplied by the caller" if tokenizer is not None else ""
        if tokenizer is None:
            try:
                tokenizer = load_tokenizer(tokenizer_dir)
                self.tokenizer_note = f"configured: {tokenizer_dir}" if tokenizer_dir else "bundled general-English tokenizer"
            except TokenizerNotFound as e:                  # a bad setting must not stop a model that carries its own
                tokenizer = load_tokenizer(None)
                self.tokenizer_note = f"bundled general-English tokenizer ({e})"
        self.tokenizer = tokenizer
        self.confidence_floor = confidence_floor
        self.loop_threshold = loop_threshold
        self.lock_timeout_s = lock_timeout_s
        self.max_new_tokens_cap = max_new_tokens_cap
        self._freshness_re = build_freshness_regex(freshness_years, freshness_patterns)
        self.queue_dir = queue_dir
        self.behavior_tuned = False  # set by load_local_model from the checkpoint's own metadata
        self.model_info: Dict[str, object] = {}  # what load_local_model actually loaded (shown by /v1/health)
        self._gen_lock = threading.Lock()  # one generation at a time: the server runs handlers in a threadpool
        # Default sampling for every local generation (override in configs/ziva.yaml -> `generation:`).
        # Defaults reproduce the behaviour from before these knobs existed. Small models loop a lot:
        # repetition_penalty ~1.1 and top_p ~0.9 are the usual first fixes.
        self.sampling: Dict[str, object] = {"temperature": 0.8, "top_k": 50, "top_p": None, "min_p": None,
                                            "repetition_penalty": 1.0}
        os.makedirs(queue_dir, exist_ok=True)

        if self.model is None and checkpoint_path:
            from .checkpoint import resolve_serving_checkpoint

            # resolve first: a run folder may hold only checkpoint_best.pt (e.g. copied from Colab)
            if os.path.exists(resolve_serving_checkpoint(checkpoint_path)):
                self.load_local_model(checkpoint_path)

        self.connectors: Dict[str, Connector] = connectors or {
            "web_search": WebSearchConnector(),
            "web_fetch": WebFetchConnector(),
        }

    # -- local model lifecycle -------------------------------------------------
    def load_local_model(self, checkpoint_path: str) -> None:
        from .checkpoint import load_model

        model, ckpt, device, src = load_model(checkpoint_path, device=self.device)
        if self._tokenizer_pinned:
            tokenizer, how = self.tokenizer, self.tokenizer_note
        else:
            try:
                tokenizer, how = tokenizer_for_checkpoint(src, ckpt, configured=self.tokenizer_dir)
            except (TokenizerMismatch, TokenizerNotFound) as e:
                raise RuntimeError(f"{src}: {e}") from e
        if model.cfg.vocab_size != tokenizer.vocab_size:
            raise RuntimeError(
                f"{src} was trained with a vocab of {model.cfg.vocab_size} tokens but the tokenizer in use has "
                f"{tokenizer.vocab_size}. Output would be garbage. If the tokenizer has 260 tokens, the "
                f"`tokenizers` package is missing (pip install tokenizers); otherwise pass the tokenizer this "
                f"checkpoint was trained with."
            )
        self.tokenizer, self.tokenizer_note = tokenizer, how
        self.model, self.device = model, device  # device is forced to "cpu" for int8 models
        # A plain pretrain/bootstrap checkpoint only ever saw raw corpus text --
        # it has no idea "hello" is a greeting to respond to, or how to follow
        # an instruction like "answer using only the information below." It
        # will confidently keep generating anyway (that's what a language
        # model does), so _run_local/_run_web use this to be honest about
        # what kind of output that actually is, and to decide whether to wrap
        # prompts in the chat template (see chat.py). Set when autolearn fine-tunes on
        # the datasets folder's chat/ conversations (train.py --behavior_tuned); False for a base model.
        extra = ckpt.get("extra", {})
        self.behavior_tuned = bool(extra.get("behavior_tuned", False))
        cfg = model.cfg
        self.model_info = {
            "file": src, "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "block_size": cfg.block_size,
            "step": ckpt.get("step"), "behavior_tuned": self.behavior_tuned,
            "quantized": bool(ckpt.get("quantized", False)),
            "tokenizer": self.tokenizer_note, "vocab_size": tokenizer.vocab_size,
        }

    def has_local_model(self) -> bool:
        return self.model is not None

    def sampling_kwargs(self, temperature=None, top_p=None, repetition_penalty=None) -> Dict[str, object]:
        """The configured sampling defaults with any per-request overrides (None = keep the default)."""
        kw = dict(self.sampling)
        for key, val in (("temperature", temperature), ("top_p", top_p), ("repetition_penalty", repetition_penalty)):
            if val is not None:
                kw[key] = val
        return kw

    # -- one generation at a time --------------------------------------------------------------------
    @contextlib.contextmanager
    def _generation_slot(self) -> Iterator[None]:
        """Hold the generation lock, but give up (ModelBusy -> HTTP 503) instead of queueing forever behind a long
        request: handlers run in a threadpool, and unbounded waiting would exhaust it and stall /v1/health too."""
        if not self._gen_lock.acquire(timeout=self.lock_timeout_s):
            raise ModelBusy(f"the model is busy with another request (waited {self.lock_timeout_s:g}s); try again shortly")
        try:
            yield
        finally:
            self._gen_lock.release()

    def new_token_limit(self) -> int:
        """The most tokens one request may ask the model to write: the configured cap, else one full context window."""
        if self.max_new_tokens_cap:
            return int(self.max_new_tokens_cap)
        block = getattr(getattr(self.model, "cfg", None), "block_size", None)
        return int(block) if block else 2048

    def chat_stream(self, messages, *, max_new_tokens: int = 256, temperature=None, top_p=None,
                    repetition_penalty=None, stop=(), info: Optional[dict] = None):
        """Reply to an OpenAI-style message list, yielding the text as it is written. A base (not chat-tuned) model
        has no chat format, so it simply continues the last user message.

        The model runs in a worker thread that pushes pieces into an unbounded queue, and holds the generation lock
        only while it is actually computing -- a slow or stalled client reading the stream cannot keep everyone
        else waiting. Closing the generator (client disconnected) stops the worker and frees the lock before
        close() returns. Nothing runs until the first piece is requested."""
        from .chat import prepare_chat_request, stream_text

        text, history, role = prepare_chat_request(messages)
        if not self.has_local_model():
            raise RuntimeError("no local checkpoint loaded")
        kw = self.sampling_kwargs(temperature, top_p, repetition_penalty)
        pieces: "queue.Queue[object]" = queue.Queue()
        cancel, done = threading.Event(), object()

        def worker() -> None:
            try:
                with self._generation_slot():
                    for piece in stream_text(self.model, self.tokenizer, text, chat=self.behavior_tuned, device=self.device,
                                             max_new_tokens=max_new_tokens, history=history if self.behavior_tuned else None,
                                             role=role, info=info, extra_stop=tuple(stop), temperature=kw["temperature"],
                                             top_k=kw["top_k"], top_p=kw["top_p"], min_p=kw["min_p"],
                                             repetition_penalty=kw["repetition_penalty"]):
                        if cancel.is_set():
                            break
                        pieces.put(piece)
            except BaseException as e:  # noqa: BLE001 - handed to the consumer, who re-raises it
                pieces.put(e)
            finally:
                pieces.put(done)

        thread = threading.Thread(target=worker, name="ziva-generate", daemon=True)
        thread.start()
        try:
            while True:
                item = pieces.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            cancel.set()
            thread.join()

    def chat(self, messages, **kw) -> Dict[str, object]:
        """Non-streaming twin of chat_stream: the whole reply plus token usage and finish_reason."""
        from .chat import build_chat_prompt, prepare_chat_request

        text, history, role = prepare_chat_request(messages)
        info: dict = {}
        reply = "".join(self.chat_stream(messages, info=info, **kw))
        prompt = build_chat_prompt(text, history, role) if self.behavior_tuned else text
        return {"text": reply, "prompt_tokens": len(self.tokenizer.encode(prompt, add_bos=True)),
                "completion_tokens": int(info.get("completion_tokens", 0)),
                "finish_reason": info.get("finish_reason", "stop")}

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, raw: bool = False) -> str:
        """Local model only, no routing. Returns just the newly generated text.
        Chat-tuned checkpoints get the chat template unless raw=True."""
        if not self.has_local_model():
            raise RuntimeError("no local model loaded")
        with self._generation_slot():
            kw = self.sampling_kwargs()
            return generate_text(self.model, self.tokenizer, prompt, chat=self.behavior_tuned and not raw,
                                 device=self.device, max_new_tokens=max_new_tokens, temperature=temperature,
                                 top_k=kw["top_k"], top_p=kw["top_p"], min_p=kw["min_p"],
                                 repetition_penalty=kw["repetition_penalty"])

    # -- routing -----------------------------------------------------------
    def _decide(self, query: str, route: str = "auto") -> RouteDecision:
        if route != "auto":
            return RouteDecision(route=route, reason="explicit override")

        url_match = _URL_PATTERN.search(query)
        if url_match:
            return RouteDecision(route="web", reason="query contains a URL", url=url_match.group(0))

        if self._freshness_re.search(query):
            return RouteDecision(route="web", reason="freshness-indicating language detected")

        if self.has_local_model():
            return RouteDecision(route="local", reason="local model available, no freshness signal")

        return RouteDecision(route="web", reason="no local model loaded; trying web search")

    # -- backends ------------------------------------------------------------
    def _run_local(self, query: str, max_new_tokens: int = 200, allow_web_fallback: bool = True) -> Answer:
        # Chat-tuned checkpoints are prompted in the format they were trained on
        # and stop when the reply ends; a base model just continues the text.
        stats: dict = {}
        kw = self.sampling_kwargs()
        with self._generation_slot():
            text = generate_text(self.model, self.tokenizer, query, chat=self.behavior_tuned, device=self.device,
                                 max_new_tokens=max_new_tokens, temperature=kw["temperature"], top_k=kw["top_k"],
                                 top_p=kw["top_p"], min_p=kw["min_p"],
                                 repetition_penalty=kw["repetition_penalty"], stats=stats)
        confidence = float(stats.get("mean_top1", 0.0))
        # Mean top-1 probability is HIGH for text stuck in a loop ("the the the ..."): the model is sure of every
        # next word because it is repeating itself. That is not a good answer, so looping text is capped at "unsure".
        words = text.split()
        looping = len(words) >= 16 and repeated_ngram_rate(words, 4) >= self.loop_threshold
        if looping:
            confidence = min(confidence, self.confidence_floor / 2)
        web = self.connectors.get("web_search")
        # allow_web_fallback=False when this call is itself the "local model
        # writing from web context" step inside _run_web -- falling back to web
        # FROM a web-triggered writer call would recurse (fallback -> _run_web ->
        # writer -> low confidence again -> fallback -> ...). Only the top-level
        # "local route answered with low confidence" case falls back; a writer
        # that's unsure about its own grounded answer just says so instead.
        if allow_web_fallback and confidence < self.confidence_floor and web is not None and web.available():
            web_answer = self._run_web(query)
            if web_answer.text:
                web_answer.reason = f"local confidence below floor ({confidence:.2f}); fell back to web -> {web_answer.reason}"
                web_answer.confidence = confidence
                return web_answer
        reason = ("local model" if self.behavior_tuned else
                  "local model (base/pretrain-only, not instruction-tuned -- this is raw text continuation "
                  "in the training corpus's style, not a direct answer; add conversations to the datasets "
                  "folder's chat/ directory for a model that actually responds to questions/greetings)")
        if confidence < self.confidence_floor:
            reason += f" (low confidence: {confidence:.2f}{', the output is repeating itself' if looping else ''})"
        return Answer(text=text, route="local", reason=reason, confidence=confidence)

    def _retrieve(self, query: str, url: Optional[str] = None) -> Tuple[str, List[str], Optional[str]]:
        """Fetch grounding text. Returns (context, sources, error); context is ""
        when nothing usable came back, with `error` saying why."""
        if url:
            page = self.connectors["web_fetch"].call(url)
            if page.ok and page.text:
                # The licence allow-list must judge the page that actually answered, not the address that was asked
                # for: a URL on an allow-listed site can redirect somewhere else, and its text is then not that
                # site's to license.
                final_url = page.meta.get("url") or url
                self._maybe_queue_for_training(final_url, page.text)
                return page.text, [final_url], None
            return "", [], page.error or "page returned no text"
        search = self.connectors["web_search"].call(query, max_results=5)
        if search.ok and search.meta.get("n_results", 1) > 0:
            return search.text, list(search.meta.get("urls", [])), None
        return "", [], search.error or "search returned no results"

    def _fit_context(self, context: str, query: str, max_new_tokens: int = 200) -> str:
        """Trim retrieved text so the writer prompt (template + context + question) plus
        room for the reply fits the model's context window. Without this a long page made
        generation fail outright."""
        block = self.model.cfg.block_size
        reserve = min(max_new_tokens, max(16, block // 4))
        overhead = len(self.tokenizer.encode(self._writer_prompt("", query), add_bos=True)) + 40  # + chat tags
        budget = max(0, block - reserve - overhead)
        ids = self.tokenizer.encode(context)
        return context if len(ids) <= budget else self.tokenizer.decode(ids[:budget])

    @staticmethod
    def _writer_prompt(context: str, query: str) -> str:
        return (
            f"Using ONLY the information below, answer the question concisely. "
            f"If the information doesn't answer it, say so.\n\nINFORMATION:\n{context}\n\nQUESTION: {query}"
        )

    def _run_web(self, query: str, url: Optional[str] = None, retrieved: Optional[tuple] = None) -> Answer:
        context, sources, error = retrieved if retrieved is not None else self._retrieve(query, url)

        # A base (pretrain-only) local model was never trained to follow an
        # instruction like this one -- it would just continue the prompt text in
        # its training corpus's style and we'd hand that back as a real answer.
        # Only use it as the writer once it's actually been instruction-tuned.
        if context and self.has_local_model() and self.behavior_tuned:
            writer_prompt = self._writer_prompt(self._fit_context(context, query), query)
            local = self._run_local(writer_prompt, allow_web_fallback=False)
            local.route, local.sources = "web", sources
            return local

        if context:
            return Answer(text=context, route="web",
                          reason="raw search context, no writer model available (add conversations "
                                 "to the datasets folder's chat/ directory so the local model can write from it)",
                          sources=sources)
        no_local = "no local model loaded"
        base_only = "local model is loaded but base/pretrain-only, not instruction-tuned"
        hint = (" Set SEARXNG_URL (or SEARCH_API_URL + SEARCH_API_KEY) in .env to enable web search."
                if error and "no search backend" in error else "")
        return Answer(text=f"No web results ({error}) and no usable local model to answer from.{hint}", route="web",
                      reason=f"no web results: {error} ({base_only if self.has_local_model() else no_local})",
                      sources=sources)

    # -- public API ------------------------------------------------------------
    def ask(self, query: str, route: str = "auto") -> Answer:
        t0 = time.time()
        decision = self._decide(query, route)
        if decision.route == "local" and not self.has_local_model():
            return Answer(text="No local model is loaded yet. Put training files in the datasets folder (see "
                               "datasets/README.md) and run `python -m zivawhite.app autolearn --once`, or pass "
                               "--checkpoint. `python -m zivawhite.app inspect` shows what would be loaded.",
                          route="local", reason="route=local requested but no local model is loaded",
                          latency_s=time.time() - t0)
        if decision.route == "local":
            answer = self._run_local(query)
        else:
            retrieved = self._retrieve(query, decision.url)
            if not retrieved[0] and route == "auto" and self.has_local_model():
                # Auto-routing guessed "needs the web" but the web gave nothing (no search
                # backend configured, offline, no hits). Answering from the local model
                # beats a dead end; an explicit route="web" still reports the failure.
                answer = self._run_local(query, allow_web_fallback=False)
                answer.reason = f"web retrieval failed ({retrieved[2]}); answered from the local model -> {answer.reason}"
            else:
                answer = self._run_web(query, url=decision.url, retrieved=retrieved)
        # Don't just clobber whatever the backend actually reports: it knows
        # things _decide() couldn't (a low-confidence local answer that fell
        # back to web -- a real route change decision.reason knows nothing
        # about; whether the local model backing an answer is actually
        # instruction-tuned). Combine them instead of overwriting, and only
        # when the backend didn't itself change the route the routing
        # decision was based on.
        if answer.route == decision.route and answer.reason != decision.reason:
            answer.reason = f"{decision.reason} -> {answer.reason}"
        answer.latency_s = time.time() - t0
        return answer

    # -- self-training loop --------------------------------------------------
    def _maybe_queue_for_training(self, url: str, text: str) -> None:
        from .data.sources import is_allow_listed

        if not text or not is_allow_listed(url):
            return
        record = {"url": url, "text": text, "queued_at": time.time()}
        path = os.path.join(self.queue_dir, "web_ingest.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
