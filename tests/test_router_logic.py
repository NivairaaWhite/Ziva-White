"""Router decisions and fallbacks, with the model faked out (no torch work), so these pin down
the routing logic itself: freshness regex, web->local fallback, confidence fallback, no recursion."""
from types import SimpleNamespace

import zivawhite.router as router_mod
from zivawhite.connectors.base import Connector, ConnectorResult
from zivawhite.router import ZivaWhite
from zivawhite.tokenizer import ByteLevelTokenizer


class FakeWeb(Connector):
    name = "web_search"

    def __init__(self, text="", ok=True, n_results=None, configured=True):
        self.text, self.ok, self.n_results, self.configured, self.calls = text, ok, n_results, configured, 0

    def available(self):
        return self.configured

    def call(self, prompt, **kw):
        self.calls += 1
        meta = {} if self.n_results is None else {"n_results": self.n_results, "urls": ["https://example.org/a"]}
        return ConnectorResult(text=self.text, source=self.name, ok=self.ok, meta=meta,
                               error=None if self.ok else "boom")


def _router(monkeypatch, tmp_path, *, tuned, web, confidence=0.9, reply="model reply", block_size=256):
    """A router whose 'model' is a stub: generate_text is replaced to return `reply` with `confidence`."""
    mw = ZivaWhite(model=SimpleNamespace(cfg=SimpleNamespace(block_size=block_size)), tokenizer=ByteLevelTokenizer(),
                   connectors={"web_search": web, "web_fetch": FakeWeb()}, queue_dir=str(tmp_path / "q"))
    mw.behavior_tuned = tuned
    mw.seen = []

    def fake_generate_text(model, tok, text, *, chat, stats=None, **kw):
        mw.seen.append((text, chat))
        if stats is not None:
            stats["mean_top1"] = confidence
        return reply

    monkeypatch.setattr(router_mod, "generate_text", fake_generate_text)  # undone after each test
    return mw


def test_physics_and_practice_questions_are_not_sent_to_web():
    for q in ["what is electric current", "how do I score a goal", "explain the current in a circuit",
              "what is a good score on a test"]:
        assert not router_mod._FRESHNESS_RE.search(q), q


def test_genuine_freshness_questions_still_go_to_web():
    for q in ["what's the weather today", "latest news", "who is the current president", "live score of the match",
              "current price of gold", "what happened this week", "who won in 2026"]:
        assert router_mod._FRESHNESS_RE.search(q), q


def test_chat_template_is_used_only_for_a_chat_tuned_model(tmp_path, monkeypatch):
    tuned = _router(monkeypatch, tmp_path, tuned=True, web=FakeWeb(configured=False))
    tuned.ask("hello", route="local")
    assert tuned.seen == [("hello", True)]
    base = _router(monkeypatch, tmp_path, tuned=False, web=FakeWeb(configured=False))
    answer = base.ask("hello", route="local")
    assert base.seen == [("hello", False)]
    assert "not instruction-tuned" in answer.reason     # still honest that it's raw continuation


def test_route_local_without_a_model_says_so_instead_of_searching(tmp_path):
    web = FakeWeb(text="irrelevant")
    mw = ZivaWhite(tokenizer=ByteLevelTokenizer(), connectors={"web_search": web, "web_fetch": FakeWeb()},
                   queue_dir=str(tmp_path / "q"))
    answer = mw.ask("hello", route="local")
    assert "No local model" in answer.text and web.calls == 0


def test_auto_web_with_no_results_falls_back_to_the_local_model(tmp_path, monkeypatch):
    mw = _router(monkeypatch, tmp_path, tuned=True, web=FakeWeb(text="", ok=False))      # e.g. no search backend configured
    answer = mw.ask("what is the latest news today")                          # freshness -> web route
    assert answer.route == "local" and answer.text == "model reply"
    assert "web retrieval failed" in answer.reason


def test_explicit_web_route_reports_the_failure_instead_of_switching(tmp_path, monkeypatch):
    mw = _router(monkeypatch, tmp_path, tuned=True, web=FakeWeb(text="", ok=False))
    answer = mw.ask("hello", route="web")
    assert answer.route == "web" and "No web results" in answer.text and mw.seen == []


def test_web_results_are_written_up_by_a_tuned_local_model_with_trimmed_context(tmp_path, monkeypatch):
    mw = _router(monkeypatch, tmp_path, tuned=True, web=FakeWeb(text="X" * 20000, n_results=3), block_size=256)
    answer = mw.ask("what is the latest news today")
    assert answer.route == "web" and answer.sources == ["https://example.org/a"]
    prompt, chat = mw.seen[0]
    assert chat is True
    assert len(mw.tokenizer.encode(prompt)) < 256        # context was cut to leave room (used to overflow the window)
    assert prompt.rstrip().endswith("QUESTION: what is the latest news today")


def test_base_model_is_not_used_as_the_rag_writer(tmp_path, monkeypatch):
    mw = _router(monkeypatch, tmp_path, tuned=False, web=FakeWeb(text="search context", n_results=1))
    answer = mw.ask("what is the latest news today")
    assert answer.text == "search context" and mw.seen == []


def test_low_confidence_local_answer_falls_back_to_web_once_without_recursing(tmp_path, monkeypatch):
    web = FakeWeb(text="grounded text", n_results=1)
    mw = _router(monkeypatch, tmp_path, tuned=True, web=web, confidence=0.01)
    answer = mw.ask("explain how photosynthesis works")                       # no freshness words -> local first
    assert answer.route == "web" and "fell back to web" in answer.reason
    assert web.calls == 1                                                      # one search, not an endless loop
    assert len(mw.seen) == 2                                                   # local attempt + the web writer call


def test_low_confidence_without_a_search_backend_keeps_the_local_answer(tmp_path, monkeypatch):
    mw = _router(monkeypatch, tmp_path, tuned=True, web=FakeWeb(configured=False), confidence=0.01)
    answer = mw.ask("explain how photosynthesis works")
    assert answer.route == "local" and "low confidence" in answer.reason


def test_a_folder_with_only_checkpoint_best_is_loaded(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "checkpoint_best.pt").write_bytes(b"x")
    loaded = []
    monkeypatch.setattr(ZivaWhite, "load_local_model", lambda self, path: loaded.append(path))
    ZivaWhite(tokenizer=ByteLevelTokenizer(), checkpoint_path=str(tmp_path / "run" / "checkpoint.pt"),
              connectors={"web_search": FakeWeb(), "web_fetch": FakeWeb()}, queue_dir=str(tmp_path / "q"))
    assert loaded == [str(tmp_path / "run" / "checkpoint.pt")]
