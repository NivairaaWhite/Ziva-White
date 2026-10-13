from zivawhite.connectors.base import Connector, ConnectorResult
from zivawhite.router import ZivaWhite


class FakeConnector(Connector):
    def __init__(self, name, ok_available, reply="fake reply"):
        self.name = name
        self._available = ok_available
        self.reply = reply
        self.calls = []

    def available(self):
        return self._available

    def call(self, prompt, **kw):
        self.calls.append(prompt)
        return ConnectorResult(text=self.reply, source=self.name)


def make_router(tmp_path, **connector_overrides):
    connectors = {
        "web_search": FakeConnector("web_search", False),
        "web_fetch": FakeConnector("web_fetch", False),
    }
    connectors.update(connector_overrides)
    return ZivaWhite(connectors=connectors, queue_dir=str(tmp_path))


def test_routes_freshness_query_to_web(tmp_path):
    web = FakeConnector("web_search", True, reply="- some snippet [http://x]")
    mw = make_router(tmp_path, web_search=web)

    answer = mw.ask("what's the weather today in Paris")
    assert answer.route == "web"
    assert web.calls, "web_search should have been called for a freshness query"


def test_auto_routes_to_web_when_no_local_model_loaded(tmp_path):
    """No API wrapper exists anymore to fall back to -- with no local model,
    'auto' should still make an honest attempt via web search rather than
    dead-ending."""
    web = FakeConnector("web_search", True, reply="- some snippet [http://x]")
    mw = make_router(tmp_path, web_search=web)

    answer = mw.ask("explain how photosynthesis works")
    assert answer.route == "web"
    assert "no local model loaded" in answer.reason


def test_explicit_route_override(tmp_path):
    web = FakeConnector("web_search", True, reply="- snippet [http://x]")
    mw = make_router(tmp_path, web_search=web)

    answer = mw.ask("hello there", route="web")
    assert answer.route == "web"
    assert web.calls


def _tiny_local_router(tmp_path, behavior_tuned: bool, **connector_overrides):
    """A router with a real (but tiny, untrained) local model attached, for
    testing the base-vs-instruction-tuned honesty logic without needing to
    actually train and round-trip a checkpoint through disk."""
    from zivawhite.model import ModelConfig, ZivaGPT
    from zivawhite.tokenizer import ByteLevelTokenizer

    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=256)
    mw = make_router(tmp_path, **connector_overrides)
    mw.model = ZivaGPT(cfg).to("cpu")
    mw.model.eval()
    mw.tokenizer = ByteLevelTokenizer()
    mw.behavior_tuned = behavior_tuned
    return mw


def test_base_model_reason_is_honest_about_not_being_instruction_tuned(tmp_path):
    mw = _tiny_local_router(tmp_path, behavior_tuned=False)
    answer = mw.ask("hello")
    assert answer.route == "local"
    assert "base/pretrain-only" in answer.reason
    assert "not instruction-tuned" in answer.reason


def test_tuned_model_reason_has_no_base_model_caveat(tmp_path):
    mw = _tiny_local_router(tmp_path, behavior_tuned=True)
    answer = mw.ask("hello")
    assert answer.route == "local"
    assert "base/pretrain-only" not in answer.reason


def test_ask_keeps_backends_specific_reason_instead_of_only_routing_rationale(tmp_path):
    """Regression test: ask() used to unconditionally overwrite whatever the
    backend reported with just _decide()'s routing rationale, so more
    specific/accurate outcomes (like this base-model caveat, or a
    confidence-based API fallback) never actually reached the caller."""
    mw = _tiny_local_router(tmp_path, behavior_tuned=False)
    answer = mw.ask("hello")
    assert "local model available, no freshness signal" in answer.reason  # the routing rationale...
    assert "base/pretrain-only" in answer.reason  # ...combined with, not replaced by, the outcome


def test_web_route_does_not_use_untuned_local_model_as_rag_writer(tmp_path):
    """A base model was never trained to follow a RAG instruction prompt --
    using it as the "writer" just returns text-continuation garbage dressed
    up as an answer. Confirm the raw search context comes back instead."""
    web = FakeConnector("web_search", True, reply="some search snippet text")
    mw = _tiny_local_router(tmp_path, behavior_tuned=False, web_search=web)

    answer = mw.ask("what is the latest news today")
    assert answer.route == "web"
    assert answer.text == "some search snippet text"
    assert "no writer model available" in answer.reason


def test_web_route_does_use_tuned_local_model_as_rag_writer(tmp_path):
    web = FakeConnector("web_search", True, reply="some search snippet text")
    mw = _tiny_local_router(tmp_path, behavior_tuned=True, web_search=web)

    answer = mw.ask("what is the latest news today")
    assert answer.route == "web"
    assert answer.text != "some search snippet text"  # local model actually ran on the RAG prompt


def test_writer_low_confidence_does_not_recurse_into_another_web_fallback(tmp_path):
    """Regression test: the writer step inside _run_web calls _run_local()
    on an untrained model, which is always low-confidence. Before
    allow_web_fallback existed, that low confidence triggered ANOTHER web
    fallback, which called the writer again, which was low-confidence
    again... an unbounded recursion that only ever stopped by crashing
    once the ever-growing prompt exceeded block_size. A single web_search
    call for the whole `ask()` is the correct behavior."""
    web = FakeConnector("web_search", True, reply="some search snippet text")
    mw = _tiny_local_router(tmp_path, behavior_tuned=True, web_search=web)

    mw.ask("what is the latest news today")
    assert len(web.calls) == 1
