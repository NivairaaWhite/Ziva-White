"""ZivaWhite.chat / chat_stream: OpenAI-style messages in, text out, generation lock never leaked.
Uses a scripted model, so it needs neither weights nor (when torch is absent) torch itself."""
import sys
import threading
import time

import pytest

from zivawhite.errors import ModelBusy

import torchless as ts   # scripted model + a tiny stand-in for torch when it is not installed
from zivawhite.tokenizer import ByteLevelTokenizer

if not ts.HAVE_TORCH:        # router.py imports ZivaGPT only as a type: give it a stub just long enough to import it
    import types

    from zivawhite.modelconfig import ModelConfig

    _stub = types.ModuleType("zivawhite.model")
    _stub.ZivaGPT, _stub.ModelConfig = object, ModelConfig
    _had = sys.modules.get("zivawhite.model")
    sys.modules["zivawhite.model"] = _stub
    try:
        from zivawhite.router import ZivaWhite  # noqa: E402
    finally:                  # never leave the stub (or the router bound to it) behind: other tests must see the real situation
        if _had is None:
            sys.modules.pop("zivawhite.model", None)
        else:
            sys.modules["zivawhite.model"] = _had
        sys.modules.pop("zivawhite.router", None)   # our ZivaWhite class keeps working: it holds its own globals
else:
    from zivawhite.router import ZivaWhite  # noqa: E402


def setup_function(function):
    ts.install()


def teardown_function(function):
    ts.uninstall()


TOK = ByteLevelTokenizer()


def make(reply="Hello there!", tuned=True):
    model = ts.ScriptedModel(TOK.encode(reply))
    z = ZivaWhite(model=model, tokenizer=TOK, device="cpu", queue_dir="/tmp/ziva_queue_test")
    z.behavior_tuned = tuned
    return z, model


MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]


def test_chat_returns_text_usage_and_finish_reason():
    z, _ = make()
    out = z.chat(MSGS, max_new_tokens=50)
    assert out["text"] == "Hello there!" and out["finish_reason"] == "stop"
    assert out["completion_tokens"] == len("Hello there!") and out["prompt_tokens"] > len("hi")


def test_the_system_prompt_and_history_reach_the_model_as_their_own_roles():
    z, model = make()
    z.chat([{"role": "system", "content": "RULES"}, {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], max_new_tokens=5)
    prompt = TOK.decode(model.last_prompt_ids)
    assert prompt.index("<|system|>\nRULES") < prompt.index("<|user|>\na") < prompt.index("<|assistant|>\nb") < prompt.index("<|user|>\nc")
    assert prompt.rstrip().endswith("<|assistant|>")


def test_stream_pieces_join_to_the_same_text_as_the_non_streaming_call():
    z, _ = make("The answer is 42.")
    pieces = list(z.chat_stream(MSGS, max_new_tokens=50))
    assert len(pieces) > 1 and "".join(pieces) == z.chat(MSGS, max_new_tokens=50)["text"] == "The answer is 42."


def test_the_generation_lock_is_free_before_the_first_piece_and_after_an_early_close():
    z, _ = make("some longer reply to stream")
    gen = z.chat_stream(MSGS, max_new_tokens=50)
    assert not z._gen_lock.locked()                    # nothing runs until the first piece is requested
    next(gen)
    gen.close()                                        # client disconnected mid-stream
    assert not z._gen_lock.locked()                    # released before close() returned
    list(z.chat_stream(MSGS, max_new_tokens=50))       # ... and a following request is not blocked
    assert not z._gen_lock.locked()


class GatedModel(ts.ScriptedModel):
    """Writes twenty-five tokens, then waits on `gate` before the rest, so a test can freeze generation mid-reply."""

    def __init__(self, script):
        super().__init__(script)
        self.gate, self.entered = threading.Event(), threading.Event()

    def generate_stream(self, idx, max_new_tokens, stop_fn=None, **kw):
        seq = ts.row_of(idx)
        for n, tok in enumerate(self.script[:max_new_tokens]):
            if n == 25:                      # past the ~12 characters stream_text holds back for stop-marker safety
                self.entered.set()
                self.gate.wait(10)
            seq.append(tok)
            yield tok


def _wait_until(cond, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_the_lock_is_held_while_the_model_computes_but_not_while_a_slow_client_reads():
    """Review item: one lock for the whole stream let a slow client block everyone. Now the lock covers only the
    generation itself."""
    reply = "a slow client reads this reply, which is long enough to stream"
    model = GatedModel(TOK.encode(reply))
    z = ZivaWhite(model=model, tokenizer=TOK, device="cpu", queue_dir="/tmp/ziva_queue_test")
    z.behavior_tuned = True
    gen = z.chat_stream(MSGS, max_new_tokens=200)
    first = next(gen)                                   # the client has read ONE piece and then stops reading
    assert model.entered.wait(3) and z._gen_lock.locked()      # the model is mid-reply: the lock is rightly held
    model.gate.set()                                    # the model finishes ...
    assert _wait_until(lambda: not z._gen_lock.locked())       # ... and frees the model although the client has read almost nothing
    # a second request is served while the first client is still holding its half-read stream
    other = ZivaWhite.chat(z, MSGS, max_new_tokens=200)
    assert other["text"] == reply
    assert first + "".join(gen) == reply                  # and the slow client still gets every piece


def test_a_request_that_cannot_get_the_model_in_time_is_told_it_is_busy():
    z, _ = make()
    z.lock_timeout_s = 0.05
    assert z._gen_lock.acquire(timeout=1)               # somebody else is generating
    try:
        with pytest.raises(ModelBusy):
            z.generate("hello")
        with pytest.raises(ModelBusy):
            list(z.chat_stream(MSGS))
    finally:
        z._gen_lock.release()
    assert z.chat(MSGS)["text"] == "Hello there!"       # and it works again once the model is free


def test_an_error_inside_generation_reaches_the_caller_and_frees_the_lock():
    class Boom(ts.ScriptedModel):
        def generate_stream(self, idx, max_new_tokens, stop_fn=None, **kw):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover - makes this a generator
    z = ZivaWhite(model=Boom([]), tokenizer=TOK, device="cpu", queue_dir="/tmp/ziva_queue_test")
    z.behavior_tuned = True
    with pytest.raises(RuntimeError, match="kaboom"):
        list(z.chat_stream(MSGS))
    assert not z._gen_lock.locked()


def test_the_new_token_limit_is_configurable_and_defaults_to_one_context_window():
    z, model = make()
    assert z.new_token_limit() == model.cfg.block_size
    z.max_new_tokens_cap = 64
    assert z.new_token_limit() == 64


def test_per_request_sampling_overrides_the_configured_defaults():
    z, model = make()
    z.sampling.update({"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1})
    z.chat(MSGS, max_new_tokens=3)
    assert (model.last_kw["temperature"], model.last_kw["top_p"], model.last_kw["repetition_penalty"]) == (0.7, 0.9, 1.1)
    z.chat(MSGS, max_new_tokens=3, temperature=0.2, repetition_penalty=1.3)
    assert (model.last_kw["temperature"], model.last_kw["top_p"], model.last_kw["repetition_penalty"]) == (0.2, 0.9, 1.3)


def test_stop_sequences_from_the_client_cut_the_reply():
    z, _ = make("one two STOP three")
    assert z.chat(MSGS, max_new_tokens=50, stop=["STOP"])["text"] == "one two"


def test_a_base_model_just_continues_the_last_message_without_chat_tags():
    z, model = make("continued text", tuned=False)
    out = z.chat([{"role": "system", "content": "ignored"}, {"role": "user", "content": "Once upon a"}], max_new_tokens=50)
    assert out["text"] == "continued text"
    assert "<|" not in TOK.decode(model.last_prompt_ids) and "Once upon a" in TOK.decode(model.last_prompt_ids)


def test_bad_requests_and_a_missing_model_are_reported_not_swallowed():
    z, _ = make()
    with pytest.raises(ValueError, match="last message"):
        z.chat([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    with pytest.raises(ValueError, match="role"):
        z.chat([{"role": "wizard", "content": "q"}])
    empty = ZivaWhite(model=None, tokenizer=TOK, device="cpu", queue_dir="/tmp/ziva_queue_test")
    with pytest.raises(RuntimeError, match="no local checkpoint"):
        list(empty.chat_stream(MSGS))


# ------------------------------------------------------------------ tool results and the agent loop
class MultiScriptModel(ts.ScriptedModel):
    """Replays a DIFFERENT scripted reply on each generation call (the agent loop calls the model repeatedly)."""

    def __init__(self, replies):
        super().__init__([])
        self.replies, self.prompts = list(replies), []

    def generate_stream(self, idx, max_new_tokens, stop_fn=None, **kw):
        self.script = TOK.encode(self.replies.pop(0))
        self.prompts.append(TOK.decode(ts.row_of(idx)))
        yield from super().generate_stream(idx, max_new_tokens, stop_fn=stop_fn, **kw)


def test_a_tool_result_can_be_the_last_message_and_is_rendered_as_a_tool_turn():
    z, model = make("It is 42.")
    out = z.chat([{"role": "user", "content": "6*7?"}, {"role": "assistant", "content": "<tool_call>x</tool_call>"},
                  {"role": "tool", "content": "42"}], max_new_tokens=20)
    assert out["text"] == "It is 42."
    assert TOK.decode(model.last_prompt_ids).rstrip().endswith("<|tool|>\n42\n<|assistant|>")


def test_the_agent_loop_drives_the_real_router_and_stops_generation_at_the_closing_tag():
    from zivawhite.tools import TOOL_CALL_CLOSE, default_registry, run_agent

    call = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call> and then a hallucinated result 99'
    model = MultiScriptModel([call, "6 times 7 is 42."])
    z = ZivaWhite(model=model, tokenizer=TOK, device="cpu", queue_dir="/tmp/ziva_queue_test")
    z.behavior_tuned = True
    chat_fn = lambda msgs: z.chat(msgs, max_new_tokens=200, stop=[TOOL_CALL_CLOSE])["text"]  # noqa: E731
    res = run_agent(chat_fn, "What is 6*7?", default_registry())
    assert res.text == "6 times 7 is 42." and res.stopped_because == "answered"
    assert res.steps == [{"tool": "calculator", "arguments": {"expression": "6*7"}, "result": "42"}]   # not 99: text after the tag was cut
    first, second = model.prompts
    assert "<|system|>\nYou can use tools" in first and first.rstrip().endswith("<|assistant|>")
    assert "<|assistant|>\n<tool_call>" in second and second.rstrip().endswith("<|tool|>\n42\n<|assistant|>")


def test_the_last_message_must_still_be_a_user_or_tool_turn():
    z, _ = make()
    with pytest.raises(ValueError, match="'user' or 'tool'"):
        z.chat([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    with pytest.raises(ValueError, match="empty"):
        z.chat([{"role": "user", "content": "q"}, {"role": "tool", "content": "  "}])
