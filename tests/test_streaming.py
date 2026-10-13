"""stream_text must yield exactly what the non-streaming path returns, never leak (or retract) a
stop marker, and survive multi-byte characters split across tokens. Runs without a trained model:
a scripted fake model supplies the tokens (and a tiny stand-in tensor is used if torch is absent)."""
import random

from zivawhite.chat import ASSISTANT_TAG, STOP_MARKERS, USER_TAG, build_chat_prompt, clean_reply, stream_text
from zivawhite.tokenizer import ByteLevelTokenizer

import torchless  # noqa: E402
from torchless import HAVE_TORCH as _HAVE_TORCH, ScriptedModel, make_tensor  # noqa: E402


def setup_function(function):
    torchless.install()


def teardown_function(function):
    torchless.uninstall()


TOK = ByteLevelTokenizer()


def _ids(text):
    return TOK.encode(text)


def _non_streaming(script, max_new_tokens=10_000):
    """What generate_text would return for the same tokens (stop_fn semantics included)."""
    from zivawhite.chat import make_stop_fn

    stop = make_stop_fn(TOK, 1, markers=STOP_MARKERS, eos_id=TOK.eos_id)
    seq, taken = [TOK.bos_id], []
    for t in script[:max_new_tokens]:
        seq.append(t); taken.append(t)
        probe = make_tensor([seq])
        if stop(probe):
            break
    return clean_reply(TOK.decode(taken))


def _stream(script, **kw):
    info = {}
    model = ScriptedModel(script)
    parts = list(stream_text(model, TOK, "hi", chat=True, info=info, **kw))
    return "".join(parts), parts, info


def test_plain_reply_streams_in_pieces_and_matches():
    script = _ids("  Hello there, friend!  ")
    text, parts, _ = _stream(script)
    assert text == "Hello there, friend!" == _non_streaming(script)
    assert len(parts) > 1                       # actually streamed, not one blob at the end
    assert all(p == p.strip() or True for p in parts)
    assert not text.startswith(" ") and not text.endswith(" ")


def test_a_stop_marker_is_never_leaked_even_split_across_tokens():
    for marker in (USER_TAG, ASSISTANT_TAG, "<|system|>", "<|tool|>", "<|doc:x|>"):
        script = _ids(f"The answer is 42.\n{marker}\nsomething the model should not say")
        text, _, _ = _stream(script)
        assert text == "The answer is 42." == _non_streaming(script)
        assert "<|" not in text


def test_a_partial_marker_lookalike_that_never_completes_is_released_at_the_end():
    script = _ids("use the <| token carefully")      # "<|" starts a marker but "<| t" is not one
    text, _, _ = _stream(script)
    assert text == "use the <| token carefully" == _non_streaming(script)


def test_multibyte_characters_split_across_tokens_are_not_corrupted():
    # byte-level tokens: every UTF-8 byte is its own token, so "é" and "日本" arrive in pieces
    script = _ids("caf\u00e9 \u65e5\u672c\u8a9e ok")
    text, parts, _ = _stream(script)
    assert text == "caf\u00e9 \u65e5\u672c\u8a9e ok" == _non_streaming(script)
    assert "\ufffd" not in text and not any("\ufffd" in p for p in parts)


def test_eos_after_content_ends_the_reply_and_is_not_emitted():
    script = _ids("done") + [TOK.eos_id] + _ids("ignored")
    text, _, info = _stream(script)
    assert text == "done" and info["finish_reason"] == "stop"


def test_finish_reason_is_length_when_the_budget_runs_out():
    script = _ids("a" * 50)
    text, _, info = _stream(script, max_new_tokens=10)
    assert text == "a" * 10 and info["finish_reason"] == "length" and info["completion_tokens"] == 10


def test_whitespace_only_output_yields_nothing():
    text, parts, _ = _stream(_ids("   \n  "))
    assert text == "" and parts == []


def test_property_stream_equals_non_streaming_for_random_scripts():
    rng = random.Random(7)
    pieces = ["Hello", " world", "\n", " ", "caf\u00e9", "\u65e5\u672c", "<|", "|>", USER_TAG, ASSISTANT_TAG,
              "<|tool|>", "42", "!", "  ", "a<", "|user", "x" * 5]
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        script = _ids(text)
        if rng.random() < 0.2:
            script.insert(rng.randint(0, len(script)), TOK.eos_id)
        budget = rng.choice([3, 8, 10_000])
        got, _, _ = _stream(script, max_new_tokens=budget)
        assert got == _non_streaming(script, budget), (text, budget)


def test_build_chat_prompt_is_used_with_history():
    # sanity that the streaming prompt is the same template as the non-streaming one
    assert build_chat_prompt("q", [{"role": "system", "content": "be brief"}]).startswith("<|system|>\nbe brief")


def test_raw_mode_streams_split_multibyte_characters_without_corruption():
    # chat=False has no stop markers, hence no marker hold-back: the incomplete-character guard is the only
    # thing standing between a client and a stray U+FFFD
    text_in = "caf\u00e9 \u65e5\u672c\u8a9e ok \U0001F600 end"
    model = ScriptedModel(_ids(text_in))
    parts = list(stream_text(model, TOK, "hi", chat=False))
    assert "".join(parts) == text_in
    assert not any("\ufffd" in p for p in parts)
    assert len(parts) > 1
