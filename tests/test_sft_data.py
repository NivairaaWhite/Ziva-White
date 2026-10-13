"""sft.py's data handling: what the model is (and is not) trained to say. Pure Python, no torch needed."""
import json

import pytest

from zivawhite.chat import ASSISTANT_TAG, build_chat_prompt
from zivawhite.sft import (IGNORE, build_examples, encode_example, load_conversations, lr_at,
                           normalize_conversation, pad_batch, prompt_text, split_conversations, supervised_tokens)
from zivawhite.tokenizer import ByteLevelTokenizer

TOK = ByteLevelTokenizer()


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


# ---------------------------------------------------------------------------- normalisation
def test_roles_from_other_formats_are_mapped_and_system_and_tool_are_kept():
    raw = [{"from": "system", "value": " be brief "}, {"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"},
           {"role": "function", "content": "42"}, {"role": "bot", "content": "done"}]
    assert normalize_conversation(raw) == msgs(("system", "be brief"), ("user", "hi"), ("assistant", "yo"),
                                               ("tool", "42"), ("assistant", "done"))


def test_consecutive_user_or_assistant_turns_merge_but_tool_results_do_not():
    raw = msgs(("user", "a"), ("user", "b"), ("assistant", "c"), ("tool", "r1"), ("tool", "r2"), ("assistant", "d"))
    out = normalize_conversation(raw)
    assert out[0] == {"role": "user", "content": "a\nb"} and [m["content"] for m in out if m["role"] == "tool"] == ["r1", "r2"]


def test_unusable_conversations_are_dropped():
    assert normalize_conversation("nope") is None
    assert normalize_conversation(msgs(("user", "only a question"))) is None
    assert normalize_conversation(msgs(("assistant", "no user"))) is None
    assert normalize_conversation([{"role": "user", "content": "  "}, {"role": "assistant", "content": ""}]) is None
    assert normalize_conversation(msgs(("assistant", "stray"), ("user", "q"), ("assistant", "a"))) == msgs(("user", "q"), ("assistant", "a"))


# ------------------------------------------------------------------------------- masking
def test_only_the_reply_and_end_of_turn_are_supervised():
    ids, labels = encode_example(TOK, msgs(("user", "What is 2+2?")), "  It is 4.  ", 256)
    assert len(ids) == len(labels)
    target = [t for t in labels if t != IGNORE]
    assert TOK.decode(target) == "It is 4." and target[-1] == TOK.eos_id          # reply (stripped) + <eos>
    first_supervised = next(i for i, t in enumerate(labels) if t != IGNORE)
    assert all(t == IGNORE for t in labels[:first_supervised]) and ids[first_supervised:] == target
    assert ids[0] == TOK.bos_id and labels[0] == IGNORE


def test_the_prompt_tokens_are_exactly_what_inference_builds():
    history = msgs(("system", "Be brief."), ("user", "a"), ("assistant", "b"), ("user", "c"))
    ids, labels = encode_example(TOK, history, "answer", 512)
    n_prompt = sum(1 for t in labels if t == IGNORE)
    expected = TOK.encode(build_chat_prompt("c", history[:-1]), add_bos=True)
    assert ids[:n_prompt] == expected
    assert prompt_text(history).endswith(f"{ASSISTANT_TAG}\n") and "<|system|>\nBe brief." in prompt_text(history)


def test_system_prompt_stays_a_system_turn_not_folded_into_the_user_message():
    text = TOK.decode(encode_example(TOK, msgs(("system", "RULES"), ("user", "hi")), "ok", 256)[0])
    assert text.index("<|system|>\nRULES") < text.index("<|user|>\nhi")


def test_every_assistant_turn_becomes_its_own_example_with_earlier_turns_as_context():
    conv = msgs(("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"), ("user", "q3"), ("assistant", "a3"))
    ex = build_examples([conv], TOK, 512)
    assert len(ex) == 3
    second_ids, second_labels = ex[1]
    assert TOK.decode(second_ids).count("a1") == 1                                 # a1 is context ...
    assert TOK.decode([t for t in second_labels if t != IGNORE]) == "a2"           # ... and only a2 is learned
    assert supervised_tokens(ex) == sum(len(r) + 1 for r in ("a1", "a2", "a3"))     # +1 for each <eos>


def test_tool_call_turns_train_like_any_other_assistant_reply():
    call = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'
    conv = msgs(("user", "6*7?"), ("assistant", call), ("tool", "42"), ("assistant", "It is 42."))
    ex = build_examples([conv], TOK, 512)
    assert len(ex) == 2
    assert TOK.decode([t for t in ex[0][1] if t != IGNORE]) == call
    assert "<|tool|>\n42" in TOK.decode(ex[1][0])


# ----------------------------------------------------------------------------- truncation
def test_a_too_long_prompt_loses_its_oldest_context_but_keeps_bos_and_the_open_assistant_turn():
    history = msgs(("user", "x" * 2000))
    ids, labels = encode_example(TOK, history, "short answer", 128)
    assert len(ids) <= 128 and ids[0] == TOK.bos_id
    text = TOK.decode(ids)
    assert f"{ASSISTANT_TAG}\nshort answer" in text                                # the template tail survived
    assert TOK.decode([t for t in labels if t != IGNORE]) == "short answer"


def test_a_too_long_reply_is_cut_and_NOT_given_an_end_of_turn():
    ids, labels = encode_example(TOK, msgs(("user", "q")), "y" * 2000, 128)
    target = [t for t in labels if t != IGNORE]
    assert len(ids) <= 128 and TOK.eos_id not in target and len(target) > 0


def test_every_example_fits_the_window_and_has_something_to_learn():
    conv = msgs(("system", "s" * 300), ("user", "u" * 300), ("assistant", "a" * 300), ("user", "v" * 300), ("assistant", "b" * 300))
    for block in (64, 128, 400):
        for ids, labels in build_examples([conv], TOK, block):
            assert len(ids) == len(labels) <= block and ids[0] == TOK.bos_id
            assert any(t != IGNORE for t in labels[1:])


# ------------------------------------------------------------------------------- batching
def test_padding_never_contributes_to_the_loss_and_targets_are_shifted_by_one():
    a = encode_example(TOK, msgs(("user", "hi")), "ok", 256)
    b = encode_example(TOK, msgs(("user", "a much longer question than the other one")), "a longer answer too", 256)
    xs, ys = pad_batch([a, b], TOK.pad_id)
    assert len(xs[0]) == len(xs[1]) == len(ys[0]) == len(ys[1]) == max(len(a[0]), len(b[0])) - 1
    short_len = len(a[0]) - 1
    assert xs[0][short_len:] == [TOK.pad_id] * (len(xs[0]) - short_len)
    assert ys[0][short_len:] == [IGNORE] * (len(ys[0]) - short_len)                 # padding is ignored
    for x, y, (ids, labels) in zip(xs, ys, (a, b)):                                 # y[t] is the label of token t+1
        for t in range(len(ids) - 1):
            assert x[t] == ids[t] and y[t] == labels[t + 1]


# ---------------------------------------------------------------------- splits and loading
def test_validation_split_is_by_conversation_deterministic_and_skipped_for_tiny_sets():
    convs = [msgs(("user", f"q{i}"), ("assistant", f"a{i}")) for i in range(100)]
    tr, va = split_conversations(convs, 0.1, seed=1)
    assert len(va) == 10 and len(tr) == 90 and not any(c in tr for c in va)
    assert split_conversations(convs, 0.1, seed=1) == (tr, va)
    tr2, va2 = split_conversations(convs[:10], 0.1)
    assert va2 == [] and len(tr2) == 10                                             # too small to hold anything out


def test_load_conversations_reads_jsonl_json_and_folders_and_skips_junk(tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join([
        json.dumps({"messages": msgs(("user", "q"), ("assistant", "a"))}),
        "{not json",
        json.dumps({"messages": msgs(("user", "only user"))}),
        json.dumps({"conversations": [{"from": "human", "value": "h"}, {"from": "gpt", "value": "g"}]}),
    ]), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps([{"messages": msgs(("system", "s"), ("user", "q2"), ("assistant", "a2"))}]), encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("hello")
    assert len(load_conversations([str(tmp_path)])) == 3
    assert len(load_conversations([str(tmp_path / "a.jsonl")])) == 2
    assert len(load_conversations([str(tmp_path / "*.json*")])) == 3


def test_learning_rate_warms_up_then_decays_to_a_floor():
    lr = [lr_at(s, 100, 1.0, 10) for s in range(100)]
    assert lr[0] < lr[9] <= 1.0 and max(lr) == pytest.approx(1.0, abs=0.05)
    assert lr[-1] == pytest.approx(0.1, abs=0.02) and all(a >= b - 1e-9 for a, b in zip(lr[10:], lr[11:]))
