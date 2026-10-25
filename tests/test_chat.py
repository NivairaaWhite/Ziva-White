"""chat.py: the one definition of the chat format, shared by training data and inference.
Needs no trained model -- these are the pieces whose bugs made "hello" return book text."""
import torch

from zivawhite.chat import (ASSISTANT_TAG, USER_TAG, build_chat_prompt, clean_reply, fit_prompt_ids,
                            make_stop_fn, render_conversation)
from zivawhite.tokenizer import ByteLevelTokenizer


def test_prompt_is_the_training_text_up_to_the_open_assistant_turn():
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi there!"}]
    trained = render_conversation(messages)
    # what the model saw in training for a first turn ...
    first_turn = render_conversation(messages[:1]) + "\n" + ASSISTANT_TAG + "\n"
    # ... is exactly the prompt inference now builds
    assert build_chat_prompt("hello") == first_turn
    assert trained.startswith(first_turn.rstrip("\n"))


def test_datasets_folder_chat_corpus_uses_exactly_the_inference_chat_format(tmp_path):
    """Training data written by the datasets folder and the prompt built at inference must agree
    byte for byte -- that mismatch is what once made "hello" return book text."""
    import json

    from zivawhite.data.folder import DatasetFolder

    folder = DatasetFolder(tmp_path / "d", tmp_path / "w", settle_s=0)
    folder.ensure_layout()
    (tmp_path / "d" / "c.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": " hi "},
                                                                  {"role": "assistant", "content": "yo"}]}))
    corpus = open(folder.rebuild().chat_corpus, encoding="utf-8").read()
    assert f"{USER_TAG}\nhi\n{ASSISTANT_TAG}\nyo" in corpus
    assert render_conversation([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]) in corpus


def test_history_is_rendered_before_the_new_turn():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    prompt = build_chat_prompt("c", history)
    assert prompt.index("a") < prompt.index("b") < prompt.index("c")
    assert prompt.endswith(f"{ASSISTANT_TAG}\n")


def test_clean_reply_cuts_at_the_first_stop_marker():
    assert clean_reply("Hello!\n<|user|>\nnext question") == "Hello!"
    assert clean_reply("Hi\n<|doc:book|> stuff") == "Hi"
    assert clean_reply("  just text  ") == "just text"
    assert clean_reply(f"A {ASSISTANT_TAG} B {USER_TAG} C") == "A"


def test_long_text_is_shortened_but_the_chat_template_survives():
    tok = ByteLevelTokenizer()
    limit = 120
    render = lambda t: build_chat_prompt(t)
    ids = fit_prompt_ids(tok, "x" * 5000, render, limit)
    assert len(ids) <= limit
    text = tok.decode(ids)
    assert text.startswith(USER_TAG) and text.endswith(f"{ASSISTANT_TAG}\n")  # not cut off


def test_short_text_is_left_alone():
    tok = ByteLevelTokenizer()
    ids = fit_prompt_ids(tok, "hello", build_chat_prompt, 500)
    assert tok.decode(ids) == build_chat_prompt("hello")
    assert ids[0] == tok.bos_id


def _stop_after(tok, prompt_ids, new_ids, **kw):
    """Feed new tokens one at a time; return how many were needed before stop_fn fired (or None)."""
    stop = make_stop_fn(tok, len(prompt_ids), **kw)
    seq = list(prompt_ids)
    for n, t in enumerate(new_ids, start=1):
        seq.append(t)
        if stop(torch.tensor([seq])):
            return n
    return None


def test_stop_fn_ends_generation_when_a_new_turn_starts():
    tok = ByteLevelTokenizer()
    prompt = tok.encode("whatever", add_bos=True)
    reply = tok.encode("Hi!\n" + USER_TAG + "\nmore")
    n = _stop_after(tok, prompt, reply, markers=(USER_TAG,))
    assert n is not None and n < len(reply)  # stopped at the marker, not at the end


def test_stop_fn_does_not_see_markers_in_the_prompt():
    tok = ByteLevelTokenizer()
    prompt = tok.encode(build_chat_prompt("hi"), add_bos=True)  # already contains <|user|>
    n = _stop_after(tok, prompt, tok.encode("Hello there, friend."), markers=(USER_TAG,))
    assert n is None


def test_eos_only_counts_after_real_content():
    tok = ByteLevelTokenizer()
    prompt = tok.encode("q", add_bos=True)
    # older checkpoints emit <eos> right away; stopping there would always return an empty reply
    assert _stop_after(tok, prompt, [tok.eos_id, *tok.encode("answer"), tok.eos_id], eos_id=tok.eos_id) == 8
