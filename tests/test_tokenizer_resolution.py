"""A model finds the tokenizer it was trained with -- it is never silently paired with the bundled 12k one.

These tests use real vocab.json / merges.txt folders on disk and a tiny stand-in for the `tokenizers` library (so they
run anywhere and say the same thing with or without it installed). What is under test is OUR logic: saving the
tokenizer with a run, finding it again, refusing a mismatch, and keeping a continued run on its own vocabulary."""
import json
import os
import sys
import types

import pytest

from zivawhite import tokenizer as tkz
from zivawhite.tokenizer import (BUNDLED_TOKENIZER_DIR, TokenizerMismatch, TokenizerNotFound, describe_tokenizer,
                                 expectation_from_checkpoint, has_tokenizer, load_tokenizer, resolve_tokenizer,
                                 run_tokenizer_dir, save_run_tokenizer, tokenizer_fingerprint, tokenizer_for_checkpoint)


class _FakeBLBPE:
    """Just enough of tokenizers.ByteLevelBPETokenizer for BPETokenizer.load()."""

    def __init__(self, vocab=None, merges=None):
        with open(vocab, encoding="utf-8") as f:
            self.vocab = json.load(f)

    def add_special_tokens(self, toks):
        for t in toks:
            self.vocab.setdefault(t, len(self.vocab))

    def get_vocab_size(self):
        return len(self.vocab)

    def token_to_id(self, t):
        return self.vocab.get(t)


@pytest.fixture(autouse=True)
def fake_tokenizers_library(monkeypatch):
    mod = types.ModuleType("tokenizers")
    mod.ByteLevelBPETokenizer = _FakeBLBPE
    monkeypatch.setitem(sys.modules, "tokenizers", mod)


def make_tok(folder, n_tokens, tag="a"):
    """A real tokenizer folder with `n_tokens` entries (specials included)."""
    os.makedirs(folder, exist_ok=True)
    vocab = {t: i for i, t in enumerate(tkz.SPECIAL_TOKENS)}
    while len(vocab) < n_tokens:
        vocab[f"{tag}{len(vocab)}"] = len(vocab)
    with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(os.path.join(folder, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
    return str(folder)


def fake_ckpt(vocab, sha=None):
    return {"config": {"vocab_size": vocab}, "extra": {"tokenizer": {"sha256": sha, "vocab_size": vocab}} if sha else {}}


def test_bundled_tokenizer_is_the_12k_one_and_is_found_in_the_package():
    assert has_tokenizer(BUNDLED_TOKENIZER_DIR)
    assert load_tokenizer(None).vocab_size == 12000
    assert load_tokenizer(None).source_dir == BUNDLED_TOKENIZER_DIR


# ------------------------------------------------------------------------------ the bug: silent fallback to the 12k vocab
def test_a_tokenizer_dir_that_holds_no_tokenizer_is_an_error_not_a_silent_switch_to_12k(tmp_path):
    """Before: load_tokenizer("typo/") quietly returned the bundled 12k tokenizer and training went ahead with it."""
    with pytest.raises(TokenizerNotFound, match="no tokenizer in"):
        load_tokenizer(str(tmp_path / "typo"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TokenizerNotFound):
        load_tokenizer(str(empty))


def test_an_explicit_folder_is_used_exactly(tmp_path):
    mine = make_tok(tmp_path / "mine", 500)
    tok = load_tokenizer(mine)
    assert tok.vocab_size == 500 and tok.source_dir == mine


# ------------------------------------------------------------------------------ saved with the run, found again
def test_training_saves_the_tokenizer_next_to_the_checkpoint_and_records_its_fingerprint(tmp_path):
    src = make_tok(tmp_path / "custom", 700)
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(src), str(run))
    assert meta["dir"] == "tokenizer" and meta["vocab_size"] == 700 and meta["kind"] == "bpe" and not meta["bundled"]
    assert meta["sha256"] == tokenizer_fingerprint(src) == tokenizer_fingerprint(str(run / "tokenizer"))
    assert has_tokenizer(str(run / "tokenizer"))
    assert run_tokenizer_dir(str(run / "checkpoint.pt")) == str(run / "tokenizer")


def test_the_bundled_tokenizer_is_saved_into_the_run_too_so_the_run_is_self_contained(tmp_path):
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(None), str(run))
    assert meta["bundled"] is True and meta["vocab_size"] == 12000
    assert load_tokenizer(str(run / "tokenizer")).vocab_size == 12000


def test_serving_finds_the_saved_tokenizer_without_being_told(tmp_path):
    """The reported symptom: train with a custom vocabulary, then ask/serve without repeating --tokenizer_dir."""
    custom = make_tok(tmp_path / "custom", 700)
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(custom), str(run))
    ckpt = fake_ckpt(700, meta["sha256"])
    tok, how = tokenizer_for_checkpoint(str(run / "checkpoint_best.pt"), ckpt)
    assert tok.vocab_size == 700 and tok.source_dir == str(run / "tokenizer") and "saved with the checkpoint" in how


def test_a_checkpoint_trained_with_the_bundled_tokenizer_still_gets_it(tmp_path):
    run = tmp_path / "runs" / "ziva"
    tok, how = tokenizer_for_checkpoint(str(run / "checkpoint.pt"), fake_ckpt(12000))     # an old run: nothing saved
    assert tok.vocab_size == 12000 and "bundled" in how


def test_the_saved_tokenizer_beats_the_configured_and_the_bundled_one(tmp_path):
    saved_src, configured = make_tok(tmp_path / "s", 700, "s"), make_tok(tmp_path / "c", 700, "c")
    run = tmp_path / "runs" / "r"
    save_run_tokenizer(load_tokenizer(saved_src), str(run))
    tok, how = resolve_tokenizer(None, str(run / "checkpoint.pt"), configured, {"vocab_size": 700})
    assert tok.source_dir == str(run / "tokenizer")
    # an old checkpoint with nothing saved falls to the configured one when its size fits
    tok, how = resolve_tokenizer(None, str(tmp_path / "other" / "checkpoint.pt"), configured, {"vocab_size": 700})
    assert tok.source_dir == configured and "autolearn.tokenizer_dir" in how


# ------------------------------------------------------------------------------ never the wrong vocabulary
def test_a_candidate_of_the_wrong_size_is_skipped_never_used(tmp_path):
    wrong = make_tok(tmp_path / "wrong", 900)
    tok, how = resolve_tokenizer(None, str(tmp_path / "x" / "checkpoint.pt"), wrong, {"vocab_size": 12000})
    assert tok.vocab_size == 12000 and "bundled" in how               # the configured 900-token one did not fit


def test_when_nothing_fits_you_get_a_precise_message_not_garbage_output(tmp_path):
    with pytest.raises(TokenizerMismatch) as exc:
        resolve_tokenizer(None, str(tmp_path / "runs" / "r" / "checkpoint.pt"), None, {"vocab_size": 32000})
    msg = str(exc.value)
    assert "32,000 tokens" in msg and "12,000 tokens" in msg and "--tokenizer_dir" in msg and "runs" in msg


def test_same_size_but_different_tokenizer_is_caught_by_the_fingerprint(tmp_path):
    a, b = make_tok(tmp_path / "a", 700, "a"), make_tok(tmp_path / "b", 700, "b")
    assert tokenizer_fingerprint(a) != tokenizer_fingerprint(b)
    run = tmp_path / "runs" / "r"
    meta = save_run_tokenizer(load_tokenizer(a), str(run))
    ckpt = fake_ckpt(700, meta["sha256"])
    with pytest.raises(TokenizerMismatch, match="different tokenizer of the same size"):
        tokenizer_for_checkpoint(str(run / "checkpoint.pt"), ckpt, explicit=b)
    assert tokenizer_for_checkpoint(str(run / "checkpoint.pt"), ckpt, explicit=a)[0].source_dir == a


def test_an_explicit_tokenizer_of_the_wrong_size_is_refused_with_advice(tmp_path):
    wrong = make_tok(tmp_path / "w", 900)
    with pytest.raises(TokenizerMismatch) as exc:
        resolve_tokenizer(wrong, None, None, {"vocab_size": 700})
    assert "900 tokens" in str(exc.value) and "700" in str(exc.value) and "NEW model" in str(exc.value)


def test_a_runs_saved_tokenizer_is_never_swapped_out_from_under_its_checkpoints(tmp_path):
    a, b = make_tok(tmp_path / "a", 700, "a"), make_tok(tmp_path / "b", 700, "b")
    run = tmp_path / "runs" / "r"
    save_run_tokenizer(load_tokenizer(a), str(run))
    with pytest.raises(TokenizerMismatch, match="cannot be replaced"):
        save_run_tokenizer(load_tokenizer(b), str(run))                 # resuming / continuing: refuse
    save_run_tokenizer(load_tokenizer(a), str(run))                     # the same one again is fine (idempotent)
    meta = save_run_tokenizer(load_tokenizer(b), str(run), replace=True)    # a brand-new run owns its folder
    assert meta["sha256"] == tokenizer_fingerprint(b) and tokenizer_fingerprint(str(run / "tokenizer")) == tokenizer_fingerprint(b)


def test_replacing_a_saved_tokenizer_leaves_no_stale_files_behind(tmp_path):
    a = make_tok(tmp_path / "a", 700)
    run = tmp_path / "runs" / "r"
    save_run_tokenizer(load_tokenizer(a), str(run))
    open(str(run / "tokenizer" / "special_tokens.json"), "w").write("[]")
    b = make_tok(tmp_path / "b", 700, "b")
    save_run_tokenizer(load_tokenizer(b), str(run), replace=True)
    assert not os.path.exists(str(run / "tokenizer" / "special_tokens.json"))


# ------------------------------------------------------------------------------ the byte-level fallback is a first-class citizen
def test_the_byte_level_fallback_is_saved_as_a_marker_and_found_again(tmp_path):
    from zivawhite.tokenizer import ByteLevelTokenizer
    run = tmp_path / "runs" / "r"
    meta = save_run_tokenizer(ByteLevelTokenizer(), str(run))
    assert meta["kind"] == "byte" and has_tokenizer(str(run / "tokenizer"))
    tok, how = tokenizer_for_checkpoint(str(run / "checkpoint.pt"), fake_ckpt(ByteLevelTokenizer.vocab_size))
    assert isinstance(tok, ByteLevelTokenizer) and "saved with the checkpoint" in how


# ------------------------------------------------------------------------------ checkpoint metadata
def test_expectation_comes_from_the_checkpoint_config_and_recorded_fingerprint():
    assert expectation_from_checkpoint({"config": {"vocab_size": 8000}, "extra": {"tokenizer": {"sha256": "abc"}}}) == \
        {"vocab_size": 8000, "sha256": "abc"}
    assert expectation_from_checkpoint({"config": {"vocab_size": 8000}}) == {"vocab_size": 8000, "sha256": None}
    assert expectation_from_checkpoint(None) == {"vocab_size": None, "sha256": None}


def test_describe_tokenizer_reports_what_it_is(tmp_path):
    d = describe_tokenizer(load_tokenizer(make_tok(tmp_path / "t", 640)))
    assert d["kind"] == "bpe" and d["vocab_size"] == 640 and d["bundled"] is False and len(d["sha256"]) == 64
    assert describe_tokenizer(load_tokenizer(None))["bundled"] is True


# ------------------------------------------------------------------------------ what a training run picks
from zivawhite.tokenizer import pick_training_tokenizer  # noqa: E402


def test_a_brand_new_model_starts_from_the_bundled_tokenizer_unless_told_otherwise(tmp_path):
    tok, how = pick_training_tokenizer(None, None, None)
    assert tok.vocab_size == 12000 and "bundled" in how and "new model" in how
    mine = make_tok(tmp_path / "mine", 800)
    tok, how = pick_training_tokenizer(mine, None, None)
    assert tok.source_dir == mine and "--tokenizer_dir" in how


def test_resuming_keeps_the_runs_own_vocabulary_even_if_the_flag_is_forgotten(tmp_path):
    custom = make_tok(tmp_path / "custom", 800)
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(custom), str(run))
    tok, how = pick_training_tokenizer(None, str(run / "checkpoint.pt"), fake_ckpt(800, meta["sha256"]))
    assert tok.vocab_size == 800 and tok.source_dir == str(run / "tokenizer")


def test_warm_starting_from_a_run_inherits_its_vocabulary(tmp_path):
    """autolearn's chat run: `train --init_from runs/ziva/checkpoint_best.pt --out_dir runs/ziva_behavior` with no flag."""
    custom = make_tok(tmp_path / "custom", 800)
    base = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(custom), str(base))
    tok, _ = pick_training_tokenizer(None, str(base / "checkpoint_best.pt"), fake_ckpt(800, meta["sha256"]))
    assert tok.source_dir == str(base / "tokenizer")
    chat_meta = save_run_tokenizer(tok, str(tmp_path / "runs" / "ziva_behavior"))      # and the chat run carries it on
    assert chat_meta["sha256"] == meta["sha256"]


def test_continuing_a_run_with_a_different_tokenizer_stops_with_an_explanation(tmp_path):
    a, b = make_tok(tmp_path / "a", 800, "a"), make_tok(tmp_path / "b", 800, "b")
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(a), str(run))
    with pytest.raises(TokenizerMismatch):
        pick_training_tokenizer(b, str(run / "checkpoint.pt"), fake_ckpt(800, meta["sha256"]))
    with pytest.raises(TokenizerNotFound):
        pick_training_tokenizer(str(tmp_path / "nope"), None, None)


# ------------------------------------------------------------------------------ the router (what ask / serve / the web UI use)
def _router(tmp_path, **kw):
    router_mod = pytest.importorskip("zivawhite.router")        # needs torch (or the model stub used by the stub-run harness)
    return router_mod.ZivaWhite(device="cpu", queue_dir=str(tmp_path / "queue"), **kw)


def _fake_load_model(monkeypatch, ckpt_file, vocab, sha=None, tuned=True):
    from types import SimpleNamespace
    model = SimpleNamespace(cfg=SimpleNamespace(vocab_size=vocab, n_layer=2, n_embd=32, block_size=64))
    ckpt = {"config": {"vocab_size": vocab}, "step": 5,
            "extra": {"behavior_tuned": tuned, **({"tokenizer": {"sha256": sha, "vocab_size": vocab}} if sha else {})}}
    monkeypatch.setattr("zivawhite.checkpoint.load_model", lambda path, device=None, prefer_best=True: (model, ckpt, "cpu", ckpt_file))


def test_the_router_pairs_a_loaded_checkpoint_with_the_tokenizer_it_was_trained_with(tmp_path, monkeypatch):
    """THE reported problem: a model trained with its own vocabulary used to be served with the bundled 12k one
    (and refused to load) unless --tokenizer_dir was passed everywhere. Now it just works."""
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(make_tok(tmp_path / "custom", 700)), str(run))
    ckpt_file = str(run / "checkpoint_best.pt")
    _fake_load_model(monkeypatch, ckpt_file, 700, meta["sha256"])
    z = _router(tmp_path)                                      # nothing about a tokenizer is passed
    assert z.tokenizer.vocab_size == 12000                     # only a placeholder until a model is loaded
    z.load_local_model(ckpt_file)
    assert z.tokenizer.vocab_size == 700 and z.tokenizer.source_dir == str(run / "tokenizer")
    assert z.model_info["vocab_size"] == 700 and "saved with the checkpoint" in z.model_info["tokenizer"]


def test_a_model_trained_with_the_bundled_tokenizer_loads_with_it(tmp_path, monkeypatch):
    ckpt_file = str(tmp_path / "runs" / "old" / "checkpoint.pt")          # an old run: no tokenizer folder, no fingerprint
    _fake_load_model(monkeypatch, ckpt_file, 12000)
    z = _router(tmp_path)
    z.load_local_model(ckpt_file)
    assert z.tokenizer.vocab_size == 12000 and "bundled" in z.model_info["tokenizer"]


def test_a_model_whose_tokenizer_cannot_be_found_fails_loudly_instead_of_writing_garbage(tmp_path, monkeypatch):
    ckpt_file = str(tmp_path / "runs" / "lost" / "checkpoint.pt")         # 32k vocabulary, and its tokenizer is nowhere
    _fake_load_model(monkeypatch, ckpt_file, 32000)
    z = _router(tmp_path)
    with pytest.raises(RuntimeError, match="32,000 tokens"):
        z.load_local_model(ckpt_file)
    assert not z.has_local_model()


def test_the_configured_tokenizer_is_the_fallback_for_a_run_that_predates_saving(tmp_path, monkeypatch):
    configured = make_tok(tmp_path / "configured", 700)
    ckpt_file = str(tmp_path / "runs" / "old" / "checkpoint.pt")
    _fake_load_model(monkeypatch, ckpt_file, 700)
    z = _router(tmp_path, tokenizer_dir=configured)
    z.load_local_model(ckpt_file)
    assert z.tokenizer.source_dir == configured and "autolearn.tokenizer_dir" in z.model_info["tokenizer"]


def test_a_bad_configured_tokenizer_does_not_stop_a_model_that_carries_its_own(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "ziva"
    meta = save_run_tokenizer(load_tokenizer(make_tok(tmp_path / "custom", 700)), str(run))
    ckpt_file = str(run / "checkpoint.pt")
    _fake_load_model(monkeypatch, ckpt_file, 700, meta["sha256"])
    z = _router(tmp_path, tokenizer_dir=str(tmp_path / "typo"))           # a typo in the config
    z.load_local_model(ckpt_file)
    assert z.tokenizer.vocab_size == 700


def test_a_tokenizer_the_caller_passes_in_is_used_as_given(tmp_path, monkeypatch):
    mine = load_tokenizer(make_tok(tmp_path / "mine", 700))
    ckpt_file = str(tmp_path / "runs" / "r" / "checkpoint.pt")
    _fake_load_model(monkeypatch, ckpt_file, 700)
    z = _router(tmp_path, tokenizer=mine)
    z.load_local_model(ckpt_file)
    assert z.tokenizer is mine
