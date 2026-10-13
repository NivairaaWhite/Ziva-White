"""build_shards: per-document <eos>, whole-document validation, streaming output."""
import re

import numpy as np
import pytest

from zivawhite.data.ingest import _iter_documents, build_shards
from zivawhite.tokenizer import load_tokenizer

TOK = load_tokenizer(None)  # the same default build_shards(..., None) uses (BPE if installed, else bytes)


def _tokens(path):
    return np.fromfile(path, dtype=np.uint16)


def _write(tmp_path, text, name="corpus.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_marker_mode_puts_one_eos_per_document_not_per_line(tmp_path):
    # a "book" hard-wrapped into many lines: per-line EOS used to put <eos> after every one
    book = "".join(f"line {i} of the first wrapped book\n" for i in range(200))
    other = "".join(f"another book, line {i}\n" for i in range(100))
    corpus = _write(tmp_path, f"<|doc:a|>\n{book}<|doc:b|>\n{other}")
    out = str(tmp_path / "shards")
    n_train, n_val = build_shards(corpus, None, out, doc_mode="marker", min_val_tokens=50, max_block_size=40)
    all_ids = np.concatenate([_tokens(out + "/train.bin"), _tokens(out + "/val.bin")])
    assert n_train + n_val == len(all_ids)
    assert (all_ids == TOK.eos_id).sum() == 2           # exactly one per document
    assert "<|doc" not in TOK.decode(all_ids.tolist())  # marker lines are boundaries, not training text


def test_line_mode_is_one_document_per_line(tmp_path):
    corpus = _write(tmp_path, "".join(f"doc number {i} with some words in it\n" for i in range(60)))
    out = str(tmp_path / "shards")
    build_shards(corpus, None, out, doc_mode="line", min_val_tokens=20, max_block_size=8)
    all_ids = np.concatenate([_tokens(out + "/train.bin"), _tokens(out + "/val.bin")])
    assert (all_ids == TOK.eos_id).sum() == 60


def test_validation_is_whole_documents_not_a_file_tail(tmp_path):
    docs = [f"<|doc|>\nconversation {i:03d} " + "word " * 30 + "\n" for i in range(40)]
    corpus = _write(tmp_path, "".join(docs))
    out = str(tmp_path / "shards")
    build_shards(corpus, None, out, doc_mode="marker", min_val_tokens=300, max_block_size=100)
    val = TOK.decode(_tokens(out + "/val.bin").tolist())
    train = TOK.decode(_tokens(out + "/train.bin").tolist())
    val_ids = {int(x) for x in re.findall(r"conversation (\d{3})", val)}
    train_ids = {int(x) for x in re.findall(r"conversation (\d{3})", train)}
    assert val_ids and train_ids and not (val_ids & train_ids)      # whole documents, no leakage
    assert val_ids != set(range(40 - len(val_ids), 40))             # not simply the last k documents


def test_split_is_deterministic(tmp_path):
    corpus = _write(tmp_path, "".join(f"<|doc|>\ndoc {i} " + "tok " * 40 + "\n" for i in range(30)))
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for out in (a, b):
        build_shards(corpus, None, out, doc_mode="marker", min_val_tokens=200, max_block_size=50)
    assert _tokens(a + "/val.bin").tolist() == _tokens(b + "/val.bin").tolist()


def test_single_giant_document_falls_back_to_a_tail_split(tmp_path):
    corpus = _write(tmp_path, "one single document " * 400 + "\n")
    out = str(tmp_path / "shards")
    n_train, n_val = build_shards(corpus, None, out, doc_mode="line", min_val_tokens=300, max_block_size=100)
    assert n_val >= 300 and n_train > 0
    assert not (tmp_path / "shards" / "_all_tokens.tmp").exists()   # temp file cleaned up


def test_empty_corpus_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        build_shards(_write(tmp_path, "\n\n  \n"), None, str(tmp_path / "s"))


def test_bad_doc_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        list(_iter_documents(_write(tmp_path, "x\n"), "paragraph"))


def test_chat_corpus_from_the_datasets_folder_has_one_eos_per_conversation(tmp_path):
    """The datasets folder's chat corpus goes through the same shard builder training uses, and every
    conversation (the user's own plus the creator-identity examples) ends with exactly one <eos>."""
    import json

    from zivawhite.data.folder import DatasetFolder
    from zivawhite.identity import Identity
    from helpers import make_cfg

    cfg = make_cfg(tmp_path)
    folder = DatasetFolder(tmp_path / "datasets", tmp_path / "work", settle_s=0)
    folder.ensure_layout()
    rows = [{"prompt": f"question {i} about the harbour", "response": f"answer {i}, given plainly"} for i in range(30)]
    (tmp_path / "datasets" / "chat" / "mine.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    identity = Identity.from_config(cfg).training_records()
    report = folder.rebuild(extra_chat=identity)
    n_examples = report.chat_examples + report.identity_examples
    assert report.chat_examples == 30 and report.identity_examples == len(identity)

    out = str(tmp_path / "shards")
    build_shards(report.chat_corpus, None, out, doc_mode="marker", min_val_tokens=100, max_block_size=64)
    ids = np.concatenate([_tokens(out + "/train.bin"), _tokens(out + "/val.bin")])
    assert (ids == TOK.eos_id).sum() == n_examples
