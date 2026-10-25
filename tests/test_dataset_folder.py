"""The datasets folder: what gets read, how it is classified, and when something needs training.
Pure Python -- no torch, no network."""
import json
import os

import pytest

from zivawhite.data.folder import (DatasetFolder, classify_record, clean_text, normalize_messages, parse_transcript,
                                   strip_gutenberg)


def _folder(tmp_path, settle_s=0.0, **kw):
    root = tmp_path / "datasets"
    root.mkdir(exist_ok=True)
    return DatasetFolder(root, tmp_path / "work", settle_s=settle_s, **kw), root


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


# ---------------------------------------------------------------------------- record shapes
def test_messages_shape_is_chat():
    kind, msgs = classify_record({"messages": [{"role": "user", "content": "hi"},
                                               {"role": "assistant", "content": "hello"}]})
    assert kind == "chat" and [m["role"] for m in msgs] == ["user", "assistant"]


def test_sharegpt_and_alpaca_and_pairs_are_chat():
    kind, msgs = classify_record({"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]})
    assert kind == "chat" and msgs[1]["content"] == "a"
    kind, msgs = classify_record({"instruction": "Translate", "input": "chat", "output": "cat"})
    assert kind == "chat" and msgs[0]["content"] == "Translate\n\nchat"
    for rec in ({"prompt": "p", "response": "r"}, {"question": "p", "answer": "r"}, {"user": "p", "assistant": "r"}):
        assert classify_record(rec)[0] == "chat"


def test_text_field_is_text_and_junk_is_rejected():
    assert classify_record({"text": "some words"}) == ("text", "some words")
    assert classify_record({"nothing": "useful"}) is None
    assert classify_record({"text": "   "}) is None
    assert classify_record(42) is None


def test_system_prompt_is_folded_into_first_user_turn_and_edges_are_trimmed():
    msgs = normalize_messages([{"role": "system", "content": "Be brief."}, {"role": "assistant", "content": "orphan"},
                               {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                               {"role": "user", "content": "unanswered"}])
    assert msgs[0] == {"role": "user", "content": "Be brief.\n\nhi"}
    assert msgs[-1]["role"] == "assistant" and len(msgs) == 2
    assert normalize_messages([{"role": "user", "content": "only a question"}]) is None


def test_transcript_blocks():
    text = "User: hi\nAssistant: hello\nthere\n---\nQ: 2+2?\nA: 4\nAnswer: four\n"
    convos = list(parse_transcript(text))
    assert len(convos) == 2
    assert convos[0][1]["content"] == "hello\nthere"
    assert convos[1][0]["content"] == "2+2?"


# ------------------------------------------------------------------------------- cleaning
def test_clean_text_neutralises_markers_and_tags():
    out = clean_text("a<|user|>b\n<|doc:evil|>\nline\r\n\n\n\n\nend")
    assert "<|user|>" not in out
    assert not any(line.startswith("<|doc") for line in out.splitlines())
    assert "\n\n\n" not in out


def test_strip_gutenberg_only_when_markers_present():
    book = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nreal text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
    assert strip_gutenberg(book).strip() == "real text"
    assert strip_gutenberg("plain text") == "plain text"


# --------------------------------------------------------------------------------- scanning
def test_scan_ignores_readme_hidden_underscore_and_reports_unsupported(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "README.md", "instructions, not data")
    _write(root / ".hidden.txt", "x")
    _write(root / "_sources" / "urls.txt", "https://example.org")
    _write(root / "_eval" / "bench.jsonl", "{}")
    _write(root / "text" / "a.txt", "hello world")
    _write(root / "chat" / "c.jsonl", "{}")
    _write(root / "deep" / "er" / "b.md", "more text")
    _write(root / "notes.docx", "binary-ish")
    scan = folder.scan()
    assert sorted(f.rel for f in scan.files) == ["chat/c.jsonl", "deep/er/b.md", "text/a.txt"]
    assert scan.ignored == ["notes.docx"]
    assert {f.rel: f.hint for f in scan.files}["text/a.txt"] == "text"


def test_missing_folder_is_not_an_error(tmp_path):
    folder = DatasetFolder(tmp_path / "nope", tmp_path / "work")
    assert folder.scan().files == []
    assert folder.check().changed is False


def test_recently_modified_files_are_reported_pending(tmp_path):
    folder, root = _folder(tmp_path, settle_s=60.0)
    _write(root / "a.txt", "just copied")
    assert folder.scan().pending == ["a.txt"]
    st = os.stat(root / "a.txt")
    os.utime(root / "a.txt", ns=(st.st_atime_ns, st.st_mtime_ns - 120 * 10 ** 9))
    assert folder.scan().pending == []


# ---------------------------------------------------------------------------------- building
def test_rebuild_builds_text_and_chat_corpora_from_mixed_files(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "book.txt", "Chapter one. It was a dark night.")
    _write(root / "page.html", "<html><script>bad()</script><body><p>Visible words</p></body></html>")
    _write(root / "rows.jsonl", "\n".join([
        json.dumps({"text": "a json text row"}),
        json.dumps({"prompt": "hi", "response": "hello"}),
        json.dumps({"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}),
        "this is not json",
        json.dumps({"mystery": 1}),
    ]))
    _write(root / "table.csv", "question,answer\nWhat is 2+2?,4\n")
    _write(root / "chat" / "talk.txt", "User: ping\nAssistant: pong\n")
    rep = folder.rebuild()
    assert rep.text_docs == 3 and rep.chat_examples == 4 and rep.bad_records == 2
    text = open(rep.text_corpus, encoding="utf-8").read()
    assert "dark night" in text and "Visible words" in text and "bad()" not in text
    chat = open(rep.chat_corpus, encoding="utf-8").read()
    assert chat.count("<|doc|>") == 4 and "<|user|>\nping\n<|assistant|>\npong" in chat
    assert "rows.jsonl" in rep.bad_examples


def test_duplicates_are_dropped(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "a.txt", "same content here")
    _write(root / "b.txt", "same content here")
    rep = folder.rebuild()
    assert rep.text_docs == 1 and rep.duplicates == 1


def test_extra_chat_is_only_added_when_there_is_real_chat_data(tmp_path):
    extra = [{"messages": [{"role": "user", "content": "Who created you?"}, {"role": "assistant", "content": "Mahasvin S S."}]}]
    folder, root = _folder(tmp_path)
    _write(root / "book.txt", "only prose lives here")
    rep = folder.rebuild(extra_chat=extra)
    assert rep.chat_corpus is None and rep.identity_examples == 0          # text-only stays text-only

    _write(root / "chat" / "c.jsonl", json.dumps({"prompt": "hi", "response": "hello"}))
    rep = folder.rebuild(extra_chat=extra)
    assert rep.chat_examples == 1 and rep.identity_examples == 1
    assert "Mahasvin S S." in open(rep.chat_corpus, encoding="utf-8").read()


def test_unreadable_files_are_reported_not_fatal(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "good.txt", "fine")
    _write(root / "broken.json", "{not json")
    _write(root / "scan.pdf", "%PDF-fake")        # pypdf missing or unreadable either way
    rep = folder.rebuild()
    assert rep.text_docs == 1
    assert "broken.json" in rep.errors and "scan.pdf" in rep.errors


def test_oversized_files_are_refused_with_a_reason(tmp_path):
    folder, root = _folder(tmp_path, max_file_mb=0.0001)   # ~100 bytes
    _write(root / "huge.txt", "x" * 5000)
    rep = folder.rebuild()
    assert rep.text_docs == 0 and "limit" in rep.errors["huge.txt"]


def test_long_documents_are_split_at_paragraphs(tmp_path):
    folder, root = _folder(tmp_path, max_doc_chars=200)
    _write(root / "long.txt", "\n\n".join(f"paragraph number {i} " + "word " * 15 for i in range(12)))
    rep = folder.rebuild()
    assert rep.text_docs > 1
    names = [line for line in open(rep.text_corpus, encoding="utf-8") if line.startswith("<|doc:")]
    assert names[0].startswith("<|doc:long.txt|>") and names[1].startswith("<|doc:long.txt#1|>")


# --------------------------------------------------------------------- change detection / training
def test_check_detects_added_modified_removed(tmp_path):
    folder, root = _folder(tmp_path)
    a = _write(root / "a.txt", "one")
    _write(root / "b.txt", "two")
    assert folder.check().added == ["a.txt", "b.txt"]
    folder.rebuild()
    assert folder.check().changed is False

    _write(root / "a.txt", "one, edited and longer")
    _write(root / "c.txt", "three")
    (root / "b.txt").unlink()
    ch = folder.check()
    assert (ch.added, ch.modified, ch.removed) == (["c.txt"], ["a.txt"], ["b.txt"])


def test_only_the_kind_that_changed_needs_training(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "book.txt", "a story to learn from")
    _write(root / "chat" / "c.jsonl", json.dumps({"prompt": "hi", "response": "hello"}))
    folder.rebuild()
    assert folder.needs_training("text") and folder.needs_training("chat")
    folder.mark_trained("text")
    folder.mark_trained("chat")
    assert not folder.needs_training("text") and not folder.needs_training("chat")

    _write(root / "chat" / "c2.jsonl", json.dumps({"prompt": "bye", "response": "see you"}))
    folder.rebuild()
    assert folder.needs_training("chat") and not folder.needs_training("text")


def test_touching_a_file_without_changing_it_does_not_retrain(tmp_path):
    folder, root = _folder(tmp_path)
    p = _write(root / "book.txt", "stable content")
    folder.rebuild()
    folder.mark_trained("text")
    _bump_mtime(p)
    assert folder.check().changed                         # the cheap check sees a new mtime...
    folder.rebuild()
    assert not folder.needs_training("text")               # ...but the corpus hash is identical


def test_a_failed_training_run_is_retried(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "book.txt", "needs learning")
    folder.rebuild()
    # no mark_trained(): the trainer crashed. The next round must still want to train.
    assert folder.check().changed is False
    assert folder.needs_training("text")


def test_removing_every_file_clears_the_corpus_but_keeps_trained_state(tmp_path):
    folder, root = _folder(tmp_path)
    p = _write(root / "book.txt", "temporary")
    folder.rebuild()
    folder.mark_trained("text")
    p.unlink()
    rep = folder.rebuild()
    assert rep.text_corpus is None and not os.path.exists(folder.corpus_path("text"))
    assert not folder.needs_training("text")


def test_state_survives_a_new_folder_object(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "book.txt", "persisted")
    folder.rebuild()
    folder.mark_trained("text")
    again = DatasetFolder(root, tmp_path / "work", settle_s=0.0)
    assert again.check().changed is False and not again.needs_training("text")


def test_status_summary(tmp_path):
    folder, root = _folder(tmp_path)
    _write(root / "text" / "a.txt", "alpha")
    _write(root / "chat" / "c.jsonl", json.dumps({"prompt": "p", "response": "r"}))
    _write(root / "loose.md", "beta")
    _write(root / "x.xyz", "?")
    st = folder.status()
    assert st["files"] == 3 and st["files_by_folder"] == {"text": 1, "chat": 1, "unsorted": 1}
    assert st["ignored"] == ["x.xyz"] and st["changed_since_last_build"] is True
