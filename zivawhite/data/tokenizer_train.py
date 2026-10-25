"""
Train a tokenizer on YOUR datasets folder, instead of using the bundled general-English one.

    python -m zivawhite.app tokenizer --vocab_size 8000

When is that worth it?  The bundled tokenizer (12,000 tokens, trained on English books) is a fine default for
English prose. A vocabulary learned from your own text is better when the language is not English, or the text is
very different (code, medicine, legal, a game's lore), because common words of YOUR data become single tokens and
the model's context window then holds more of it. It needs enough text to learn from -- see MIN_CHARS_PER_TOKEN.

A tokenizer is part of the model: it decides what every number the model sees means. So this must happen BEFORE
the first training run. Once a model has trained with a vocabulary, that vocabulary stays with it
(runs/<run>/tokenizer/), and changing it means training a new model from scratch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Optional

from .folder import DatasetFolder, UnsupportedFile, iter_file

# A vocabulary entry needs to be seen many times to be worth having. About 150 characters of text per token of
# vocabulary is a sensible floor (8,000 tokens -> 1.2 MB); below it the merges are mostly noise.
MIN_CHARS_PER_TOKEN = 150


class NotEnoughText(RuntimeError):
    """The datasets folder holds too little text to learn a vocabulary of the requested size."""


def iter_dataset_texts(folder: DatasetFolder) -> Iterator[str]:
    """Every document of text (and every chat turn) in the datasets folder, in the same cleaned form training sees.
    Unreadable files are skipped; `datasets` reports them."""
    scan = folder.scan()
    for df in scan.files:
        try:
            for kind, payload in iter_file(df, folder.max_file_bytes, folder.max_doc_chars):
                if kind == "text" and isinstance(payload, str) and payload.strip():
                    yield payload
                elif kind == "chat" and isinstance(payload, list):
                    for message in payload:
                        content = message.get("content") if isinstance(message, dict) else None
                        if isinstance(content, str) and content.strip():
                            yield content
        except UnsupportedFile:
            continue


def count_chars(folder: DatasetFolder, limit: Optional[int] = None) -> int:
    """Characters of text in the folder (stops counting at `limit`, if given)."""
    total = 0
    for text in iter_dataset_texts(folder):
        total += len(text)
        if limit is not None and total >= limit:
            break
    return total


def train_from_datasets(folder: DatasetFolder, vocab_size: int, out_dir: str, *, force: bool = False,
                        trainer: Optional[Callable] = None) -> dict:
    """Train a byte-level BPE of `vocab_size` tokens on the folder's text and save it to `out_dir`.

    Raises NotEnoughText if there is too little to learn from (override with force=True, at your own risk).
    `trainer(texts, vocab_size, out_dir)` is BPETokenizer.train_from_iterator unless a test supplies another.
    """
    needed = vocab_size * MIN_CHARS_PER_TOKEN
    have = count_chars(folder, limit=needed)
    if have < needed and not force:
        raise NotEnoughText(
            f"the datasets folder holds {have:,}+ characters of text; a {vocab_size:,}-token vocabulary needs about "
            f"{needed:,} to be worth learning. Add more text, ask for a smaller --vocab_size, or use the bundled "
            f"tokenizer (just leave this out). --force trains anyway.")
    if trainer is None:
        from ..tokenizer import BPETokenizer

        trainer = BPETokenizer.train_from_iterator
    tok = trainer(iter_dataset_texts(folder), vocab_size, out_dir)
    return {"out_dir": os.fspath(out_dir), "vocab_size": int(tok.vocab_size), "chars_seen_at_least": have}


def default_out_dir(cfg) -> Path:
    """Where your own tokenizer goes by default: <data>/tokenizer."""
    return cfg.data_dir / "tokenizer"
