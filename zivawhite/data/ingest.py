"""
Turns raw text (bulk downloads, or the router's opportunistic
self-training queue) into tokenized, memory-mapped training shards --
the same format data/dataset.py reads for training, laptop or GPU.

Every shard has a companion manifest.jsonl recording exactly which
URL/source/license each chunk of tokens came from, so the training
data is auditable after the fact, not just "some internet text."
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Iterable, Iterator, List, Tuple

import numpy as np

from .sources import is_allow_listed, source_for
from ..chat import DOC_MARKER_PREFIX
from ..connectors.web_fetch_connector import WebFetchConnector
from ..tokenizer import load_tokenizer


def fold_queue(queue_dir: str, corpus_out: str, manifest_out: str) -> int:
    """Move everything sitting in the router's *.jsonl ingest queue into
    the durable corpus + manifest, then clear the queue files. Returns
    the number of documents folded in."""
    os.makedirs(os.path.dirname(corpus_out) or ".", exist_ok=True)
    n = 0
    with open(corpus_out, "a", encoding="utf-8") as corpus_f, open(manifest_out, "a", encoding="utf-8") as manifest_f:
        for qf in glob.glob(os.path.join(queue_dir, "*.jsonl")):
            with open(qf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    url = rec.get("url", "")
                    if not is_allow_listed(url):
                        continue  # belt-and-suspenders; router already filters
                    src = source_for(url)
                    corpus_f.write(rec["text"].replace("\n", " ") + "\n")
                    manifest_f.write(json.dumps({
                        "url": url,
                        "source": src.key if src else "unknown",
                        "license": src.license if src else "unknown",
                        "queued_at": rec.get("queued_at"),
                    }) + "\n")
                    n += 1
            os.remove(qf)
    return n


def fetch_bulk(urls: Iterable[str], corpus_out: str, manifest_out: str) -> int:
    """Bulk-fetch a list of allow-listed URLs (e.g. a batch of Gutenberg
    book pages) straight into the corpus. Still robots.txt-gated and
    rate-limited via WebFetchConnector, still registry-gated via
    is_allow_listed."""
    fetcher = WebFetchConnector()
    os.makedirs(os.path.dirname(corpus_out) or ".", exist_ok=True)
    n = 0
    with open(corpus_out, "a", encoding="utf-8") as corpus_f, open(manifest_out, "a", encoding="utf-8") as manifest_f:
        for url in urls:
            if not is_allow_listed(url):
                continue
            result = fetcher.call(url, max_chars=1_000_000)
            if not result.ok or not result.text:
                continue
            src = source_for(url)
            corpus_f.write(result.text.replace("\n", " ") + "\n")
            manifest_f.write(json.dumps({
                "url": url, "source": src.key if src else "unknown",
                "license": src.license if src else "unknown",
            }) + "\n")
            n += 1
    return n


def _iter_documents(corpus_path: str, doc_mode: str) -> Iterator[str]:
    """Yield one document's text at a time.

    doc_mode="line"  : every line is its own document. This is the convention
                       fold_queue()/fetch_bulk() write (one web page per line).
    doc_mode="marker": documents are separated by lines starting with "<|doc"
                       (see chat.DOC_MARKER_PREFIX); the marker line itself is
                       dropped. Used for books and for the behavior corpus.
    """
    if doc_mode not in ("line", "marker"):
        raise ValueError(f"doc_mode must be 'line' or 'marker', got {doc_mode!r}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        if doc_mode == "line":
            for line in f:
                if line.strip():
                    yield line
            return
        buf: List[str] = []
        for line in f:
            if line.startswith(DOC_MARKER_PREFIX):
                if buf:
                    yield "".join(buf)
                buf = []
            else:
                buf.append(line)
        if buf:
            yield "".join(buf)


def build_shards(
    corpus_path: str,
    tokenizer_path: str | None,
    out_dir: str,
    val_fraction: float = 0.005,
    min_val_tokens: int = 2000,
    max_block_size: int = 2048,
    doc_mode: str = "line",
    max_val_tokens: int = 2_000_000,
    seed: int = 1337,
) -> Tuple[int, int]:
    """Tokenize `corpus_path` and write train.bin / val.bin uint16 memmaps,
    nanoGPT-style, readable by data/dataset.py. Returns (n_train, n_val).

    * <eos> goes after each DOCUMENT (see _iter_documents for what that means),
      not after every line. Books are hard-wrapped, so per-line EOS put one EOS
      roughly every 13 tokens and taught the model that text ends mid-sentence.
    * Validation is a set of whole, randomly chosen documents. It used to be
      the tail of the file, which for the behavior data was one entire category
      -- a validation set unlike anything in training.
    * Tokens stream to disk as they're produced; RAM use doesn't grow with the
      corpus (it used to hold every token in a Python list).

    `val_fraction` alone is only safe on large corpora (0.5% of 150K tokens is
    ~750, less than one tiny-tier block), so the val target is
    max(val_fraction * n, min_val_tokens, max_block_size + 1), capped at
    `max_val_tokens` (more validation than that just slows evals down) and at
    half the corpus. If whole-document selection can't hit that (one giant
    document), it falls back to holding out the tail of the token stream.
    """
    tok = load_tokenizer(tokenizer_path)
    if tok.vocab_size > 65536:
        raise ValueError(f"vocab_size {tok.vocab_size} doesn't fit the uint16 shard format")
    os.makedirs(out_dir, exist_ok=True)

    # pass 1: tokenize document by document straight into one temp file
    tmp_path = os.path.join(out_dir, "_all_tokens.tmp")
    lengths: List[int] = []
    with open(tmp_path, "wb") as tmp:
        for doc in _iter_documents(corpus_path, doc_mode):
            ids = tok.encode(doc, add_eos=True)
            np.asarray(ids, dtype=np.uint16).tofile(tmp)
            lengths.append(len(ids))
    n = int(sum(lengths))
    if n == 0:
        os.remove(tmp_path)
        raise ValueError(f"{corpus_path} contains no text")

    val_floor = max(min_val_tokens, max_block_size + 1)
    target = min(max(int(n * val_fraction), val_floor), max_val_tokens, n // 2)

    # choose whole documents for validation, in a deterministic shuffled order
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    val_docs, val_n = set(), 0
    for i in order:
        if val_n >= target:
            break
        val_docs.add(i)
        val_n += lengths[i]
    docs_ok = len(val_docs) < len(lengths) and val_n <= n // 2

    data = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    train_path, val_path = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
    if docs_ok:
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        n_train = n_val = 0
        with open(train_path, "wb") as ft, open(val_path, "wb") as fv:
            for i in range(len(lengths)):
                chunk = data[offsets[i]:offsets[i + 1]]
                if i in val_docs:
                    fv.write(chunk.tobytes()); n_val += len(chunk)
                else:
                    ft.write(chunk.tobytes()); n_train += len(chunk)
    else:
        split = n - target
        data[:split].tofile(train_path)
        data[split:].tofile(val_path)
        n_train, n_val = split, n - split
    del data
    try:
        os.remove(tmp_path)
    except OSError:
        pass  # Windows can keep a just-closed memmap locked for a moment; harmless leftover

    if n_val <= max_block_size:
        print(
            f"[zivawhite] warning: corpus only has {n:,} tokens -- val split ({n_val:,} tokens) is still "
            f"<= block_size ({max_block_size}). train.py's evaluate() will skip validation rather than "
            f"crash, but this corpus is really only big enough for a smoke test."
        )
    return n_train, n_val
