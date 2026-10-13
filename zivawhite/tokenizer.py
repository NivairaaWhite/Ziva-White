"""
Tokenizer layer.

Primary path: a real trainable byte-level BPE via the `tokenizers`
library (Rust-backed, trains in seconds even on a laptop). A copy
trained once on the full 25-book bundled corpus ships with the
package at data/tokenizer/ (vocab_size=12000) and is the *default*
every entry point uses -- see load_tokenizer() below. This one change
is what actually gets ZivaGPT predicting whole subwords instead of
single bytes: on the bundled corpus the same text that costs 124
byte-level tokens costs 32 BPE tokens (~3.9 chars/token), so a fixed
block_size covers ~4x more real text and the model spends its (CPU-
limited) capacity learning language structure instead of re-deriving
"which bytes make a word" from scratch every run.

Fallback path: a pure-Python byte-level tokenizer (vocab size 256 + a
few specials) that needs zero extra dependencies and always works, so
ZivaWhite never hard-fails on a machine where `tokenizers` couldn't
be installed. Same interface either way -- only reached if the
`tokenizers` import genuinely fails, which shouldn't happen since it's
a hard requirements.txt dependency, not an optional extra.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import List, Optional

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
EXTRA_SPECIALS_FILE = "special_tokens.json"   # extra special tokens a tokenizer was trained with (beyond the four)
JSON_TOKENIZER_SIDECAR = "ziva_tokenizer.json"  # ids for a tokenizer.json-style tokenizer (imported models)
MAX_VOCAB = 65536                               # shards store token ids as uint16
THIS_DIR = os.path.dirname(__file__)
BUNDLED_TOKENIZER_DIR = os.path.join(THIS_DIR, "data", "tokenizer")


class ByteLevelTokenizer:
    """Zero-dependency fallback: every byte is a token id (+4 specials)."""

    vocab_size = 256 + len(SPECIAL_TOKENS)

    def __init__(self):
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [b + len(SPECIAL_TOKENS) for b in text.encode("utf-8")]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode(self, ids: List[int]) -> str:
        byte_vals = bytes(i - len(SPECIAL_TOKENS) for i in ids if i >= len(SPECIAL_TOKENS))
        return byte_vals.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"type": "byte_level", "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path: str) -> "ByteLevelTokenizer":
        return cls()


class BPETokenizer:
    """Thin wrapper around HuggingFace `tokenizers` ByteLevelBPETokenizer."""

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.vocab_size = self._tok.get_vocab_size()
        self.pad_id = self._tok.token_to_id("<pad>")
        self.bos_id = self._tok.token_to_id("<bos>")
        self.eos_id = self._tok.token_to_id("<eos>")
        self.unk_id = self._tok.token_to_id("<unk>")

    @classmethod
    def train(cls, files: List[str], vocab_size: int, out_dir: str,
              extra_special: Optional[List[str]] = None) -> "BPETokenizer":
        from tokenizers import ByteLevelBPETokenizer

        _check_vocab(vocab_size)
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=vocab_size, min_frequency=2,
                  special_tokens=SPECIAL_TOKENS + list(extra_special or []))
        return cls._finish_training(tok, out_dir, extra_special)

    @classmethod
    def train_from_iterator(cls, texts, vocab_size: int, out_dir: str,
                            extra_special: Optional[List[str]] = None) -> "BPETokenizer":
        """Train on any iterator of strings (e.g. a streamed dataset) without writing it to a file first."""
        from tokenizers import ByteLevelBPETokenizer

        _check_vocab(vocab_size)
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2, show_progress=True,
                                special_tokens=SPECIAL_TOKENS + list(extra_special or []))
        return cls._finish_training(tok, out_dir, extra_special)

    @classmethod
    def _finish_training(cls, tok, out_dir: str, extra_special: Optional[List[str]]) -> "BPETokenizer":
        os.makedirs(out_dir, exist_ok=True)
        tok.save_model(out_dir)
        if extra_special:
            with open(os.path.join(out_dir, EXTRA_SPECIALS_FILE), "w", encoding="utf-8") as f:
                json.dump(list(extra_special), f)
        return cls(tok)

    @classmethod
    def load(cls, out_dir: str) -> "BPETokenizer":
        from tokenizers import ByteLevelBPETokenizer

        tok = ByteLevelBPETokenizer(
            os.path.join(out_dir, "vocab.json"), os.path.join(out_dir, "merges.txt")
        )
        # vocab.json/merges.txt don't record which entries are *special*, so a
        # tokenizer rebuilt from them treats "<bos>" as ordinary text: decode()
        # leaks it into replies and encode() splits it into pieces. Re-register
        # the four specials (plus any extras it was trained with). They're already in
        # the vocab, so this reuses their ids and doesn't change vocab_size.
        extras: List[str] = []
        extras_path = os.path.join(out_dir, EXTRA_SPECIALS_FILE)
        if os.path.exists(extras_path):
            with open(extras_path, encoding="utf-8") as f:
                extras = list(json.load(f))
        tok.add_special_tokens(SPECIAL_TOKENS + extras)
        return cls(tok)

    def save_hf_json(self, path: str) -> None:
        """Write a single self-contained Hugging Face tokenizer.json (used by export_hf.py)."""
        self._tok.save(path)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Tokenize many documents at once (parallel in the Rust backend: the fast path for building shards)."""
        rows = [e.ids for e in self._tok.encode_batch(list(texts))]
        if add_bos:
            rows = [[self.bos_id] + r for r in rows]
        if add_eos:
            rows = [r + [self.eos_id] for r in rows]
        return rows

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str) -> None:
        self._tok.save_model(os.path.dirname(path) or ".")


def _check_vocab(vocab_size: int) -> None:
    if vocab_size > MAX_VOCAB:
        raise ValueError(f"vocab_size {vocab_size} > {MAX_VOCAB}: training shards store token ids as uint16")
    if vocab_size < 300:
        raise ValueError(f"vocab_size {vocab_size} is too small for a byte-level BPE (needs 256 bytes + specials + merges)")


class JSONTokenizer:
    """A tokenizer loaded from a Hugging Face `tokenizer.json` (any BPE/Unigram a Llama-family model ships
    with). Used for IMPORTED models, whose vocabulary is not ours. Same interface as the others. The special
    ids come from a small ziva_tokenizer.json sidecar that import_hf writes; a model with no BOS token (the
    GPT-2 / SmolLM2 convention) uses its end-of-text token as the document boundary."""

    def __init__(self, hf_tokenizer, ids: dict):
        self._tok = hf_tokenizer
        self.vocab_size = max(self._tok.get_vocab_size(), int(ids.get("vocab_size", 0) or 0))
        self.eos_id = ids.get("eos_id")
        self.bos_id = ids.get("bos_id") if ids.get("bos_id") is not None else self.eos_id
        self.pad_id = ids.get("pad_id") if ids.get("pad_id") is not None else self.eos_id
        self.unk_id = ids.get("unk_id") if ids.get("unk_id") is not None else 0
        if self.eos_id is None:
            raise ValueError("tokenizer sidecar must define eos_id")

    @classmethod
    def load(cls, path: str) -> "JSONTokenizer":
        from tokenizers import Tokenizer

        with open(os.path.join(path, JSON_TOKENIZER_SIDECAR), encoding="utf-8") as f:
            ids = json.load(f)
        return cls(Tokenizer.from_file(os.path.join(path, "tokenizer.json")), ids)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids   # we place bos/eos ourselves
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, path: str) -> None:
        self._tok.save(os.path.join(os.path.dirname(path) or ".", "tokenizer.json"))


def load_tokenizer(path: Optional[str] = None):
    """Load a BPE tokenizer. Resolution order:

    1. `path`, if given and it already holds a trained tokenizer
       (vocab.json/merges.txt) -- an explicit, user-trained tokenizer
       always wins.
    2. The tokenizer bundled with the package (data/tokenizer/,
       a general English byte-level BPE trained once offline). Used
       whenever `path` is None or points somewhere that doesn't have a
       trained tokenizer yet. This is what makes autolearn's base and
       conversation training and ad-hoc ingest all share one real BPE
       vocabulary *by default*, with no flag required and no risk of
       silently training different, incompatible vocabularies depending
       on which command happened to run first.
    3. The dependency-free byte-level tokenizer -- only reached if
       `tokenizers` isn't actually importable.

    Whatever a checkpoint was trained with must keep being used for
    that checkpoint (mixing vocabularies corrupts the model), so don't
    retrain over an existing tokenizer directory; train a new one
    somewhere else and pass it explicitly if you want a different vocab.
    """
    if path and os.path.exists(os.path.join(path, JSON_TOKENIZER_SIDECAR)) and os.path.exists(os.path.join(path, "tokenizer.json")):
        try:
            return JSONTokenizer.load(path)   # an imported model's own tokenizer
        except ImportError:
            warnings.warn("pip install tokenizers is required for this tokenizer", RuntimeWarning, stacklevel=2)
    for candidate in (path, BUNDLED_TOKENIZER_DIR):
        if candidate and os.path.exists(os.path.join(candidate, "vocab.json")):
            try:
                return BPETokenizer.load(candidate)
            except ImportError:
                # `tokenizers` isn't installed, so no candidate will load. A
                # checkpoint trained with the BPE vocab will NOT work with the
                # byte-level fallback (different vocab size), so say so.
                warnings.warn(
                    "the `tokenizers` package is not installed -- falling back to the byte-level "
                    "tokenizer (vocab 260). Checkpoints trained with the BPE tokenizer will not load "
                    "with it. Run: pip install tokenizers",
                    RuntimeWarning, stacklevel=2,
                )
                break
    return ByteLevelTokenizer()
