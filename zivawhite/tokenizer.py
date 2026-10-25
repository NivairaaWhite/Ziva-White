"""
Tokenizer layer.

Primary path: a real trainable byte-level BPE via the `tokenizers`
library (Rust-backed, trains in seconds even on a laptop). A general
English tokenizer (vocab_size=12000) ships with the package at
data/tokenizer/ as the *starting point* for a brand-new model when you
haven't chosen one. It is a default, not a requirement:

  * Every training run SAVES the tokenizer it used next to its checkpoint
    (runs/<run>/tokenizer/) and records a fingerprint of it in the
    checkpoint. A model therefore always carries its own vocabulary.
  * Everything that later loads a checkpoint (ask, serve, generate, eval,
    sft, dpo, export, the web server) finds that saved tokenizer by itself
    -- see resolve_tokenizer() below. You never have to repeat
    --tokenizer_dir for a model you trained, and the bundled 12k
    vocabulary is only ever used for a model that was trained with it.
  * To train on YOUR vocabulary instead, point `autolearn.tokenizer_dir`
    (or --tokenizer_dir) at a tokenizer folder before the first training run.

Why a BPE at all: on English text the same passage that costs 124
byte-level tokens costs about 32 BPE tokens (~3.9 chars/token), so a fixed
block_size covers ~4x more real text and the model spends its capacity on
language structure instead of re-deriving which bytes make a word.

Fallback path: a pure-Python byte-level tokenizer (vocab size 256 + a
few specials) that needs zero extra dependencies, so ZivaWhite never
hard-fails on a machine where `tokenizers` couldn't be installed. Same
interface either way -- only reached if the `tokenizers` import genuinely
fails, which shouldn't happen since it's a hard requirements.txt dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
EXTRA_SPECIALS_FILE = "special_tokens.json"   # extra special tokens a tokenizer was trained with (beyond the four)
JSON_TOKENIZER_SIDECAR = "ziva_tokenizer.json"  # ids for a tokenizer.json-style tokenizer (imported models)
MAX_VOCAB = 65536                               # shards store token ids as uint16
THIS_DIR = os.path.dirname(__file__)
BUNDLED_TOKENIZER_DIR = os.path.join(THIS_DIR, "data", "tokenizer")
RUN_TOKENIZER_DIRNAME = "tokenizer"            # runs/<run>/tokenizer/: the vocabulary a checkpoint was trained with
BYTE_LEVEL_MARKER = "byte_level.json"          # written into a run folder trained with the zero-dependency fallback
# the files that DEFINE a tokenizer (copied with a checkpoint, and hashed to fingerprint it)
_TOKENIZER_FILES = ("vocab.json", "merges.txt", EXTRA_SPECIALS_FILE, "tokenizer.json", JSON_TOKENIZER_SIDECAR, BYTE_LEVEL_MARKER)


class TokenizerNotFound(FileNotFoundError):
    """A tokenizer folder was named explicitly but holds no tokenizer."""


class TokenizerMismatch(RuntimeError):
    """The tokenizer found does not belong to the model it was found for."""


class ByteLevelTokenizer:
    """Zero-dependency fallback: every byte is a token id (+4 specials)."""

    vocab_size = 256 + len(SPECIAL_TOKENS)
    source_dir: Optional[str] = None      # a byte-level tokenizer has no files

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
        self.source_dir: Optional[str] = None      # the folder this was loaded from (set by load_tokenizer)
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
        self.source_dir: Optional[str] = None
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


def has_tokenizer(path: Optional[str]) -> bool:
    """True if `path` is a folder that holds a tokenizer ZivaWhite can load."""
    if not path or not os.path.isdir(path):
        return False
    here = lambda name: os.path.exists(os.path.join(path, name))  # noqa: E731
    return here("vocab.json") or (here("tokenizer.json") and here(JSON_TOKENIZER_SIDECAR)) or here(BYTE_LEVEL_MARKER)


def load_tokenizer(path: Optional[str] = None):
    """Load a tokenizer.

    * `path` given: that folder's tokenizer, always. A folder that holds none is an ERROR (TokenizerNotFound),
      not a quiet switch to the bundled vocabulary: a typo in --tokenizer_dir used to train a model on a
      different vocabulary than you asked for without a word.
    * `path` None: the tokenizer bundled with the package (a general English byte-level BPE, 12,000 tokens).
      That is the starting point for a NEW model only. For a model you already trained, don't call this
      directly: resolve_tokenizer() finds the one saved with the checkpoint.
    * Last resort, only if the `tokenizers` package isn't importable: the dependency-free byte-level tokenizer.

    Whatever a checkpoint was trained with must keep being used for that checkpoint (mixing vocabularies
    corrupts the model). Train a new vocabulary somewhere else and point at it; never retrain over a folder
    that already has checkpoints.

    The returned object has `.source_dir`: the folder it was loaded from (None for the byte-level fallback).
    """
    if path:
        path = os.fspath(path)
        if not has_tokenizer(path):
            raise TokenizerNotFound(
                f"no tokenizer in {path!r}: expected vocab.json + merges.txt (a trained BPE) or tokenizer.json + "
                f"{JSON_TOKENIZER_SIDECAR} (an imported model's). Train one first (`python -m zivawhite.app "
                f"tokenizer --help`), or leave the setting empty to use the bundled general-English tokenizer.")
    if path and os.path.exists(os.path.join(path, JSON_TOKENIZER_SIDECAR)) and os.path.exists(os.path.join(path, "tokenizer.json")):
        try:
            tok = JSONTokenizer.load(path)   # an imported model's own tokenizer
            tok.source_dir = path
            return tok
        except ImportError:
            warnings.warn("pip install tokenizers is required for this tokenizer", RuntimeWarning, stacklevel=2)
    if path and not os.path.exists(os.path.join(path, "vocab.json")) and os.path.exists(os.path.join(path, BYTE_LEVEL_MARKER)):
        tok = ByteLevelTokenizer()
        tok.source_dir = path
        return tok
    for candidate in (path, BUNDLED_TOKENIZER_DIR):
        if candidate and os.path.exists(os.path.join(candidate, "vocab.json")):
            try:
                tok = BPETokenizer.load(candidate)
                tok.source_dir = candidate
                return tok
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


# ------------------------------------------------------------------ a model keeps its own tokenizer
def tokenizer_fingerprint(path: Optional[str]) -> Optional[str]:
    """sha256 over the files that define the tokenizer in `path` (None if there is none). Two folders with the same
    fingerprint encode text identically; the same vocabulary SIZE does not prove that."""
    if not has_tokenizer(path):
        return None
    h = hashlib.sha256()
    for name in sorted(_TOKENIZER_FILES):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            h.update(name.encode() + b"\0")
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


def _same_dir(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b) and os.path.realpath(a) == os.path.realpath(b)


def describe_tokenizer(tok) -> Dict[str, Any]:
    """Facts about a loaded tokenizer, small enough to store in a checkpoint's metadata."""
    src = getattr(tok, "source_dir", None)
    kind = "byte" if isinstance(tok, ByteLevelTokenizer) else "json" if isinstance(tok, JSONTokenizer) else "bpe"
    return {"kind": kind, "vocab_size": int(tok.vocab_size), "sha256": tokenizer_fingerprint(src),
            "bundled": _same_dir(src, BUNDLED_TOKENIZER_DIR)}


def run_tokenizer_dir(checkpoint_path: Optional[str]) -> Optional[str]:
    """`<folder holding the checkpoint>/tokenizer`, if that holds a tokenizer. This is where training saves it."""
    if not checkpoint_path:
        return None
    folder = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), RUN_TOKENIZER_DIRNAME)
    return folder if has_tokenizer(folder) else None


def save_run_tokenizer(tok, run_dir: str, replace: bool = False) -> Dict[str, Any]:
    """Save the tokenizer `tok` was loaded from into `<run_dir>/tokenizer/`, so the checkpoints in `run_dir`
    carry their own vocabulary. Returns the metadata to store in the checkpoint (see describe_tokenizer).

    A folder that already holds a DIFFERENT tokenizer is never overwritten unless `replace=True` (a brand-new
    run owns its folder): swapping the vocabulary under existing checkpoints would silently corrupt them."""
    dest = os.path.join(run_dir, RUN_TOKENIZER_DIRNAME)
    meta = describe_tokenizer(tok)
    meta["dir"] = RUN_TOKENIZER_DIRNAME
    src = getattr(tok, "source_dir", None)
    if _same_dir(src, dest):
        return meta
    existing = tokenizer_fingerprint(dest)
    if existing and not replace and existing != meta["sha256"]:
        raise TokenizerMismatch(
            f"{dest} already holds a different tokenizer than the one in use. The checkpoints in {run_dir} were "
            f"trained with it, so it cannot be replaced. Use a new --out_dir, or pass the original --tokenizer_dir.")
    os.makedirs(dest, exist_ok=True)
    for name in _TOKENIZER_FILES:               # clear the old definition so no stale file survives next to the new one
        stale = os.path.join(dest, name)
        if os.path.isfile(stale):
            os.remove(stale)
    if src is None:                             # the byte-level fallback has no files: leave a marker instead
        with open(os.path.join(dest, BYTE_LEVEL_MARKER), "w", encoding="utf-8") as f:
            json.dump({"type": "byte_level", "vocab_size": int(tok.vocab_size)}, f)
        return meta
    for name in _TOKENIZER_FILES:
        if os.path.isfile(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(dest, name))
    return meta


def expectation_from_checkpoint(ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a loaded checkpoint says about the tokenizer it needs: {"vocab_size", "sha256"} (either may be None;
    checkpoints written before tokenizers were saved carry no fingerprint)."""
    ckpt = ckpt or {}
    meta = (ckpt.get("extra") or {}).get("tokenizer") or {}
    return {"vocab_size": (ckpt.get("config") or {}).get("vocab_size") or meta.get("vocab_size"),
            "sha256": meta.get("sha256")}


def _fits(tok, expect: Optional[Dict[str, Any]]) -> bool:
    if not expect:
        return True
    want_vocab, want_sha = expect.get("vocab_size"), expect.get("sha256")
    if want_vocab is not None and int(want_vocab) != int(tok.vocab_size):
        return False
    if want_sha:
        have = tokenizer_fingerprint(getattr(tok, "source_dir", None))
        if have and have != want_sha:
            return False
    return True


def resolve_tokenizer(explicit: Optional[str] = None, checkpoint: Optional[str] = None,
                      configured: Optional[str] = None, expect: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """Find the tokenizer that belongs to a model. Returns (tokenizer, how it was found). First match wins:

      1. `explicit`    -- --tokenizer_dir. Always honoured; it is an error if it does not fit the checkpoint.
      2. the tokenizer SAVED WITH the checkpoint: <checkpoint folder>/tokenizer/   (training writes it there)
      3. `configured`  -- autolearn.tokenizer_dir, for checkpoints from before tokenizers were saved
      4. the bundled general-English tokenizer, but only if the checkpoint was trained with it

    `expect` (see expectation_from_checkpoint) is what the checkpoint needs; a candidate that doesn't match it in
    vocabulary size or fingerprint is skipped, so a model can never be paired with the wrong vocabulary by default.
    Raises TokenizerMismatch with the places it looked when nothing fits.
    """
    if explicit:
        tok = load_tokenizer(explicit)
        if not _fits(tok, expect):
            raise TokenizerMismatch(
                f"{explicit} has a vocabulary of {tok.vocab_size:,} tokens"
                + ("" if not expect or expect.get("vocab_size") is None else
                   f" but this checkpoint was trained with {int(expect['vocab_size']):,}")
                + " (or a different tokenizer of the same size). Pass the tokenizer folder this model was trained with. "
                  "To use a different tokenizer you must train a NEW model: use a new --out_dir (or delete the old run).")
        return tok, f"--tokenizer_dir {explicit}"
    candidates: List[Tuple[Optional[str], str]] = []
    saved = run_tokenizer_dir(checkpoint)
    if saved:
        candidates.append((saved, f"saved with the checkpoint ({saved})"))
    if configured:
        candidates.append((configured, f"autolearn.tokenizer_dir ({configured})"))
    candidates.append((None, "bundled general-English tokenizer"))
    looked: List[str] = []
    for path, how in candidates:
        if path is not None and not has_tokenizer(path):
            looked.append(f"{how}: no tokenizer there")
            continue
        tok = load_tokenizer(path)
        if _fits(tok, expect):
            return tok, how
        looked.append(f"{how}: {tok.vocab_size:,} tokens")
    need = (f"{int(expect['vocab_size']):,} tokens" if expect and expect.get("vocab_size") is not None else "a different vocabulary")
    raise TokenizerMismatch(
        f"this checkpoint needs a tokenizer with {need}, and none of the places searched has it ({'; '.join(looked)}). "
        f"Copy the tokenizer it was trained with to {os.path.join(os.path.dirname(os.path.abspath(checkpoint or '.')), RUN_TOKENIZER_DIRNAME)}, "
        f"or pass --tokenizer_dir. (If every candidate has 260 tokens, the `tokenizers` package is missing: pip install tokenizers.)")


def tokenizer_for_checkpoint(checkpoint_path: Optional[str], ckpt: Optional[Dict[str, Any]], explicit: Optional[str] = None,
                             configured: Optional[str] = None) -> Tuple[Any, str]:
    """resolve_tokenizer() for a loaded checkpoint payload: reads the vocabulary size / fingerprint it needs."""
    return resolve_tokenizer(explicit, checkpoint_path, configured, expectation_from_checkpoint(ckpt))


def pick_training_tokenizer(explicit: Optional[str], seed_path: Optional[str], seed_ckpt: Optional[Dict[str, Any]]) -> Tuple[Any, str]:
    """Which tokenizer should a training run use? (train.py calls this; it is a function so the rule can be tested.)

      * `explicit` (--tokenizer_dir)                       always, as long as it fits any checkpoint being continued;
      * continuing a run (resume) or starting from one
        (--init_from): `seed_ckpt` loaded from `seed_path`  the vocabulary that checkpoint was trained with -- the one
                                                           saved in its folder -- never anything else;
      * a brand-new model with no seed                     the bundled general-English tokenizer.
    Raises TokenizerNotFound / TokenizerMismatch with an explanation when it cannot be done safely."""
    if seed_ckpt is not None:
        return tokenizer_for_checkpoint(seed_path, seed_ckpt, explicit=explicit)
    if explicit:
        return load_tokenizer(explicit), f"--tokenizer_dir {explicit}"
    return load_tokenizer(None), "bundled general-English tokenizer (the default for a new model)"


def tokenizer_or_exit(checkpoint_path: Optional[str], ckpt: Optional[Dict[str, Any]], explicit: Optional[str] = None) -> Any:
    """For command-line tools: the tokenizer that belongs to this checkpoint, or a clear one-paragraph exit message.
    (Also prints nothing on success: the tool's own output stays clean.)"""
    try:
        return tokenizer_for_checkpoint(checkpoint_path, ckpt, explicit=explicit)[0]
    except (TokenizerNotFound, TokenizerMismatch) as e:
        raise SystemExit(f"[zivawhite] {e}") from None
