"""
Crash-safe checkpointing.

A checkpoint is written to a temp file in the same directory, flushed to
disk (fsync), and only then atomically renamed over the target -- so a
process killed mid-write (power loss, OOM kill, laptop lid closed) never
leaves a corrupt checkpoint on disk. The previous checkpoint is kept as
`*.bak` (a hard link where the filesystem allows it, so keeping it costs
no copy of a multi-gigabyte file).

Loading is safe by default: checkpoints are read with torch's
weights_only loader, which cannot run code. A file that needs full pickle
is refused unless you opt in with ZIVAWHITE_ALLOW_PICKLE=1.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

SAFETENSORS_SUFFIX = ".safetensors"


def save_checkpoint(path: str, *, model, optimizer=None, step: int, cfg, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "saved_at": time.time(),
        "extra": extra or {},
    }
    write_payload(path, payload)


def write_payload(path: str, payload: Dict[str, Any]) -> None:
    """Atomically and durably write a checkpoint dict: temp file -> fsync -> rename over `path`."""
    import torch

    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".ckpt_tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())            # the bytes are on disk before the name points at them
        _keep_backup(path)
        os.replace(tmp_path, path)          # atomic: readers see the old file or the new one, never half of one
        _fsync_dir(dir_name)                # and the rename itself survives a power cut
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _keep_backup(path: str) -> None:
    """Keep the current file as `<path>.bak` without copying it: a hard link to the same data. The new checkpoint
    is written to a different file and renamed over `path`, so the link keeps the OLD content. Filesystems that
    cannot hard-link (FAT/exFAT, some network drives) fall back to a copy."""
    if not os.path.exists(path):
        return
    backup, staging = path + ".bak", path + ".bak.tmp"
    try:
        if os.path.exists(staging):
            os.remove(staging)
        os.link(path, staging)
        os.replace(staging, backup)
    except OSError:
        if os.path.exists(staging):
            os.remove(staging)
        shutil.copy2(path, backup)


def _fsync_dir(dir_name: str) -> None:
    """Flush a directory entry (POSIX only; Windows has no equivalent and doesn't need it)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(dir_name, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class UnsafeCheckpointError(RuntimeError):
    """The file can't be read by the safe loader, and loading it with pickle has not been allowed."""


def pickle_allowed() -> bool:
    """Full pickle can run arbitrary code, so it is OFF unless you set ZIVAWHITE_ALLOW_PICKLE=1 (only for files
    you made yourself)."""
    return os.environ.get("ZIVAWHITE_ALLOW_PICKLE", "0").strip().lower() in ("1", "true", "yes")


def load_torch_file(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """torch.load with weights_only=True: tensors, numbers, strings and dicts only, so a downloaded or shared
    .pt file cannot execute code on load. A file that holds arbitrary Python objects (an int8 file written by an
    older `compress`, which pickled a whole module) is refused with an explanation, unless
    ZIVAWHITE_ALLOW_PICKLE=1 says you trust it."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_error:  # noqa: BLE001 - UnpicklingError, or RuntimeError on older torch / damaged files
        if not pickle_allowed():
            raise UnsafeCheckpointError(
                f"{path} could not be loaded with the safe loader ({type(safe_error).__name__}: {str(safe_error)[:140]}). "
                f"That means it is damaged, or it holds Python objects rather than just weights (an old int8 file: "
                f"re-run `compress` on the float checkpoint to write the safe format). If you made this file yourself "
                f"and trust it, set ZIVAWHITE_ALLOW_PICKLE=1 to load it with pickle.") from safe_error
        warnings.warn(
            f"{path} is being loaded with pickle (ZIVAWHITE_ALLOW_PICKLE=1), which can run arbitrary code. "
            f"Only do this for files you created yourself. [safe loader said: {str(safe_error)[:120]}]",
            RuntimeWarning, stacklevel=3)
        return torch.load(path, map_location=map_location, weights_only=False)


_torch_load = load_torch_file      # the old private name


def save_safetensors(path: str, *, model_state: Dict[str, Any], cfg, step: int = 0,
                     extra: Optional[Dict[str, Any]] = None) -> str:
    """Write weights in the safetensors format: no pickle at all, memory-mappable, loads in a blink.
    `path` is a .safetensors file (or a folder, which gets model.safetensors). The config, step and
    `extra` metadata travel inside the same file's header, so it is self-contained; load_checkpoint()
    reads it back into the same payload shape as a .pt file. The tied output embedding is stored once."""
    from safetensors.torch import save_file

    if not path.endswith(SAFETENSORS_SUFFIX):
        path = os.path.join(path, "model.safetensors")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in model_state.items() if k != "head.weight"}
    cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    metadata = {"format": "zivagpt", "config": json.dumps(cfg_dict), "step": str(int(step)),
                "extra": json.dumps(extra or {}, default=str), "saved_at": str(time.time())}
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path) or ".")
    return path


def _load_safetensors_payload(path: str, map_location: str) -> Dict[str, Any]:
    from safetensors import safe_open

    state: Dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            state[key] = f.get_tensor(key)
    if "tok_emb.weight" in state and "head.weight" not in state:
        state["head.weight"] = state["tok_emb.weight"]          # tied: stored once, restored here
    if map_location not in (None, "cpu"):
        state = {k: v.to(map_location) for k, v in state.items()}
    return {"model_state": state, "optimizer_state": None, "step": int(meta.get("step", 0)),
            "config": json.loads(meta["config"]) if "config" in meta else {},
            "saved_at": float(meta.get("saved_at", 0.0)), "extra": json.loads(meta.get("extra", "{}"))}


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by save_checkpoint (.pt) or save_safetensors (.safetensors), returning
    the same payload dict either way. A damaged .pt falls back to its .bak."""
    if path.endswith(SAFETENSORS_SUFFIX):
        return _load_safetensors_payload(path, map_location)
    try:
        return _torch_load(path, map_location)
    except Exception:
        backup = path + ".bak"
        if os.path.exists(backup):
            return _torch_load(backup, map_location)
        raise


def resolve_serving_checkpoint(path: str) -> str:
    """For loading a model to actually talk to (ask/serve), prefer the sibling
    `checkpoint_best.pt` -- but ONLY when `path` is the rolling `checkpoint.pt`.
    train.py only (re)writes checkpoint_best.pt when validation loss improves,
    so what gets served can only get better run over run, even though the
    rolling checkpoint.pt keeps moving (and can temporarily get worse mid-run).

    An explicitly named file (my_model.pt, checkpoint_int8.pt, ...) is always
    loaded as-is. This used to substitute checkpoint_best.pt for ANY filename in
    the same folder, silently ignoring what you asked for.

    Do NOT use this for --resume: continuing training must always pick up the
    literal last state (optimizer state, exact step count), not the best-so-far
    snapshot, or the step counter and optimizer momentum go inconsistent."""
    if os.path.basename(path) != "checkpoint.pt":
        return path
    best = os.path.join(os.path.dirname(path) or ".", "checkpoint_best.pt")
    return best if os.path.exists(best) else path


# Where `ask` / `serve` / `generate` look when you don't pass --checkpoint.
# The behavior-tuned run is listed first: it is what you want to talk to.
DEFAULT_CHECKPOINTS = (
    os.path.join("runs", "ziva_behavior", "checkpoint.pt"),
    os.path.join("runs", "ziva", "checkpoint.pt"),
)


def find_default_checkpoint(root: str = ".", run_dirs: Optional[Iterable[str]] = None) -> Optional[str]:
    """First run folder that holds a checkpoint. `run_dirs` (from configs/ziva.yaml:
    autolearn.chat_run, autolearn.base_run) replaces the built-in order when given."""
    candidates = [os.path.join(d, "checkpoint.pt") for d in run_dirs] if run_dirs else DEFAULT_CHECKPOINTS
    for rel in candidates:
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if os.path.exists(path) or os.path.exists(os.path.join(os.path.dirname(path), "checkpoint_best.pt")):
            return path
        st = os.path.join(os.path.dirname(path), "model" + SAFETENSORS_SUFFIX)   # a folder holding only an export
        if os.path.exists(st):
            return st
    return None


def load_model(path: str, device: str = "cpu", prefer_best: bool = True):
    """Build a ready-to-run model from a checkpoint file. Returns
    (model, checkpoint_payload, device_actually_used, file_actually_loaded).

    Handles both a normal checkpoint (config + state_dict) and an int8 file from
    `compress` (int8 weights + scales: plain tensors, no pickle; the model is rebuilt and dynamically quantized
    on load). Quantized models are CPU-only in PyTorch, so for those the returned device is always "cpu".
    A legacy int8 file that pickled a whole module loads only with ZIVAWHITE_ALLOW_PICKLE=1."""
    from .model import ModelConfig, ZivaGPT

    src = resolve_serving_checkpoint(path) if prefer_best else path
    ckpt = load_checkpoint(src, map_location="cpu")  # always CPU first; moved below
    if ckpt.get("quantized"):
        if "quantized_model" in ckpt:               # legacy pickled module (only reachable when pickle was allowed)
            return ckpt["quantized_model"].eval(), ckpt, "cpu", src
        from .compress import build_int8_model

        return build_int8_model(ckpt), ckpt, "cpu", src
    model = ZivaGPT(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt, device, src


def describe_checkpoint(path: str) -> Dict[str, Any]:
    """One-line-able facts about a checkpoint file, for `app.py inspect` and /v1/health."""
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    extra = ckpt.get("extra", {})
    return {
        "file": path,
        "n_layer": cfg.get("n_layer"), "n_embd": cfg.get("n_embd"), "block_size": cfg.get("block_size"),
        "n_head": cfg.get("n_head"), "n_kv_head": cfg.get("n_kv_head") or cfg.get("n_head"),
        "vocab_size": cfg.get("vocab_size"),
        "step": ckpt.get("step"),
        "behavior_tuned": bool(extra.get("behavior_tuned", False)),
        "val_loss": extra.get("val_loss", extra.get("best_val_loss")),
        "quantized": bool(ckpt.get("quantized", False)),
        "saved_at": ckpt.get("saved_at"),
        "tokenizer": _describe_tokenizer_of(path, extra, cfg.get("vocab_size")),
    }


def _describe_tokenizer_of(path: str, extra: Dict[str, Any], vocab_size: Any) -> str:
    """One line: which tokenizer goes with this checkpoint and where it is (see tokenizer.resolve_tokenizer)."""
    from .tokenizer import run_tokenizer_dir, tokenizer_fingerprint

    meta = extra.get("tokenizer") or {}
    saved = run_tokenizer_dir(path)
    if saved:
        fp = (tokenizer_fingerprint(saved) or "")[:10]
        fits = " (matches the checkpoint)" if not meta.get("sha256") or meta["sha256"] == tokenizer_fingerprint(saved) else \
            " (DOES NOT match the one this checkpoint was trained with!)"
        return f"{saved}  [{vocab_size:,} tokens, id {fp}]{fits}" if vocab_size else saved
    kind = "the bundled general-English tokenizer" if meta.get("bundled") else "a tokenizer that was not saved with it"
    return (f"none saved next to the checkpoint -- trained with {kind}; the bundled/configured tokenizer is used if its size "
            f"({vocab_size:,}) fits, otherwise pass --tokenizer_dir" if vocab_size else "none saved next to the checkpoint")
