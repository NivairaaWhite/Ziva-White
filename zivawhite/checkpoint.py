"""
Crash-safe checkpointing.

A checkpoint is written to a temp file in the same directory and then
atomically renamed over the target -- so a process killed mid-write
(power loss, OOM kill, laptop lid closed) never leaves a corrupt
checkpoint on disk. We also keep the previous checkpoint as `*.bak`
until the new one is confirmed on disk.
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
    import torch  # lazy: path helpers below (resolve_serving_checkpoint, find_default_checkpoint) need no torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "saved_at": time.time(),
        "extra": extra or {},
    }
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".ckpt_tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _torch_load(path: str, map_location: str) -> Dict[str, Any]:
    """torch.load, safe first. A normal checkpoint is just tensors, numbers, strings and dicts, which
    weights_only=True loads WITHOUT running pickle's arbitrary-code machinery (so a downloaded or shared
    .pt file cannot execute code on load). Only a file that holds arbitrary Python objects (the int8 file
    from `compress` pickles a whole quantized module) falls back to the unsafe loader, loudly. Set
    ZIVAWHITE_ALLOW_PICKLE=0 to forbid that fallback entirely."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as safe_error:  # noqa: BLE001 - UnpicklingError, or RuntimeError on older torch
        if os.environ.get("ZIVAWHITE_ALLOW_PICKLE", "1") == "0":
            raise
        warnings.warn(
            f"{path} contains Python objects, not just weights, so it can only be loaded with pickle "
            f"(which can run arbitrary code). Only do this for files you created yourself. "
            f"[safe loader said: {str(safe_error)[:120]}]", RuntimeWarning, stacklevel=3)
        return torch.load(path, map_location=map_location, weights_only=False)


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
    os.replace(tmp, path)
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
    `compress` (a pickled, dynamically-quantized model). Quantized models are
    CPU-only in PyTorch, so for those the returned device is always "cpu"."""
    from .model import ModelConfig, ZivaGPT

    src = resolve_serving_checkpoint(path) if prefer_best else path
    ckpt = load_checkpoint(src, map_location="cpu")  # always CPU first; moved below
    if ckpt.get("quantized"):
        return ckpt["quantized_model"].eval(), ckpt, "cpu", src
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
    }
