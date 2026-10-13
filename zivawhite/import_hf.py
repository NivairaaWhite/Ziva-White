"""
import_hf.py - start from an open-weights model instead of from random numbers.

    # 1. download any plain Llama-architecture model you are licensed to use (e.g. a SmolLM2 checkpoint)
    huggingface-cli download HuggingFaceTB/SmolLM2-360M --local-dir models/smollm2-360m
    # 2. convert it to a ZivaWhite checkpoint
    python -m zivawhite.import_hf --src models/smollm2-360m --out runs/base
    # 3. fine-tune with YOUR tools: SFT, DPO, long-context, evaluation, serving ...
    python -m zivawhite.app sft --init_from runs/base/checkpoint.pt --tokenizer_dir runs/base/tokenizer ...

Why this exists: a model trained from scratch here sees a few billion tokens at most; the open models people run
under Ollama saw trillions. Importing their weights is the one thing that closes most of that gap without a
data-center, and everything downstream (chat tuning, DPO, serving, export) then works unchanged.

What can be imported: `model_type: llama` with plain rotary embeddings, SwiGLU, no biases and tied embeddings
(Llama-2-style models, SmolLM2, many others). What cannot, and is refused with the reason: Qwen2 (attention
biases), Llama-3.1/3.2 ('llama3' rope scaling), untied output embeddings (Llama-3-8B), and vocabularies over
65,536 entries (shards store token ids as uint16; Llama-3's 128k vocabulary does not fit).

LICENSE: imported weights keep the license they came with (Apache-2.0, Llama Community License, ...).
That license governs whatever you publish afterwards. Check it before you do.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from .hf_compat import hf_to_zivagpt_state, zivagpt_config_from_hf
from .tokenizer import JSON_TOKENIZER_SIDECAR, MAX_VOCAB


def weight_files(src: str) -> List[str]:
    """The .safetensors files that make up a model folder (a single file, or the shards its index lists)."""
    index = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(src, n) for n in names]
    single = os.path.join(src, "model.safetensors")
    if os.path.exists(single):
        return [single]
    found = sorted(n for n in os.listdir(src) if n.endswith(".safetensors"))
    if not found:
        raise FileNotFoundError(f"no .safetensors weights in {src} (pickle .bin weights are not imported, on purpose)")
    return [os.path.join(src, n) for n in found]


def _first_id(value) -> Optional[int]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return int(value) if value is not None else None


def tokenizer_sidecar(hf_cfg: Dict[str, Any], vocab_size: int) -> Dict[str, Any]:
    """The special-token ids ZivaWhite needs, read from the model's own config.json."""
    eos = _first_id(hf_cfg.get("eos_token_id"))
    if eos is None:
        raise ValueError("config.json has no eos_token_id; cannot tell where documents end")
    return {"eos_id": eos, "bos_id": _first_id(hf_cfg.get("bos_token_id")), "pad_id": _first_id(hf_cfg.get("pad_token_id")),
            "unk_id": None, "vocab_size": vocab_size}


def import_hf(src: str, out_dir: str, *, block_size: Optional[int] = None) -> Dict[str, Any]:
    import torch
    from safetensors import safe_open

    from .checkpoint import save_checkpoint
    from .model import ZivaGPT

    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        hf_cfg = json.load(f)
    cfg = zivagpt_config_from_hf(hf_cfg, block_size=block_size)
    if cfg.vocab_size > MAX_VOCAB:
        raise ValueError(f"vocabulary of {cfg.vocab_size:,} does not fit ZivaWhite's uint16 shard format "
                         f"(max {MAX_VOCAB:,}); choose a model with a smaller vocabulary")
    if not os.path.exists(os.path.join(src, "tokenizer.json")):
        raise FileNotFoundError(f"{src}/tokenizer.json is missing (needed to tokenize training data)")

    hf_state: Dict[str, Any] = {}
    for path in weight_files(src):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_state[key] = f.get_tensor(key)
    state = hf_to_zivagpt_state(hf_state, cfg)
    state["head.weight"] = state["tok_emb.weight"]
    state = {k: v.float() for k, v in state.items()}          # fp32 master weights for training
    model = ZivaGPT(cfg)
    model.load_state_dict(state, strict=True)                 # any shape mistake fails loudly here

    os.makedirs(os.path.join(out_dir, "tokenizer"), exist_ok=True)
    shutil.copy2(os.path.join(src, "tokenizer.json"), os.path.join(out_dir, "tokenizer", "tokenizer.json"))
    with open(os.path.join(out_dir, "tokenizer", JSON_TOKENIZER_SIDECAR), "w", encoding="utf-8") as f:
        json.dump(tokenizer_sidecar(hf_cfg, cfg.vocab_size), f, indent=2)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    save_checkpoint(ckpt, model=model, step=0, cfg=cfg,
                    extra={"behavior_tuned": False, "imported_from": os.path.abspath(src), "val_loss": None})
    return {"checkpoint": ckpt, "tokenizer_dir": os.path.join(out_dir, "tokenizer"), "params": cfg.total_params(),
            "config": cfg.to_dict()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Import a Llama-architecture Hugging Face model as a ZivaWhite checkpoint")
    p.add_argument("--src", required=True, help="folder with config.json, tokenizer.json and *.safetensors")
    p.add_argument("--out", required=True, help="run folder to create (checkpoint.pt + tokenizer/)")
    p.add_argument("--block_size", type=int, default=None, help="context length to train/run at (default min(model's, 2048))")
    a = p.parse_args(argv)
    res = import_hf(a.src, a.out, block_size=a.block_size)
    print(f"imported {res['params'] / 1e6:.1f}M parameters -> {res['checkpoint']}")
    print(f"tokenizer: {res['tokenizer_dir']}  (pass it as --tokenizer_dir everywhere)")
    print("It is a BASE model. Next: app.py sft (chat tuning), then optionally app.py dpo. "
          "Remember the weights keep their ORIGINAL license.")


if __name__ == "__main__":
    main()
