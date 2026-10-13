"""
compress.py - shrink a trained checkpoint for fast CPU inference.

Uses PyTorch's dynamic int8 quantization (torch.ao.quantization.
quantize_dynamic) on the model's Linear layers. This is deliberately
the *only* quantization scheme used here: it's CPU-only in PyTorch (no
CUDA kernel for it), needs no calibration dataset or extra training
step, and applies directly to the module ZivaWhite already runs -- a
genuinely good fit for "make this fast in the one environment it
actually needs to run well in," rather than a general GPU-serving
compression scheme bolted on for its own sake.

Typical result on a "tiny"/"small" tier ZivaGPT: roughly 4x smaller on
disk (int8 weights instead of fp32 in every Linear layer, which is
most of the parameters) and faster generation on CPU, since int8
matmul moves a quarter of the memory traffic fp32 does. The tradeoff
is a small, measurable quality cost -- `compare_perplexity()` below
measures it on the model's own held-out val shard rather than
asserting a number.

The quantized model is saved as a whole pickled nn.Module, not a
state_dict: PyTorch's dynamically-quantized Linear layers pack their
weights into an opaque format with no plain "weight" tensor to reload
into a fresh, unquantized ZivaGPT, so the usual state_dict round trip
(checkpoint.py) doesn't apply here. Loading it back just needs
`zivawhite.model` importable, same as any other pickle of a project's
own module.
"""
from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from dataclasses import asdict
from typing import Optional, Tuple

import torch

from .checkpoint import load_checkpoint, resolve_serving_checkpoint
from .data.dataset import get_batch
from .model import ModelConfig, ZivaGPT


def quantize_checkpoint(checkpoint_path: str, out_path: str) -> Tuple[int, int, "ZivaGPT"]:
    """Load `checkpoint_path` (preferring checkpoint_best.pt if present --
    same rule as serving, see checkpoint.resolve_serving_checkpoint),
    dynamically int8-quantize its Linear layers, and save the result to
    `out_path`. Returns (original_bytes, compressed_bytes, quantized_model)."""
    src_path = resolve_serving_checkpoint(checkpoint_path)
    ckpt = load_checkpoint(src_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = ZivaGPT(cfg).to("cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # PyTorch's own dynamic-quantization internals still call a
    # tensor-creation function it has marked deprecated for removal in a
    # *future* release (not this one) -- nothing here calls the
    # deprecated API directly, so silence that specific noise rather than
    # let it look like something's wrong with this code.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.nn.quantized")
        quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "quantized_model": quantized,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "step": ckpt["step"],
        "quantized": True,
        "extra": ckpt.get("extra", {}),
    }, out_path)

    original_bytes = os.path.getsize(src_path)
    compressed_bytes = os.path.getsize(out_path)
    return original_bytes, compressed_bytes, quantized


@torch.no_grad()
def compare_perplexity(original_model, quantized_model, data_dir: str, block_size: int,
                        batch_size: int = 8, eval_iters: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """Perplexity of both models on the same held-out val batches, so the
    quality cost of quantization is a measured number, not a guess.
    Returns (original_ppl, quantized_ppl); either can be None if the val
    shard is too small (same convention as train.evaluate)."""

    def _ppl(m) -> Optional[float]:
        losses = []
        for _ in range(eval_iters):
            try:
                x, y = get_batch(data_dir, "val", block_size, batch_size, "cpu")
            except ValueError:
                break
            _, loss = m(x, y)
            losses.append(loss.item())
        return math.exp(min(sum(losses) / len(losses), 20)) if losses else None

    return _ppl(original_model), _ppl(quantized_model)


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamically int8-quantize a checkpoint for fast CPU inference.")
    p.add_argument("--checkpoint", default="runs/ziva/checkpoint.pt")
    p.add_argument("--out", default=None, help="default: checkpoint_int8.pt next to --checkpoint")
    p.add_argument("--data_dir", default=None, help="if given, also reports before/after val perplexity on this shard")
    args = p.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint) or ".", "checkpoint_int8.pt")

    src_path = resolve_serving_checkpoint(args.checkpoint)
    ckpt = load_checkpoint(src_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    original = ZivaGPT(cfg).to("cpu")
    original.load_state_dict(ckpt["model_state"])
    original.eval()

    original_bytes, compressed_bytes, quantized = quantize_checkpoint(args.checkpoint, out_path)
    print(f"[zivawhite] {src_path}: {original_bytes:,} bytes -> {out_path}: {compressed_bytes:,} bytes "
          f"({original_bytes / max(compressed_bytes, 1):.1f}x smaller)")

    prompt = torch.zeros((1, 1), dtype=torch.long)
    for name, m in (("original", original), ("quantized", quantized)):
        t0 = time.time()
        m.generate(prompt.clone(), max_new_tokens=40, temperature=0.8)
        print(f"[zivawhite] {name} generation: {time.time() - t0:.2f}s for 40 tokens")

    if args.data_dir:
        orig_ppl, quant_ppl = compare_perplexity(original, quantized, args.data_dir, cfg.block_size)
        if orig_ppl is not None:
            print(f"[zivawhite] val perplexity: original {orig_ppl:.1f} -> quantized {quant_ppl:.1f}")
        else:
            print("[zivawhite] val shard too small to measure perplexity")


if __name__ == "__main__":
    main()
