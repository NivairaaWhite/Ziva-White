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

The int8 file stores NO pickled code. Every Linear weight (except the output head, which is tied to the
embedding) is written as an int8 tensor plus one float scale, next to the other parameters as ordinary tensors.
Loading it (checkpoint.load_model -> build_int8_model) dequantizes those into a normal model and runs
quantize_dynamic on it, so the file is ~4x smaller and loads with torch's safe weights_only loader. Older files
that pickled a whole quantized module still load, but only with ZIVAWHITE_ALLOW_PICKLE=1.
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

from .checkpoint import load_checkpoint, resolve_serving_checkpoint, write_payload
from .data.dataset import get_batch
from .model import ModelConfig, ZivaGPT

INT8_FORMAT = "int8-per-tensor-v1"
_INT8_QMAX = 127.5      # symmetric qint8: scale = max|w| / 127.5, the same rule torch's per-tensor observer uses


def pack_int8_state(model) -> Tuple[dict, dict]:
    """(state, scales): the model's parameters with every Linear weight (but the tied head) as int8, plus the one
    float scale each needs to be turned back into floats. All plain tensors: nothing here needs pickle."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}   # tied head/embedding keep sharing storage
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "head":
            w = module.weight.detach().float().cpu()
            scale = (w.abs().max() / _INT8_QMAX).clamp_min(1e-12)
            state[f"{name}.weight"] = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
            scales[f"{name}.weight"] = scale.reshape(1).clone()
    return state, scales


def build_int8_model(ckpt: dict):
    """Rebuild a CPU int8 model from a payload written by quantize_checkpoint (see the module docstring)."""
    if ckpt.get("quant_format") != INT8_FORMAT:
        raise ValueError(f"unknown int8 checkpoint format {ckpt.get('quant_format')!r} (expected {INT8_FORMAT!r}); "
                         f"re-run `compress` on the float checkpoint")
    cfg = ModelConfig.from_dict(ckpt["config"])
    state = dict(ckpt["int8_state"])
    for name, scale in ckpt["int8_scales"].items():
        state[name] = state[name].float() * scale.float()
    model = ZivaGPT(cfg)
    model.load_state_dict(state)
    model.eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.nn.quantized")
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()


def quantize_checkpoint(checkpoint_path: str, out_path: str) -> Tuple[int, int, "ZivaGPT"]:
    """Load `checkpoint_path` (preferring checkpoint_best.pt if present --
    same rule as serving, see checkpoint.resolve_serving_checkpoint),
    int8-quantize its Linear layers, and save the result to `out_path` in the safe format described in the
    module docstring. Returns (original_bytes, compressed_bytes, quantized_model), where the model is the
    one you get by loading `out_path` -- so any quality number measured on it is the real one."""
    src_path = resolve_serving_checkpoint(checkpoint_path)
    ckpt = load_checkpoint(src_path, map_location="cpu")
    if ckpt.get("quantized"):
        raise ValueError(f"{src_path} is already an int8 checkpoint; quantize the float one")
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = ZivaGPT(cfg).to("cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    state, scales = pack_int8_state(model)
    payload = {
        "quantized": True,
        "quant_format": INT8_FORMAT,
        "int8_state": state,
        "int8_scales": scales,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "step": ckpt["step"],
        "extra": ckpt.get("extra", {}),
    }
    write_payload(out_path, payload)
    quantized = build_int8_model(payload)

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
    cfg = ModelConfig.from_dict(ckpt["config"])
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
