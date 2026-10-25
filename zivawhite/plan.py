"""
plan.py - how big a model and how much data can you afford? (Advice only; nothing here limits what you train.)

    python -m zivawhite.plan --gpu a100 --n_gpus 1 --hours 48                  # best size for a time budget
    python -m zivawhite.plan --preset large --tokens 20b --gpu rtx4090 --n_gpus 2   # how long would this take?
    python -m zivawhite.plan --params 360m --gpu t4 --hours 24

The arithmetic is the standard one. Training costs about
    FLOPs = (6 * N + 12 * n_layer * n_embd * context) * D
for N parameters and D training tokens (the second term is attention over the context). A GPU delivers its peak
throughput times an efficiency ("MFU", model FLOPs utilisation): 25-45% is typical for well-tuned training of models
this size, and less on small models or old GPUs. Peak numbers below are approximate vendor figures for dense
fp16/bf16 tensor-core math; they are only a table in this file, so edit it or pass --tflops for your own card.

Compute-optimal sizing (Hoffmann et al. 2022, "Chinchilla"): for a fixed budget the best model sees roughly 20
tokens per parameter. Models meant to be used a lot are usually trained far beyond that ("over-trained", 100-200+
tokens per parameter for small ones) because a smaller, longer-trained model is cheaper to serve at the same
quality; the planner shows both.
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Optional

from .modelconfig import PRESETS, ModelConfig, estimate_train_memory_gb

# name -> (peak dense fp16/bf16 TFLOPs, memory GB). Approximate vendor specifications; edit freely.
GPUS: Dict[str, tuple] = {
    "t4": (65.0, 16), "l4": (121.0, 24), "a10": (125.0, 24), "rtx3090": (71.0, 24), "rtx4090": (165.0, 24),
    "a6000": (155.0, 48), "l40s": (362.0, 48), "a100-40": (312.0, 40), "a100": (312.0, 80),
    "h100-pcie": (756.0, 80), "h100": (989.0, 80),
}
CHINCHILLA_TOKENS_PER_PARAM = 20.0


def parse_count(text: str) -> float:
    """'125m' -> 1.25e8, '1.3b' -> 1.3e9, '20b' -> 2e10, '350000' -> 350000."""
    t = str(text).strip().lower().replace(",", "").replace("_", "")
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
    if t and t[-1] in mult:
        return float(t[:-1]) * mult[t[-1]]
    return float(t)


def human(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3g}{unit}"
    return f"{n:.3g}"


def flops_per_token(n_params: float, n_layer: int, n_embd: int, context: int) -> float:
    return 6.0 * n_params + 12.0 * n_layer * n_embd * context


def duration(seconds: float) -> str:
    if seconds < 3600 * 2:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def nearest_shape(n_params: float, context: int, vocab_size: int = 32000) -> ModelConfig:
    """A reasonable transformer shape of about n_params: llama-like proportions (width / depth between 20 and 90,
    as in real models: GPT-2 small 64, SmolLM2-360M 30, Llama-7B 128 at larger scale), 4:1 grouped-query attention."""
    best, best_key = None, None
    for n_layer in range(4, 65, 2):
        for head_dim in (64, 128):
            for n_head in range(4, 65, 2):
                n_embd = n_head * head_dim
                ratio = n_embd / n_layer
                if not 20 <= ratio <= 90:
                    continue
                kv = n_head // 4 if n_head % 4 == 0 else n_head // 2     # 4:1 grouped-query where possible, else 2:1
                cfg = ModelConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                  n_kv_head=kv, block_size=context)
                err = abs(cfg.total_params() - n_params) / n_params
                key = (round(err, 2), abs(math.log(ratio / 48)))       # closest size first, then the most typical shape
                if best_key is None or key < best_key:
                    best, best_key = cfg, key
    return best


def plan(*, params: Optional[float] = None, preset: Optional[str] = None, tokens: Optional[float] = None,
         hours: Optional[float] = None, gpu: str = "a100", n_gpus: int = 1, mfu: float = 0.35,
         context: Optional[int] = None, tflops: Optional[float] = None, vocab_size: int = 32000,
         batch_size: int = 8) -> Dict[str, Any]:
    """Return a dict describing the cost of training. Give a model (params or preset) and tokens to get the time,
    or a model and hours to get the tokens you can afford, or only hours to get a compute-optimal model size."""
    if gpu not in GPUS and tflops is None:
        raise ValueError(f"unknown gpu {gpu!r}; known: {', '.join(GPUS)} (or pass tflops=)")
    peak, mem = (tflops, None) if tflops is not None else GPUS[gpu]
    mem = GPUS[gpu][1] if gpu in GPUS else mem
    if not 0.01 <= mfu <= 0.9:
        raise ValueError("mfu must be between 0.01 and 0.9")
    eff = peak * 1e12 * mfu * n_gpus                                     # sustained FLOP/s of the whole setup
    out: Dict[str, Any] = {"gpu": gpu, "n_gpus": n_gpus, "peak_tflops": peak, "mfu": mfu,
                           "sustained_tflops": eff / 1e12, "gpu_memory_gb": mem}
    ctx = context or 2048

    if params is None and preset is None:
        if hours is None:
            raise ValueError("give a model (params= or preset=) and/or hours=")
        budget = eff * hours * 3600
        n_opt = math.sqrt(budget / (6 * CHINCHILLA_TOKENS_PER_PARAM))     # C = 6 N (20 N)  ->  N = sqrt(C / 120)
        for _ in range(2):   # ... then account for the attention FLOPs too: C = (6N + 12 L d s) * 20N
            shape = nearest_shape(n_opt, ctx, vocab_size)
            attn_overhead = 12 * shape.n_layer * shape.n_embd * ctx / (6 * shape.total_params())
            n_opt = math.sqrt(budget / (6 * CHINCHILLA_TOKENS_PER_PARAM * (1 + attn_overhead)))
        cfg = nearest_shape(n_opt, ctx, vocab_size)
        per_tok = flops_per_token(cfg.total_params(), cfg.n_layer, cfg.n_embd, ctx)
        d_opt = budget / per_tok
        out.update(mode="size_for_budget", budget_flops=budget, compute_optimal_params=n_opt, shape=cfg.to_dict(),
                   shape_params=cfg.total_params(), tokens=d_opt, tokens_per_param=d_opt / cfg.total_params(), hours=hours)
        # the same budget spent on a model 4x smaller trained 4x+ longer (cheaper to run afterwards)
        small = nearest_shape(n_opt / 4, ctx, vocab_size)
        d_small = budget / flops_per_token(small.total_params(), small.n_layer, small.n_embd, ctx)
        out["overtrained_alternative"] = {"params": small.total_params(), "tokens": d_small,
                                          "tokens_per_param": d_small / small.total_params(), "shape": small.to_dict()}
        cfg_for_mem = cfg
    else:
        cfg_for_mem = (ModelConfig.preset(preset, vocab_size=vocab_size, block_size=ctx) if preset
                       else nearest_shape(params, ctx, vocab_size))
        n = cfg_for_mem.total_params() if preset or params is None else params
        per_tok = flops_per_token(n, cfg_for_mem.n_layer, cfg_for_mem.n_embd, ctx)
        tps = eff / per_tok
        out.update(mode="time_for_tokens" if tokens else "tokens_for_budget", params=n, shape=cfg_for_mem.to_dict(),
                   flops_per_token=per_tok, tokens_per_second=tps, chinchilla_tokens=CHINCHILLA_TOKENS_PER_PARAM * n)
        if tokens:
            out.update(tokens=tokens, seconds=tokens * per_tok / eff, tokens_per_param=tokens / n,
                       total_flops=tokens * per_tok)
        elif hours:
            out.update(hours=hours, tokens=tps * hours * 3600, tokens_per_param=tps * hours * 3600 / n)
        else:
            raise ValueError("give tokens= or hours=")
    mem_est = estimate_train_memory_gb(cfg_for_mem, batch_size, False)
    mem_ckpt = estimate_train_memory_gb(cfg_for_mem, batch_size, True)
    out.update(memory_gb_per_gpu=mem_est, memory_gb_per_gpu_grad_ckpt=mem_ckpt,
               fits=(mem_est <= 0.9 * mem) if mem else None, fits_with_grad_ckpt=(mem_ckpt <= 0.9 * mem) if mem else None,
               context=ctx)
    return out


def format_plan(p: Dict[str, Any]) -> str:
    lines = [f"setup: {p['n_gpus']} x {p['gpu']} (peak {p['peak_tflops']:g} TFLOPs each, assuming {p['mfu']:.0%} efficiency "
             f"-> {p['sustained_tflops']:.1f} sustained TFLOPs)"]
    if p["mode"] == "size_for_budget":
        s = p["shape"]
        lines += [f"budget: {p['hours']:g} h = {p['budget_flops']:.2e} FLOPs",
                  f"compute-optimal model: ~{human(p['shape_params'])} params "
                  f"(--n_layer {s['n_layer']} --n_head {s['n_head']} --n_kv_head {s['n_kv_head']} --n_embd {s['n_embd']} "
                  f"--block_size {p['context']})",
                  f"  trains on ~{human(p['tokens'])} tokens ({p['tokens_per_param']:.0f} tokens/param, Chinchilla ~20)"]
        a = p["overtrained_alternative"]
        lines.append(f"over-trained alternative (cheaper to serve): ~{human(a['params'])} params on ~{human(a['tokens'])} "
                     f"tokens ({a['tokens_per_param']:.0f} tokens/param)")
    else:
        lines.append(f"model: ~{human(p['params'])} params, context {p['context']}, {human(p['flops_per_token'])} FLOPs/token "
                     f"-> {human(p['tokens_per_second'])} tokens/s")
        if p["mode"] == "time_for_tokens":
            lines.append(f"{human(p['tokens'])} tokens ({p['tokens_per_param']:.0f}/param) takes ~{duration(p['seconds'])} "
                         f"({p['total_flops']:.2e} FLOPs)")
        else:
            lines.append(f"{p['hours']:g} h buys ~{human(p['tokens'])} tokens ({p['tokens_per_param']:.0f} tokens/param)")
        lines.append(f"for reference, compute-optimal data for this size is ~{human(p['chinchilla_tokens'])} tokens")
    fit = ("fits" if p["fits"] else "probably does NOT fit") if p["fits"] is not None else "unknown"
    fit2 = ("fits" if p["fits_with_grad_ckpt"] else "probably does NOT fit") if p["fits_with_grad_ckpt"] is not None else "unknown"
    lines.append(f"memory per GPU at batch 8: ~{p['memory_gb_per_gpu']:.1f} GB ({fit} in {p['gpu_memory_gb']} GB); "
                 f"with --grad_ckpt ~{p['memory_gb_per_gpu_grad_ckpt']:.1f} GB ({fit2}). Rough (+-40%).")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Training compute planner (advice only)")
    ap.add_argument("--params", default=None, help="model size, e.g. 125m, 1.3b")
    ap.add_argument("--preset", default=None, choices=[None, *PRESETS])
    ap.add_argument("--tokens", default=None, help="training tokens, e.g. 10b")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--gpu", default="a100", help=f"one of: {', '.join(GPUS)}")
    ap.add_argument("--n_gpus", type=int, default=1)
    ap.add_argument("--mfu", type=float, default=0.35)
    ap.add_argument("--tflops", type=float, default=None, help="peak dense fp16/bf16 TFLOPs of a GPU not in the table")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args(argv)
    print(format_plan(plan(params=parse_count(a.params) if a.params else None, preset=a.preset,
                           tokens=parse_count(a.tokens) if a.tokens else None, hours=a.hours, gpu=a.gpu,
                           n_gpus=a.n_gpus, mfu=a.mfu, context=a.context, tflops=a.tflops, batch_size=a.batch_size)))


if __name__ == "__main__":
    main()
