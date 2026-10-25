"""
train.py - one training script, any model size, any hardware.

MODEL SIZE IS YOUR CHOICE, never inferred from the machine. Pick a preset and/or set the numbers:

    python -m zivawhite.train --tier medium --max_steps 20000                  # ~300M, a named size
    python -m zivawhite.train --tier small --n_layer 12 --n_embd 768 --n_head 12 --n_kv_head 4 --block_size 2048
    python -m zivawhite.train --tier xl --grad_ckpt --batch_size 4 --global_batch_tokens 524288

Presets: tiny / small / medium / large / xl (see modelconfig.py for exact sizes). With no --tier the
default starting point is "small". Hardware only decides device and precision (CUDA / MPS / CPU, bf16 vs
fp16). If the memory estimate says a run probably will not fit, a warning is printed; nothing is capped.

Overfitting guards (the reason a chat fine-tune used to memorise its data):
    --max_epochs 3      never take more than ~3 passes over the training tokens
    --patience 5        stop after 5 evaluations without a better validation loss
    --dropout 0.1       regularise while fine-tuning (weights are unaffected; it is off at inference)
checkpoint_best.pt is only rewritten when validation loss improves, and is what `ask`/`serve` load.

Single process:
    python -m zivawhite.train --data_dir data/processed --out_dir runs/ziva

Multi-GPU, one machine, 4 GPUs (each rank trains on different random
batches; gradients are all-reduced/averaged across ranks every step by
DistributedDataParallel automatically -- see the DDP wiring below):
    torchrun --standalone --nproc_per_node=4 -m zivawhite.train --tier large

Multi-node (2 machines x 4 GPUs each -- run this on each machine, only
--node_rank differs):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=<node0-ip> --master_port=29500 -m zivawhite.train --tier large
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
             --master_addr=<node0-ip> --master_port=29500 -m zivawhite.train --tier large

No `--ddp` flag to remember: torchrun sets RANK/LOCAL_RANK/WORLD_SIZE in
the environment for every process it launches, device.py:resolve_runtime
reads those directly, and this script checks WORLD_SIZE>1 to decide
whether it's a distributed run. Plain `python -m zivawhite.train` never
sets WORLD_SIZE, so it's always exactly 1 there -- single-process
behavior is completely unchanged.

--batch_size is PER PROCESS (nanoGPT/PyTorch convention): the effective
global batch actually trained on each step is
batch_size * grad_accum * world_size, printed at startup so it's never
ambiguous what a given run actually did.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data.dataset import get_batch, token_count
from .device import resolve_runtime
from .model import ZivaGPT, ModelConfig
from .modelconfig import PRESETS, TRAIN_DEFAULTS, estimate_train_memory_gb
from .tokenizer import (TokenizerMismatch, TokenizerNotFound, describe_tokenizer, pick_training_tokenizer,
                        save_run_tokenizer)

MAX_BAD_STEPS = 3          # consecutive non-finite steps before a run is stopped (see the training loop)
DEFAULT_PRESET = "small"   # the starting point when you pass neither --tier nor size flags (not hardware-derived)


def lr_at(step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float,
          schedule: str = "cosine", horizon: Optional[int] = None) -> float:
    """Learning rate at an absolute `step`.

    "cosine": warm up, then decay to min_lr over `horizon` steps (default: max_steps).
    "constant": warm up, then hold max_lr.

    `horizon` exists because chunked training (keep_training.py, autolearn.py) calls
    train.py repeatedly with a growing --max_steps. If the schedule decayed over each
    chunk's own max_steps, LR would jump back up at every chunk boundary (a sawtooth).
    Pass the real total as horizon, or use "constant" when there is no fixed end."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return max_lr
    horizon = horizon or max_steps
    if step > horizon:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, horizon - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int = 50,
             amp: Optional[tuple] = None) -> Optional[float]:
    """Returns None (never crashes) if the val shard is too small for even
    one block-sized batch -- e.g. a freshly-bootstrapped tiny corpus where
    build_shards() had to shrink the split. build_shards() itself now keeps
    val >= block_size by default (see data/ingest.py), so this is a safety
    net for hand-rolled shards, not the primary fix. Call with the raw
    (unwrapped) model, not a DDP wrapper -- eval reads no gradients, so
    there's nothing for DDP's backward hooks to do here, and it's rank 0's
    call to make alone (see main()'s "only rank 0 evaluates" note)."""
    model.eval()
    losses = []
    import contextlib

    # amp=(device_type, dtype): evaluate under the same autocast as training (fp32 eval on a fp16/bf16 GPU
    # is several times slower and measures a slightly different function than the one being trained)
    amp_ctx = (lambda: torch.autocast(device_type=amp[0], dtype=amp[1])) if amp else contextlib.nullcontext
    for _ in range(eval_iters):
        try:
            x, y = get_batch(data_dir, "val", block_size, batch_size, device)
        except ValueError as e:
            if losses:
                break  # got at least one usable batch before running out; use what we have
            print(f"  [eval] skipped: {e}")
            model.train()
            return None
        with amp_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else None


def main() -> None:
    # --config is read BEFORE the real parser so its values can become the defaults
    # (explicit command-line flags still win). See configs/*.yaml.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    config_path = pre.parse_known_args()[0].config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML of default flag values, e.g. configs/gpu_small.yaml")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir", default="runs/ziva")
    p.add_argument("--tokenizer_dir", default=None,
                    help="folder with vocab.json + merges.txt. Default: the tokenizer saved with the run you resume / warm-start from "
                         "(<run>/tokenizer/), or for a brand-new model the bundled general-English one. The tokenizer used is "
                         "always saved into <out_dir>/tokenizer/ so every later command finds it by itself")
    p.add_argument("--tier", default=None, choices=[None, *PRESETS],
                    help="a named model size to start from (tiny/small/medium/large/xl). An explicit choice: "
                         "never inferred from your hardware. Ignored when resuming or warm-starting (the "
                         "checkpoint's own shape wins). Default: small")
    g = p.add_argument_group("model shape (new runs only; each overrides the preset)")
    g.add_argument("--n_layer", type=int, default=None)
    g.add_argument("--n_head", type=int, default=None)
    g.add_argument("--n_kv_head", type=int, default=None, help="key/value heads for grouped-query attention (must divide n_head)")
    g.add_argument("--n_embd", type=int, default=None)
    g.add_argument("--intermediate_size", type=int, default=None, help="SwiGLU width (default 8/3 * n_embd)")
    g.add_argument("--no_gqa", action="store_true", help="use plain multi-head attention for the preset (n_kv_head = n_head)")
    g.add_argument("--rope_theta", type=float, default=None)
    h = p.add_argument_group("may also change when resuming / fine-tuning")
    h.add_argument("--block_size", type=int, default=None, help="context length. Can be raised when fine-tuning, with --rope_scaling")
    h.add_argument("--rope_scaling", default=None, choices=[None, "linear", "ntk"],
                    help="stretch rotary embeddings to a longer --block_size than the model was trained at (fine-tune on long text afterwards)")
    h.add_argument("--rope_scaling_factor", type=float, default=None)
    h.add_argument("--dropout", type=float, default=None, help="dropout during training (e.g. 0.1 for small fine-tuning sets)")
    p.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing: ~30%% slower, much less memory (lets a bigger model or batch fit)")
    p.add_argument("--global_batch_tokens", type=int, default=None,
                    help="target tokens per optimizer step across all processes (e.g. 524288); sets --grad_accum for you unless you pass it")
    p.add_argument("--max_epochs", type=float, default=None,
                    help="cap the run at about this many passes over the training tokens (overfitting guard; fractional ok)")
    p.add_argument("--patience", type=int, default=0,
                    help="stop early after this many evaluations in a row without a better validation loss (0 = never)")
    p.add_argument("--eval_iters", type=int, default=50, help="batches averaged per validation measurement")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--extra_steps", type=int, default=None,
                    help="train this many MORE steps from wherever the checkpoint (or warm start) currently is, "
                         "instead of to an absolute --max_steps. Used by autolearn so it never has to load the "
                         "checkpoint just to read its step counter. Overrides --max_steps.")
    p.add_argument("--batch_size", type=int, default=None, help="PER PROCESS. Effective global batch = batch_size * grad_accum * world_size")
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--ckpt_interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337,
                    help="each rank seeds with seed+rank so distributed ranks sample different random "
                         "batches (identical batches across ranks would make DDP's gradient all-reduce "
                         "average N copies of the same gradient -- correct but pointless, no data-parallel "
                         "benefit at all). Single-process runs are unaffected by this beyond being seeded.")
    p.add_argument("--compile", action="store_true",
                    help="wrap the model in torch.compile() for a real speedup on CUDA (PyTorch 2.x kernel "
                         "fusion) -- opt-in because compilation adds real startup latency and, on some "
                         "CPU/driver combinations, no gain or an outright failure; falls back to the "
                         "uncompiled model automatically if compilation itself raises.")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"],
                    help="LR schedule after warmup. Use 'constant' for open-ended/chunked runs")
    p.add_argument("--lr_horizon", type=int, default=None,
                    help="cosine decays to min_lr at this absolute step (default: --max_steps). Set it to the "
                         "run's REAL total when training in chunks, so every chunk follows one curve")
    p.add_argument("--init_from", default=None,
                    help="warm-start: load only the WEIGHTS of this checkpoint and begin at step 0 with a fresh "
                         "optimizer, writing to --out_dir (the source checkpoint is never modified). This is "
                         "fine-tuning. Unlike --resume it doesn't continue the source run. The model shape comes "
                         "from the checkpoint, not from this machine's hardware tier")
    p.add_argument("--reset_best", action="store_true",
                    help="forget the best-val-loss gate. Needed when the data changed: val loss on a new dataset "
                         "isn't comparable to the old one, and without this a fine-tune could never replace "
                         "checkpoint_best.pt (so serving kept loading the old model). Implied by --init_from")
    p.add_argument("--resume", action="store_true",
                    help="continue from --out_dir/checkpoint.pt if present. --max_steps is an absolute target "
                         "step count, not an additional-steps count -- it must exceed the checkpoint's saved "
                         "step or the training loop (range(start_step, max_steps)) simply won't run.")
    p.add_argument("--ewc", default=None, help="path to a consolidation.pt from `app.py consolidate` -- adds an EWC penalty so this run doesn't overwrite what an earlier dataset taught the model")
    p.add_argument("--ewc_lambda", type=float, default=1000.0)
    p.add_argument("--autotune", action="store_true", help="empirically pick batch_size for this machine before training instead of using the tier default")
    p.add_argument("--behavior_tuned", action="store_true",
                    help="stamp the checkpoint as instruction/conversation-tuned (set automatically by "
                         "autolearn's conversation fine-tune; don't pass this for plain text training). The router "
                         "reads this flag to know whether the local model can sensibly be asked a direct "
                         "question/used as a RAG writer, or is still a base model that only continues text in "
                         "its training corpus's style -- see router.py's _run_local/_run_web.")
    if config_path:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in p._actions}
        unknown = sorted(k for k in file_cfg if k not in valid and k != "notes")
        if unknown:
            print(f"[zivawhite] warning: ignoring unknown keys in {config_path}: {unknown}")
        p.set_defaults(**{k: v for k, v in file_cfg.items() if k in valid and k != "config"})
    args = p.parse_args()

    rt = resolve_runtime(force_tier=args.tier)
    is_ddp = rt.world_size > 1
    is_main = rt.is_main_process  # only this process prints/evals/checkpoints when is_ddp
    # device_type (bare "cuda", never "cuda:0") is what torch.autocast's
    # device_type= actually wants -- rt.device (possibly "cuda:0") is what
    # every .to(device)/tensor-placement call wants. Conflating the two was
    # the exact-match bug just fixed in model.py/dataset.py; keeping this
    # split explicit here avoids reintroducing it.
    device_type = rt.device.split(":")[0]

    if is_ddp:
        import torch.distributed as dist

        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    torch.manual_seed(args.seed + rt.rank)

    if is_main:
        print(f"[zivawhite] {rt.summary()}")
        if is_ddp:
            print(f"[zivawhite] DDP: {rt.world_size} process(es), backend={'nccl' if device_type == 'cuda' else 'gloo'}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    # Resuming or warm-starting must rebuild the model the checkpoint was trained with.
    # The hardware-tier preset is only right for a brand-new run: it used to be applied
    # to every run, so resuming a Colab "small" checkpoint on a laptop (auto-tier "tiny")
    # failed with a shape-mismatch error.
    resume_ckpt = load_checkpoint(ckpt_path, map_location="cpu") if args.resume and os.path.exists(ckpt_path) else None
    init_ckpt = load_checkpoint(args.init_from, map_location="cpu") if args.init_from and resume_ckpt is None else None
    seed_ckpt = resume_ckpt or init_ckpt
    is_resume = resume_ckpt is not None

    # Which vocabulary? An explicit --tokenizer_dir wins. Otherwise a run being resumed / warm-started keeps the
    # tokenizer it was trained with (saved in its folder), and only a brand-new model starts from the bundled
    # general-English one. Mixing vocabularies silently corrupts a model, so every mismatch stops here.
    try:
        tok, tok_how = pick_training_tokenizer(args.tokenizer_dir, ckpt_path if is_resume else args.init_from, seed_ckpt)
    except (TokenizerNotFound, TokenizerMismatch) as e:
        raise SystemExit(f"[zivawhite] {e}")
    arch_flags = {k: getattr(args, k) for k in ("n_layer", "n_head", "n_kv_head", "n_embd", "intermediate_size", "rope_theta")}
    if seed_ckpt is not None:
        asked = [k for k, v in arch_flags.items() if v is not None] + (["no_gqa"] if args.no_gqa else [])
        if asked:
            raise SystemExit(f"cannot change {asked} of an existing model (resume / --init_from keep the checkpoint's own "
                             f"shape). Only --block_size, --rope_scaling[_factor] and --dropout may change here. "
                             f"To train a different shape, start a new run without --resume/--init_from.")
        cfg = ModelConfig.from_dict(seed_ckpt["config"])
        if cfg.vocab_size != tok.vocab_size:
            raise SystemExit(f"checkpoint vocab_size={cfg.vocab_size} but the tokenizer has {tok.vocab_size:,} tokens; "
                             f"pass the --tokenizer_dir this checkpoint was trained with")
        base_preset = args.tier if args.tier in TRAIN_DEFAULTS else DEFAULT_PRESET
    else:
        base_preset = args.tier or DEFAULT_PRESET
        cfg = ModelConfig.preset(base_preset, vocab_size=tok.vocab_size, gqa=not args.no_gqa, **arch_flags)
    # settings that are safe to change on an existing model (they do not alter any weight's shape)
    for key in ("block_size", "rope_scaling", "rope_scaling_factor", "dropout"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    cfg.validate()

    # The model carries its own vocabulary: copy the tokenizer into <out_dir>/tokenizer/ and record its fingerprint
    # in every checkpoint, so ask / serve / generate / eval / sft / export find it without being told.
    tok_meta = describe_tokenizer(tok)
    if is_main:
        try:
            tok_meta = save_run_tokenizer(tok, args.out_dir, replace=not is_resume)
        except TokenizerMismatch as e:
            raise SystemExit(f"[zivawhite] {e}")
        print(f"[zivawhite] tokenizer: {tok_how}, {tok.vocab_size:,} tokens -> saved with this run in "
              f"{os.path.join(args.out_dir, 'tokenizer')}")

    # Training defaults come from the CHOSEN preset (custom shapes use the "small" row), not from hardware.
    row = TRAIN_DEFAULTS.get(base_preset, TRAIN_DEFAULTS[DEFAULT_PRESET])
    batch_size = args.batch_size or row["batch_size"]
    if args.grad_accum:
        grad_accum = args.grad_accum
    elif args.global_batch_tokens:
        grad_accum = max(1, round(args.global_batch_tokens / (batch_size * cfg.block_size * rt.world_size)))
    else:
        grad_accum = row["grad_accum"]
    max_steps = args.max_steps or row["max_steps"]

    if is_main:
        est = estimate_train_memory_gb(cfg, batch_size, args.grad_ckpt)
        have = rt.vram_gb if device_type == "cuda" else rt.ram_gb
        if have and est > 0.9 * have:
            print(f"[zivawhite] WARNING: rough memory estimate {est:.1f} GB vs {have:.1f} GB available on {rt.device_name}. "
                  f"This may run out of memory. Try --grad_ckpt, a smaller --batch_size (raise --grad_accum or "
                  f"--global_batch_tokens to keep the effective batch), or a smaller --block_size. Continuing anyway: "
                  f"nothing here caps your model size.")

    if args.autotune:
        from .device import benchmark_and_tune

        if is_main:
            print("[zivawhite] autotuning batch size on this machine...")
        bench = benchmark_and_tune(rt, lambda tier: ModelConfig.preset(tier, vocab_size=tok.vocab_size), tier=base_preset)
        batch_size = max(bench, key=bench.get)
        if is_main:
            print(f"[zivawhite] autotune picked batch_size={batch_size} "
                  f"({bench[batch_size]:,.0f} tok/s measured on this machine)")

    consolidation = None
    if args.ewc:
        from .continual import load_consolidation

        consolidation = load_consolidation(args.ewc, map_location=rt.device)
        if is_main:
            print(f"[zivawhite] EWC enabled from {args.ewc} (lambda={args.ewc_lambda})")

    # raw_model is always the plain ZivaGPT -- the thing with .configure_optimizer(),
    # .num_params(), .cfg, and a state_dict with no "module." prefix. `model` is what
    # actually gets called in the training loop: raw_model itself outside DDP, or a
    # DistributedDataParallel wrapper around it under torchrun. DDP does NOT forward
    # arbitrary attribute access to the wrapped module (only forward() gets special
    # handling), so every non-forward call below deliberately goes through raw_model,
    # never through `model` -- and every checkpoint save/load uses raw_model's
    # state_dict so a DDP checkpoint loads with zero changes anywhere else in the
    # project (router.py, generate, etc. never need to know a checkpoint was trained
    # under torchrun).
    raw_model = ZivaGPT(cfg).to(rt.device)
    raw_model.gradient_checkpointing = bool(args.grad_ckpt)
    optimizer = raw_model.configure_optimizer(args.weight_decay, args.max_lr, (0.9, 0.95), device_type)
    start_step = 0
    best_val_loss = float("inf")
    behavior_tuned = args.behavior_tuned
    best_ckpt_path = os.path.join(args.out_dir, "checkpoint_best.pt")

    # A fresh or warm-started run also resets the gate: a checkpoint_best.pt left in out_dir by an
    # earlier run would otherwise survive a run that never reaches an eval, and win at serve time.
    reset_best = args.reset_best or resume_ckpt is None
    if resume_ckpt is not None:
        raw_model.load_state_dict(resume_ckpt["model_state"])
        if resume_ckpt.get("optimizer_state"):
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_step = resume_ckpt["step"]
        if not reset_best:
            best_val_loss = resume_ckpt.get("extra", {}).get("best_val_loss", float("inf"))
        behavior_tuned = behavior_tuned or resume_ckpt.get("extra", {}).get("behavior_tuned", False)
        if is_main:
            print(f"[zivawhite] resumed from step {start_step} (best_val_loss so far: "
                  f"{best_val_loss if best_val_loss != float('inf') else 'none yet'})")
    elif init_ckpt is not None:
        raw_model.load_state_dict(init_ckpt["model_state"])
        behavior_tuned = behavior_tuned or init_ckpt.get("extra", {}).get("behavior_tuned", False)
        if is_main:
            print(f"[zivawhite] warm-started weights from {args.init_from} (source checkpoint left untouched); "
                  f"training from step 0 -> {args.out_dir}")
    del resume_ckpt, init_ckpt, seed_ckpt  # weights are copied into raw_model; free the CPU copies
    if args.extra_steps:
        max_steps = start_step + args.extra_steps
    best_written = False

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # device_ids=[local GPU index] is required for the NCCL backend and
        # meaningless (and disallowed) for CPU/gloo, hence the branch.
        # A mixture-of-experts layer leaves experts that received no tokens this step without a gradient; DDP
        # then waits for them forever unless it is told unused parameters are expected.
        unused = bool(cfg.use_moe)
        model = (DDP(raw_model, device_ids=[rt.local_rank], find_unused_parameters=unused) if device_type == "cuda"
                 else DDP(raw_model, find_unused_parameters=unused))
    else:
        model = raw_model

    if args.compile:
        try:
            model = torch.compile(model)
            if is_main:
                print("[zivawhite] torch.compile() enabled (first step will be slower -- that's compilation, "
                      "not a regression; steps after it are the ones that are actually faster)")
        except Exception as e:  # pragma: no cover - depends on host toolchain/driver
            if is_main:
                print(f"[zivawhite] torch.compile() failed on this machine ({e}); continuing uncompiled")

    # CPU gets autocast too now, but only when device.py's startup
    # benchmark actually measured bfloat16 as faster on this machine (see
    # _benchmark_cpu_dtype) -- never assumed. bf16 has fp32's exponent
    # range, so unlike fp16 it needs no loss-scaling; the GradScaler below
    # stays fp16/cuda-only for exactly that reason.
    use_amp = device_type == "cuda" or (device_type == "cpu" and rt.dtype == "bfloat16")
    amp_dtype = torch.bfloat16 if rt.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    n_train = token_count(args.data_dir, "train")
    tokens_per_step = batch_size * grad_accum * rt.world_size * cfg.block_size
    if args.max_epochs and n_train > 0:
        # Overfitting guard #1: a fixed step count means a *small* dataset gets read over and over
        # (100 steps x ~33K tokens/step on the `small` tier is ~3.3M tokens: ~30 passes over a 100K-token chat set).
        epoch_cap = max(1, math.ceil(args.max_epochs * n_train / tokens_per_step))
        if start_step + epoch_cap < max_steps:
            if is_main:
                print(f"[zivawhite] --max_epochs {args.max_epochs:g}: {n_train:,} train tokens at {tokens_per_step:,} "
                      f"tokens/step is {epoch_cap} step(s) -> capping this run at {epoch_cap} step(s) instead of "
                      f"{max_steps - start_step}")
            max_steps = start_step + epoch_cap
    if is_main:
        effective_batch = batch_size * grad_accum * rt.world_size
        print(f"[zivawhite] model params (non-embed): {raw_model.num_params():,} | train tokens: {n_train:,}")
        print(f"[zivawhite] batch_size={batch_size}/process grad_accum={grad_accum} world_size={rt.world_size} "
              f"-> effective batch={effective_batch} | block_size={cfg.block_size} max_steps={max_steps}")
        print(f"[zivawhite] this run covers ~{(max_steps - start_step) * tokens_per_step / max(n_train, 1):.2f} epoch(s) of the "
              f"training tokens; tokens/step={tokens_per_step:,}")

    model.train()
    t0 = time.time()
    t_last, step_last = t0, start_step
    evals_without_gain = 0
    bad_steps = 0
    final_step = max_steps
    eval_amp = (device_type if device_type != "mps" else "cpu", amp_dtype) if use_amp else None
    for step in range(start_step, max_steps):
        lr = lr_at(step, warmup_steps=args.warmup_steps, max_steps=max_steps, max_lr=args.max_lr, min_lr=args.min_lr,
                   schedule=args.schedule, horizon=args.lr_horizon)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=rt.device)   # summed on the device: one host sync per step, not one per micro-step
        for micro_step in range(grad_accum):
            x, y = get_batch(args.data_dir, "train", cfg.block_size, batch_size, rt.device)
            # Only all-reduce gradients on the last accumulation micro-step:
            # DDP's default behavior is to sync every backward() call, which
            # would mean grad_accum-1 wasted all-reduces per optimizer step.
            # no_sync() is DDP's own documented way to skip that until the
            # gradients are actually about to be used.
            sync_context = (
                model.no_sync() if is_ddp and micro_step < grad_accum - 1 else torch.enable_grad()
            )
            with sync_context:
                with torch.autocast(device_type=device_type if device_type != "mps" else "cpu", dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                    if consolidation is not None:
                        from .continual import ewc_penalty

                        loss = loss + ewc_penalty(raw_model, consolidation, args.ewc_lambda) / grad_accum
                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            loss_sum += loss.detach().float()

        total_loss = float(loss_sum)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0))

        # A NaN/inf loss means the weights are already damaged or about to be. Never step on it, and never let it
        # reach checkpoint.pt: after a few in a row, stop and leave the last good checkpoint untouched.
        # (With fp16 loss scaling a non-finite GRADIENT is routine -- the scaler skips that step and lowers its scale --
        # so only a non-finite loss counts there.)
        bad = (not math.isfinite(total_loss)) or (not scaler.is_enabled() and not math.isfinite(grad_norm))
        if is_ddp:                                  # the loss is per-rank: every rank must take the same branch
            import torch.distributed as dist

            flag = torch.tensor([1.0 if bad else 0.0], device=rt.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            bad = bool(flag.item())
        if bad:
            bad_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.update()                     # keep the loss-scale bookkeeping consistent after a skipped step
            if is_main:
                print(f"step {step:6d} | loss {total_loss} grad_norm {grad_norm} is not finite: update skipped "
                      f"({bad_steps}/{MAX_BAD_STEPS} in a row)")
            if bad_steps >= MAX_BAD_STEPS:
                if is_main:
                    print(f"[zivawhite] ERROR: training diverged (non-finite loss {MAX_BAD_STEPS} steps in a row). Stopping "
                          f"WITHOUT saving, so {ckpt_path} still holds the last good weights. Lower --max_lr, raise "
                          f"--warmup_steps, or check the data.")
                if is_ddp:
                    import torch.distributed as dist

                    dist.destroy_process_group()
                raise SystemExit(2)
            continue
        bad_steps = 0
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if is_main and step % 20 == 0:
            now = time.time()
            tps = (step - step_last) * tokens_per_step / max(now - t_last, 1e-9) if step > step_last else 0.0
            t_last, step_last = now, step
            print(f"step {step:6d} | loss {total_loss:.4f} | grad_norm {grad_norm:.2f} | lr {lr:.2e} | {now - t0:.1f}s elapsed"
                  + (f" | {tps:,.0f} tok/s" if tps else ""))

        # Only rank 0 evaluates/checkpoints. Every rank's model weights are
        # identical after each step (DDP guarantees this via the gradient
        # all-reduce every rank just did), so rank 0's copy is exactly as
        # good a thing to evaluate/save as any other rank's -- there's
        # nothing to gain by having every rank redundantly eval and
        # nothing to lose by having only one rank do it.
        # Evaluate every eval_interval steps AND at the last step. Without the last-step eval a run shorter
        # than eval_interval (every autolearn chat fine-tune) never measured validation loss at all, so
        # nothing could tell a good fine-tune from a memorised one. Identical on every rank (DDP-safe).
        do_eval = (step > 0 and step % args.eval_interval == 0) or step == max_steps - 1
        stop_now = False
        if is_main and do_eval:
            val_loss = evaluate(raw_model, args.data_dir, cfg.block_size, batch_size, rt.device,
                                eval_iters=args.eval_iters, amp=eval_amp)
            if val_loss is not None:
                print(f"  [eval] step {step} val_loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.1f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Inference-only snapshot (no optimizer state -> smaller,
                    # faster to load). This is the "only ever gets better"
                    # guarantee: ask/serve load this file in preference to
                    # the rolling checkpoint.pt (see checkpoint.py:
                    # resolve_serving_checkpoint), and it's only overwritten
                    # right here, when eval loss actually improved.
                    save_checkpoint(best_ckpt_path, model=raw_model, optimizer=None, step=step + 1, cfg=cfg,
                                     extra={"val_loss": val_loss, "behavior_tuned": behavior_tuned, "tokenizer": tok_meta})
                    best_written = True
                    evals_without_gain = 0
                    print(f"  [best] new best val_loss {val_loss:.4f} -> {best_ckpt_path}")
                else:
                    evals_without_gain += 1
                    # Overfitting guard #2: validation loss that stops improving while training loss keeps
                    # falling is memorisation; checkpoint_best.pt already holds the best point, so just stop.
                    if args.patience and evals_without_gain >= args.patience:
                        stop_now = True
                        print(f"  [early stop] val_loss has not improved for {evals_without_gain} evaluation(s) "
                              f"(best {best_val_loss:.4f}); stopping. The best checkpoint is already saved.")
        if is_ddp and args.patience and do_eval:   # every rank must leave the loop on the same step
            flag = torch.tensor([1 if stop_now else 0], device=rt.device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())

        if is_main and step > 0 and step % args.ckpt_interval == 0:
            save_checkpoint(ckpt_path, model=raw_model, optimizer=optimizer, step=step + 1, cfg=cfg,
                             extra={"best_val_loss": best_val_loss, "behavior_tuned": behavior_tuned, "tokenizer": tok_meta})
            print(f"  [ckpt] saved at step {step} -> {ckpt_path}")

        if stop_now:
            final_step = step + 1
            break

    if is_main:
        save_checkpoint(ckpt_path, model=raw_model, optimizer=optimizer, step=final_step, cfg=cfg,
                         extra={"best_val_loss": best_val_loss, "behavior_tuned": behavior_tuned, "tokenizer": tok_meta})
        print(f"[zivawhite] training complete -> {ckpt_path}")
        if reset_best and not best_written:
            # The data changed, so any checkpoint_best.pt in this folder is from a different
            # dataset (e.g. the base model). If no eval beat the reset gate (a short run, or a
            # val shard too small to evaluate) it would stay behind and be served in preference
            # to what was just trained. Make the final state the new best.
            save_checkpoint(best_ckpt_path, model=raw_model, optimizer=None, step=final_step, cfg=cfg,
                             extra={"val_loss": None, "behavior_tuned": behavior_tuned, "tokenizer": tok_meta})
            print(f"[zivawhite] no eval improved on the reset gate; final weights saved as best -> {best_ckpt_path}")
        if os.path.exists(best_ckpt_path):
            shown = f"val_loss {best_val_loss:.4f}" if best_val_loss != float("inf") else "no val_loss measured"
            print(f"[zivawhite] best checkpoint ({shown}) -> {best_ckpt_path}")

    if is_ddp:
        import torch.distributed as dist

        dist.barrier()  # let every rank reach the end before anyone tears down the process group
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
