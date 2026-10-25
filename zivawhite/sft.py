"""
sft.py - supervised fine-tuning that learns ONLY from what the assistant should say.

    python -m zivawhite.sft --data data/chat/*.jsonl --init_from runs/ziva/checkpoint.pt --out_dir runs/ziva_chat
    python -m zivawhite.app sft ...                                  # same thing

Compared with fine-tuning through the generic language-model trainer (train.py on a chat corpus):
  * Loss is computed on the assistant's reply and its end-of-turn token only. The user's words, system
    prompts and tool results are context, not targets, so the model is not rewarded for imitating users.
  * Every assistant turn of a multi-turn conversation becomes its own example (earlier turns as context),
    each ending in <eos>, so the model learns WHEN TO STOP instead of running into the next "<|user|>".
  * Runs in epochs over the data, with a held-out validation split, early stopping, and a best-checkpoint
    that is only overwritten when validation loss improves. A small dataset cannot be memorised for
    hundreds of passes the way a fixed step count allowed.
  * `system` and `tool` roles are kept (the dataset-folder path folds system prompts into the first user
    turn and drops tool messages), which is what system-prompt adherence and tool use are trained from.

Input: JSONL (one conversation per line), a JSON list, or folders of them. Each conversation is
    {"messages": [{"role": "system|user|assistant|tool", "content": "..."}, ...]}
ShareGPT-style {"conversations": [{"from": "human|gpt|system|tool", "value": "..."}]} also works.
Output: a normal ZivaWhite checkpoint (checkpoint.pt + checkpoint_best.pt), stamped behavior_tuned.

The data handling below is plain Python (no torch), so it is unit-tested without a GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .chat import ASSISTANT_TAG, render_conversation

IGNORE = -100   # F.cross_entropy's ignore_index: positions with this label contribute no loss

_ROLE_ALIASES = {
    "user": "user", "human": "user", "customer": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "ai": "assistant", "model": "assistant",
    "system": "system",
    "tool": "tool", "function": "tool", "observation": "tool", "function_response": "tool", "ipython": "tool",
}
Example = Tuple[List[int], List[int]]   # (ids, labels), aligned: labels[i] is ids[i] if supervised else IGNORE


# --------------------------------------------------------------------------------- conversations
def normalize_conversation(raw: Any) -> Optional[List[Dict[str, str]]]:
    """A list of {"role","content"} (or {"from","value"}) -> clean messages, or None if unusable.
    Keeps system and tool roles; merges consecutive same-role user/assistant turns; needs >= 1 assistant turn
    that follows something to answer."""
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = _ROLE_ALIASES.get(str(m.get("role", m.get("from", ""))).strip().lower())
        content = m.get("content", m.get("value"))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()
        if out and out[-1]["role"] == role and role in ("user", "assistant"):
            out[-1]["content"] += "\n" + content
        else:
            out.append({"role": role, "content": content})
    # a conversation starts with an optional system prompt, then a user turn
    while out and out[0]["role"] in ("assistant", "tool"):
        out.pop(0)
    if not any(m["role"] == "assistant" for m in out):
        return None
    if not any(m["role"] == "user" for m in out):
        return None
    return out


def _iter_records(path: str) -> Iterable[Any]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        else:
            data = json.load(f)
            yield from (data if isinstance(data, list) else [data])


def load_conversations(sources: Sequence[str]) -> List[List[Dict[str, str]]]:
    """Read every conversation from files, globs or folders (.jsonl / .json)."""
    paths: List[str] = []
    for s in sources:
        if os.path.isdir(s):
            paths += sorted(glob.glob(os.path.join(s, "**", "*.json*"), recursive=True))
        else:
            paths += sorted(glob.glob(s)) or [s]
    convs = []
    for path in paths:
        if not path.endswith((".json", ".jsonl")) or not os.path.isfile(path):
            continue
        for rec in _iter_records(path):
            if isinstance(rec, dict):
                msgs = normalize_conversation(rec.get("messages", rec.get("conversations")))
                if msgs:
                    convs.append(msgs)
    return convs


def split_conversations(convs: List[Any], val_fraction: float = 0.05, seed: int = 1337,
                        min_for_val: int = 20) -> Tuple[List[Any], List[Any]]:
    """Deterministic train/validation split BY CONVERSATION (so no conversation straddles both). Sets too small
    to spare a validation example return ([all], []) and train for a fixed number of epochs without early stopping."""
    if len(convs) < min_for_val:
        return list(convs), []
    order = list(range(len(convs)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(convs) * val_fraction))
    val_idx = set(order[:n_val])
    return [c for i, c in enumerate(convs) if i not in val_idx], [c for i, c in enumerate(convs) if i in val_idx]


# ------------------------------------------------------------------------------------ examples
def prompt_text(history: Sequence[Dict[str, str]]) -> str:
    """The text the model is given before it speaks: exactly what chat.build_chat_prompt builds at inference."""
    return (render_conversation(list(history)) + "\n" if history else "") + ASSISTANT_TAG + "\n"


def encode_example(tok, history: Sequence[Dict[str, str]], reply: str, block_size: int,
                   add_eos: bool = True) -> Optional[Example]:
    """(ids, labels) for teaching `reply` after `history`. The prompt is left-truncated (oldest context
    first, keeping <bos>) when it does not fit; an over-long reply is cut at the end and then gets no <eos>
    (the model must not learn that a cut-off answer is a finished one)."""
    reply = reply.strip()
    if not reply:
        return None
    prompt = tok.encode(prompt_text(history), add_bos=True)
    answer = tok.encode(reply)
    eos = [tok.eos_id] if add_eos else []
    max_answer = max(1, block_size - 16)
    if len(answer) + len(eos) > max_answer:
        answer, eos = answer[:max_answer], []
    room = block_size - len(answer) - len(eos)
    if len(prompt) > room:
        prompt = prompt[:1] + prompt[len(prompt) - (room - 1):] if room > 1 else prompt[:1]
    ids = prompt + answer + eos
    labels = [IGNORE] * len(prompt) + answer + eos
    return ids, labels


def build_examples(convs: Sequence[Sequence[Dict[str, str]]], tok, block_size: int) -> List[Example]:
    """One example per assistant turn, with everything before it as context."""
    examples: List[Example] = []
    for msgs in convs:
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or i == 0:
                continue
            ex = encode_example(tok, msgs[:i], m["content"], block_size)
            if ex is not None:
                examples.append(ex)
    return examples


def pad_batch(batch: Sequence[Example], pad_id: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Right-pad to the longest example. Inputs are ids[:-1], targets are labels[1:] (next-token), with padded
    targets IGNOREd. Causal attention means padding on the right cannot affect earlier positions."""
    width = max(len(ids) - 1 for ids, _ in batch)
    xs, ys = [], []
    for ids, labels in batch:
        x, y = ids[:-1], labels[1:]
        xs.append(x + [pad_id] * (width - len(x)))
        ys.append(y + [IGNORE] * (width - len(y)))
    return xs, ys


def supervised_tokens(examples: Sequence[Example]) -> int:
    return sum(sum(1 for t in labels[1:] if t != IGNORE) for _, labels in examples)


def lr_at(step: int, total: int, max_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


# -------------------------------------------------------------------------------------- training
def run_sft(args) -> Dict[str, Any]:
    import torch

    from .checkpoint import load_checkpoint, save_checkpoint
    from .device import resolve_runtime
    from .model import ModelConfig, ZivaGPT
    from .tokenizer import save_run_tokenizer, tokenizer_or_exit

    rt = resolve_runtime()
    device_type = rt.device.split(":")[0]
    ckpt = load_checkpoint(args.init_from)
    tok = tokenizer_or_exit(args.init_from, ckpt, args.tokenizer_dir)     # the one the model was trained with
    cfg = ModelConfig.from_dict(ckpt["config"])
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"checkpoint vocab_size={cfg.vocab_size} but tokenizer has {tok.vocab_size}; pass --tokenizer_dir")
    if args.block_size:
        cfg.block_size = args.block_size
    cfg.dropout = args.dropout
    cfg.validate()

    convs = load_conversations(args.data)
    if not convs:
        raise SystemExit(f"no usable conversations found in {args.data}")
    train_convs, val_convs = split_conversations(convs, args.val_fraction, args.seed)
    train_ex = build_examples(train_convs, tok, cfg.block_size)
    val_ex = build_examples(val_convs, tok, cfg.block_size)
    if not train_ex:
        raise SystemExit("no trainable examples (every conversation needs a user turn followed by an assistant turn)")
    print(f"[sft] {len(convs)} conversation(s) -> {len(train_ex)} train / {len(val_ex)} val example(s); "
          f"{supervised_tokens(train_ex):,} supervised tokens | device {rt.device}")

    model = ZivaGPT(cfg).to(rt.device)
    model.load_state_dict(ckpt["model_state"])
    model.gradient_checkpointing = bool(args.grad_ckpt)
    behavior = True
    del ckpt
    opt = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), device_type)
    use_amp = device_type == "cuda" or (device_type == "cpu" and rt.dtype == "bfloat16")
    amp_dtype = torch.bfloat16 if rt.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    amp_dev = device_type if device_type != "mps" else "cpu"

    def to_tensors(batch):
        xs, ys = pad_batch(batch, tok.pad_id)
        return (torch.tensor(xs, dtype=torch.long, device=rt.device), torch.tensor(ys, dtype=torch.long, device=rt.device))

    def evaluate() -> Optional[float]:
        if not val_ex:
            return None
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ex), args.batch_size):
                x, y = to_tensors(val_ex[i:i + args.batch_size])
                n = int((y != IGNORE).sum())
                with torch.autocast(device_type=amp_dev, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                total += float(loss) * n
                count += n
        model.train()
        return total / max(count, 1)

    micro = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_ex) / micro))
    epochs = args.epochs
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))
    eval_every = args.eval_every or steps_per_epoch
    print(f"[sft] {epochs} epoch(s) x {steps_per_epoch} step(s) = {total_steps} optimizer step(s), "
          f"effective batch {micro} example(s); eval every {eval_every} step(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path, best_path = os.path.join(args.out_dir, "checkpoint.pt"), os.path.join(args.out_dir, "checkpoint_best.pt")
    # the run carries the vocabulary it was trained with, so serving finds it by itself (see tokenizer.resolve_tokenizer)
    tok_meta = save_run_tokenizer(tok, args.out_dir, replace=True)
    extra_base = {"behavior_tuned": behavior, "sft": {"data": list(args.data), "epochs": epochs, "lr": args.lr},
                  "tokenizer": tok_meta}
    rng = random.Random(args.seed)
    best, bad_evals, step, t0, final_step = float("inf"), 0, 0, time.time(), 0
    best_written = False
    model.train()
    stop = False
    for epoch in range(epochs):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        for start in range(0, len(order), micro):
            chunk = [train_ex[i] for i in order[start:start + micro]]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, warmup)
            opt.zero_grad(set_to_none=True)
            n_tok_total = sum(sum(1 for t in lab[1:] if t != IGNORE) for _, lab in chunk)
            loss_value = 0.0
            for j in range(0, len(chunk), args.batch_size):
                sub = chunk[j:j + args.batch_size]
                x, y = to_tensors(sub)
                n_sub = int((y != IGNORE).sum())
                with torch.autocast(device_type=amp_dev, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                    loss = loss * (n_sub / max(n_tok_total, 1))   # token-weighted across micro-batches
                (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
                loss_value += float(loss)
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                print(f"[sft] epoch {epoch + 1}/{epochs} step {step}/{total_steps} | train loss {loss_value:.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
            if step % eval_every == 0 or step == total_steps:
                val = evaluate()
                if val is not None:
                    print(f"[sft]   val loss {val:.4f} (ppl {math.exp(min(val, 20)):.1f})")
                    if val < best:
                        best, bad_evals = val, 0
                        save_checkpoint(best_path, model=model, optimizer=None, step=step, cfg=cfg,
                                        extra={**extra_base, "val_loss": val})
                        best_written = True
                        print(f"[sft]   new best -> {best_path}")
                    else:
                        bad_evals += 1
                        if args.patience and bad_evals >= args.patience:
                            print(f"[sft]   no improvement for {bad_evals} evaluation(s): stopping early "
                                  f"(best val loss {best:.4f} is already saved)")
                            stop = True
            final_step = step
            if stop:
                break
        if stop:
            break
    save_checkpoint(ckpt_path, model=model, optimizer=None, step=final_step, cfg=cfg,
                    extra={**extra_base, "val_loss": best if best != float("inf") else None})
    if not best_written:   # nothing to validate against: the final weights are what there is
        save_checkpoint(best_path, model=model, optimizer=None, step=final_step, cfg=cfg,
                        extra={**extra_base, "val_loss": None})
    print(f"[sft] done -> {ckpt_path} (serving loads {best_path})")
    return {"steps": final_step, "best_val_loss": None if best == float("inf") else best,
            "checkpoint": ckpt_path, "best": best_path}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assistant-only supervised fine-tuning")
    p.add_argument("--data", nargs="+", required=True, help="JSONL/JSON files, globs or folders of conversations")
    p.add_argument("--init_from", required=True, help="checkpoint to fine-tune (never modified)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--epochs", type=int, default=3, help="passes over the data (the overfitting guard; default 3)")
    p.add_argument("--patience", type=int, default=2, help="stop after this many evaluations without a better validation loss (0 = never)")
    p.add_argument("--batch_size", type=int, default=8, help="examples per micro-batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout while fine-tuning (off at inference)")
    p.add_argument("--block_size", type=int, default=None, help="context length (default: the model's)")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=0, help="steps between validations (default: once per epoch)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv=None) -> None:
    run_sft(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
