"""
dpo.py - Direct Preference Optimization: teach the model which of two answers is better.

    # pairs you already have (JSONL: prompt / chosen / rejected)
    python -m zivawhite.dpo --pairs data/prefs.jsonl --init_from runs/ziva_chat/checkpoint.pt --out_dir runs/ziva_dpo
    # no preference data? build pairs from your SFT data by self-play: chosen = the real reply, rejected = what
    # the current model says (the first round of SPIN, Chen et al. 2024), then train on them
    python -m zivawhite.dpo --from_sft data/chat/*.jsonl --pairs_out data/spin_pairs.jsonl \\
           --init_from runs/ziva_chat/checkpoint.pt --out_dir runs/ziva_dpo

WHERE IT FITS. DPO is a SECOND stage. It only reshapes a model that can already follow the chat format, so run
`sft` first; DPO on a base model has nothing to refine. It does not add knowledge or fix hallucination by itself:
it moves probability toward the style of the "chosen" answers and away from the "rejected" ones. On a very small
model expect modest effects (shorter, less repetitive, better formatted answers), not a leap in capability.

How it works. For a prompt x with chosen y_w and rejected y_l, with log-probabilities from the model being trained
(pi) and from a frozen copy of the starting model (ref):
    loss = -log sigmoid( beta * [ (pi(y_w|x) - ref(y_w|x)) - (pi(y_l|x) - ref(y_l|x)) ] )
The reference log-probabilities are computed ONCE up front (one pass over the data, no second model in memory).
Common failure on small models: DPO lowers the likelihood of BOTH answers. --sft_alpha adds a small
likelihood term on the chosen answer to hold it up (0.1-0.2 is a reasonable start).

Input formats (JSONL), all normalised by normalize_preference():
    {"prompt": "question", "chosen": "good answer", "rejected": "bad answer"}
    {"prompt": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}
    {"chosen": [{"role":"user",...},{"role":"assistant","content":"good"}], "rejected": [... "bad"]}   (UltraFeedback style)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sft import IGNORE, Example, encode_example, load_conversations, normalize_conversation, pad_batch

Pair = Tuple[List[Dict[str, str]], str, str]   # (history messages, chosen reply, rejected reply)


# ------------------------------------------------------------------------------- data
def _reply_of(value: Any) -> Optional[str]:
    """A reply given as a string, a message dict, or a message list (the last assistant message)."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return _reply_of(value.get("content"))
    if isinstance(value, list):
        for m in reversed(value):
            if isinstance(m, dict) and str(m.get("role", "")).lower() in ("assistant", "gpt", "bot", "model"):
                return _reply_of(m.get("content"))
    return None


def normalize_preference(rec: Any) -> Optional[Pair]:
    if not isinstance(rec, dict):
        return None
    chosen, rejected = _reply_of(rec.get("chosen")), _reply_of(rec.get("rejected"))
    if not chosen or not rejected or chosen == rejected:
        return None
    prompt = rec.get("prompt")
    history: Optional[List[Dict[str, str]]] = None
    if isinstance(prompt, str) and prompt.strip():
        history = [{"role": "user", "content": prompt.strip()}]
    elif isinstance(prompt, list):
        history = _context(prompt)
    elif isinstance(rec.get("chosen"), list):   # prompt implied: everything before the final assistant turn
        history = _context(rec["chosen"][:-1])
    if not history:
        return None
    return history, chosen, rejected


def _context(messages: Sequence[Any]) -> Optional[List[Dict[str, str]]]:
    """Clean context messages (no trailing assistant answer required): needs at least one user turn."""
    msgs = normalize_conversation(list(messages) + [{"role": "assistant", "content": "x"}])
    if not msgs:
        return None
    return msgs[:-1] or None


def load_preferences(sources: Sequence[str]) -> List[Pair]:
    paths: List[str] = []
    for s in sources:
        paths += sorted(glob.glob(os.path.join(s, "**", "*.json*"), recursive=True)) if os.path.isdir(s) else (sorted(glob.glob(s)) or [s])
    pairs: List[Pair] = []
    for path in paths:
        if not os.path.isfile(path) or not path.endswith((".json", ".jsonl")):
            continue
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                records = []
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        pairs += [p for p in map(normalize_preference, records) if p]
    return pairs


def encode_pair(tok, pair: Pair, block_size: int) -> Optional[Tuple[Example, Example]]:
    history, chosen, rejected = pair
    c = encode_example(tok, history, chosen, block_size)
    r = encode_example(tok, history, rejected, block_size)
    return (c, r) if c is not None and r is not None else None


def split_pairs(pairs: List[Any], val_fraction: float, seed: int, min_for_val: int = 20) -> Tuple[List[Any], List[Any]]:
    if len(pairs) < min_for_val:
        return list(pairs), []
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(pairs) * val_fraction))
    val = set(order[:n_val])
    return [p for i, p in enumerate(pairs) if i not in val], [p for i, p in enumerate(pairs) if i in val]


# ----------------------------------------------------------------------------- self-play pairs
def build_self_play_pairs(model, tok, convs: Sequence[Sequence[Dict[str, str]]], *, device: str, limit: int,
                          max_new_tokens: int = 160, temperature: float = 0.8, seed: int = 1337) -> List[Pair]:
    """chosen = the reply in the data, rejected = what `model` says for the same context. Turns where the model
    already says exactly the data's reply are skipped (no learning signal)."""
    import torch

    from .chat import generate_text

    rng = random.Random(seed)
    turns = [(list(m[:i]), m[i]["content"]) for m in convs for i in range(1, len(m)) if m[i]["role"] == "assistant"]
    rng.shuffle(turns)
    torch.manual_seed(seed)
    pairs: List[Pair] = []
    for history, gold in turns[: max(limit * 2, limit)]:
        if len(pairs) >= limit:
            break
        if not history or history[-1]["role"] != "user":
            continue   # generate_text renders its first argument as a user turn, so only user-last contexts fit
        # the generator wants (user text, earlier history); rebuild that split from the context
        last = history[-1]
        sample = generate_text(model, tok, last["content"], chat=True, device=device, max_new_tokens=max_new_tokens,
                               temperature=temperature, history=history[:-1], repetition_penalty=1.0)
        if sample.strip() and sample.strip() != gold.strip():
            pairs.append((history, gold.strip(), sample.strip()))
    return pairs


def write_pairs(path: str, pairs: Sequence[Pair]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for history, chosen, rejected in pairs:
            f.write(json.dumps({"prompt": history, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------- the maths
def sequence_logps(model, x, y):
    """Sum of log p(token) over each row's supervised positions (y != IGNORE), plus the token counts. x, y: (B, T)."""
    import torch

    logits = model.hidden_logits(x).float()                       # (B, T, V)
    mask = y != IGNORE
    safe = y.clamp(min=0)
    token_logit = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    logp = token_logit - torch.logsumexp(logits, dim=-1)
    logp = logp * mask
    return logp.sum(-1), mask.sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float, label_smoothing: float = 0.0):
    """Per-pair DPO loss and the implicit-reward margin (beta * log-ratio difference). Inputs: (B,) tensors."""
    import torch.nn.functional as F

    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    return loss, margin


def run_dpo(args) -> Dict[str, Any]:
    import torch

    from .checkpoint import load_checkpoint, save_checkpoint
    from .device import resolve_runtime
    from .model import ModelConfig, ZivaGPT
    from .tokenizer import load_tokenizer

    rt = resolve_runtime()
    device_type = rt.device.split(":")[0]
    tok = load_tokenizer(args.tokenizer_dir)
    ckpt = load_checkpoint(args.init_from)
    cfg = ModelConfig.from_dict(ckpt["config"])
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"checkpoint vocab_size={cfg.vocab_size} but tokenizer has {tok.vocab_size}; pass --tokenizer_dir")
    if args.block_size:
        cfg.block_size = args.block_size
    cfg.dropout = args.dropout            # keep it 0 so the policy and the frozen reference are compared fairly
    model = ZivaGPT(cfg).to(rt.device)
    model.load_state_dict(ckpt["model_state"])
    if not (ckpt.get("extra") or {}).get("behavior_tuned"):
        print("[dpo] WARNING: this checkpoint is not marked chat-tuned. DPO refines a model that already follows the "
              "chat format; run `sft` first or the result will not mean much.")
    del ckpt

    pairs: List[Pair] = []
    if args.pairs:
        pairs += load_preferences(args.pairs)
    if args.from_sft:
        convs = load_conversations(args.from_sft)
        print(f"[dpo] building up to {args.n_self_play} self-play pair(s) from {len(convs)} conversation(s) ...")
        model.eval()
        built = build_self_play_pairs(model, tok, convs, device=rt.device, limit=args.n_self_play,
                                      max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed)
        if args.pairs_out:
            write_pairs(args.pairs_out, built)
            print(f"[dpo] wrote {len(built)} pair(s) -> {args.pairs_out}")
        pairs += built
    if args.build_only:
        return {"pairs": len(pairs)}
    if not pairs:
        raise SystemExit("no preference pairs: pass --pairs FILE and/or --from_sft FILE")
    encoded = [e for e in (encode_pair(tok, p, cfg.block_size) for p in pairs) if e is not None]
    train_p, val_p = split_pairs(encoded, args.val_fraction, args.seed)
    print(f"[dpo] {len(train_p)} train / {len(val_p)} val pair(s), beta={args.beta}, sft_alpha={args.sft_alpha}")

    use_amp = device_type == "cuda" or (device_type == "cpu" and rt.dtype == "bfloat16")
    amp_dtype = torch.bfloat16 if rt.dtype == "bfloat16" else torch.float16
    amp_dev = device_type if device_type != "mps" else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def tensors(exs: Sequence[Example]):
        xs, ys = pad_batch(exs, tok.pad_id)
        return (torch.tensor(xs, dtype=torch.long, device=rt.device), torch.tensor(ys, dtype=torch.long, device=rt.device))

    def policy_logps(batch):
        """(chosen logps, rejected logps, chosen token counts) with one forward over both halves."""
        exs = [c for c, _ in batch] + [r for _, r in batch]
        x, y = tensors(exs)
        with torch.autocast(device_type=amp_dev, dtype=amp_dtype, enabled=use_amp):
            lp, n = sequence_logps(model, x, y)
        b = len(batch)
        return lp[:b], lp[b:], n[:b]

    # frozen-reference log-probabilities, computed once with the starting weights
    def reference(pairs_):
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(pairs_), args.batch_size):
                c, r, _ = policy_logps(pairs_[i:i + args.batch_size])
                out += list(zip(c.tolist(), r.tolist()))
        return out

    print("[dpo] computing reference log-probabilities ...")
    ref_train, ref_val = reference(train_p), reference(val_p)

    def evaluate() -> Optional[Dict[str, float]]:
        if not val_p:
            return None
        model.eval()
        tot, acc, marg, n = 0.0, 0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_p), args.batch_size):
                batch = val_p[i:i + args.batch_size]
                c, r, _ = policy_logps(batch)
                rc = torch.tensor([x[0] for x in ref_val[i:i + len(batch)]], device=rt.device)
                rr = torch.tensor([x[1] for x in ref_val[i:i + len(batch)]], device=rt.device)
                loss, m = dpo_loss(c, r, rc, rr, args.beta, args.label_smoothing)
                tot += float(loss.sum()); acc += int((m > 0).sum()); marg += float(m.sum()); n += len(batch)
        model.train()
        return {"loss": tot / n, "reward_acc": acc / n, "margin": marg / n}

    opt = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), device_type)
    steps_per_epoch = max(1, math.ceil(len(train_p) / args.batch_size))
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(0.1 * total))
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path, best_path = os.path.join(args.out_dir, "checkpoint.pt"), os.path.join(args.out_dir, "checkpoint_best.pt")
    extra = {"behavior_tuned": True, "dpo": {"beta": args.beta, "lr": args.lr, "epochs": args.epochs, "pairs": len(train_p)}}
    rng = random.Random(args.seed)
    best, bad, step, t0, final_step, best_written, stop = float("inf"), 0, 0, time.time(), 0, False, False
    base = evaluate()
    if base:
        print(f"[dpo] before training: val loss {base['loss']:.4f} (= ln2 = 0.693 when policy == reference), "
              f"reward accuracy {base['reward_acc']:.2%}")
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(train_p)))
        rng.shuffle(order)
        for s in range(0, len(order), args.batch_size):
            idxs = order[s:s + args.batch_size]
            batch = [train_p[i] for i in idxs]
            lr = args.lr * (min(1.0, (step + 1) / warmup) if step < warmup else
                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            c, r, n_c = policy_logps(batch)
            rc = torch.tensor([ref_train[i][0] for i in idxs], device=rt.device)
            rr = torch.tensor([ref_train[i][1] for i in idxs], device=rt.device)
            loss, margin = dpo_loss(c, r, rc, rr, args.beta, args.label_smoothing)
            total_loss = loss.mean()
            if args.sft_alpha > 0:                       # hold up the likelihood of the chosen answers
                total_loss = total_loss + args.sft_alpha * (-(c / n_c.clamp(min=1))).mean()
            (scaler.scale(total_loss) if scaler.is_enabled() else total_loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            step += 1
            final_step = step
            if step % args.log_every == 0:
                print(f"[dpo] epoch {epoch + 1}/{args.epochs} step {step}/{total} | loss {float(loss.mean()):.4f} | "
                      f"reward acc {float((margin > 0).float().mean()):.2f} | margin {float(margin.mean()):+.3f} | "
                      f"{time.time() - t0:.0f}s")
        val = evaluate()
        if val:
            print(f"[dpo]   val loss {val['loss']:.4f} | reward accuracy {val['reward_acc']:.2%} | margin {val['margin']:+.3f}")
            if val["loss"] < best:
                best, bad = val["loss"], 0
                save_checkpoint(best_path, model=model, optimizer=None, step=step, cfg=cfg, extra={**extra, "val_loss": None, "dpo_val_loss": best})
                best_written = True
                print(f"[dpo]   new best -> {best_path}")
            else:
                bad += 1
                if args.patience and bad >= args.patience:
                    print(f"[dpo]   no improvement for {bad} evaluation(s): stopping early")
                    stop = True
        if stop:
            break
    save_checkpoint(ckpt_path, model=model, optimizer=None, step=final_step, cfg=cfg, extra=extra)
    if not best_written:
        save_checkpoint(best_path, model=model, optimizer=None, step=final_step, cfg=cfg, extra=extra)
    print(f"[dpo] done -> {ckpt_path} (serving loads {best_path})")
    return {"steps": final_step, "best_val_loss": None if best == float("inf") else best, "pairs": len(train_p)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct Preference Optimization")
    p.add_argument("--init_from", required=True, help="chat-tuned checkpoint to refine (never modified)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--pairs", nargs="*", default=[], help="preference JSONL file(s) with prompt/chosen/rejected")
    p.add_argument("--from_sft", nargs="*", default=[], help="SFT conversation files to build self-play pairs from")
    p.add_argument("--n_self_play", type=int, default=500, help="how many self-play pairs to generate")
    p.add_argument("--pairs_out", default=None, help="write the generated self-play pairs here")
    p.add_argument("--build_only", action="store_true", help="only build/write pairs; do not train")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft_alpha", type=float, default=0.0, help="weight of an NLL term on chosen answers (0.1-0.2 helps small models)")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4, help="pairs per step (each is two sequences)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv=None) -> None:
    run_dpo(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
