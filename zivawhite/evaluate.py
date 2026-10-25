"""
evaluate.py - measure a checkpoint instead of guessing.

    python -m zivawhite.evaluate --checkpoint runs/ziva/checkpoint.pt --text data/heldout.txt
    python -m zivawhite.evaluate --checkpoint runs/chat/checkpoint.pt --chat data/heldout_chat.jsonl
    python -m zivawhite.evaluate --checkpoint runs/chat/checkpoint.pt --prompts prompts.txt --samples 3
    python -m zivawhite.app eval ...                                  # same thing

Three views of a model:

  --text FILE     Deterministic perplexity and BITS PER BYTE over a held-out text file (non-overlapping windows,
                  every token counted once; no random sampling noise). Bits-per-byte does not depend on the tokenizer,
                  so it is the number to compare when you change vocabulary or compare against another model.
                  Use text the model has NEVER trained on, or the number only measures memorisation.
  --chat FILE     Assistant-only loss/perplexity over held-out conversations (the same masking sft.py trains with).
  --prompts FILE  Generate from each line of FILE and report degeneration metrics: distinct-1/2 (variety) and the
                  repeated-4-gram rate (loops). Reads like a smoke test, not a benchmark: it tells you the model is
                  not stuck repeating itself; it does not tell you the answers are right.

No benchmark scores are produced here on purpose: published suites (MMLU, HellaSwag, GSM8K, ...) must be run with a
dedicated harness (e.g. EleutherAI's lm-evaluation-harness against the exported Hugging Face folder), and a model of
this size sits near chance on most of them.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence


# ------------------------------------------------------------------------- pure metrics (no torch)
def windows(ids: Sequence[int], block_size: int, stride: Optional[int] = None) -> List[List[int]]:
    """Cut `ids` into windows of block_size+1 tokens (inputs + the one extra target). With the default stride
    (= block_size) every token is predicted exactly once. A short tail with at least 2 tokens becomes a final window."""
    stride = stride or block_size
    out = []
    for start in range(0, len(ids) - 1, stride):
        w = list(ids[start:start + block_size + 1])
        if len(w) >= 2:
            out.append(w)
        if start + block_size + 1 >= len(ids):
            break
    return out


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Tokenizer-independent compression: total negative log-likelihood (in nats) -> bits, per UTF-8 byte of the text."""
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return total_nll_nats / math.log(2) / n_bytes


def perplexity(mean_nll_nats: float) -> float:
    return math.exp(min(mean_nll_nats, 50.0))


def distinct_n(tokens: Sequence[Any], n: int) -> float:
    """Fraction of n-grams that are unique (1.0 = no repeats, near 0 = a loop). Defined as 1.0 for too-short input."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def repeated_ngram_rate(tokens: Sequence[Any], n: int = 4) -> float:
    """Fraction of n-gram positions whose n-gram already occurred earlier in the same text."""
    seen, repeats, total = set(), 0, 0
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        total += 1
        repeats += g in seen
        seen.add(g)
    return repeats / total if total else 0.0


def degeneration_report(texts: Sequence[str]) -> Dict[str, float]:
    """Average variety / looping metrics over generated texts (split on whitespace)."""
    rows = [t.split() for t in texts if t.strip()]
    if not rows:
        return {"n": 0, "distinct_1": 0.0, "distinct_2": 0.0, "repeated_4gram_rate": 0.0, "empty_fraction": 1.0}
    avg = lambda f: sum(f(r) for r in rows) / len(rows)  # noqa: E731
    return {"n": len(rows), "distinct_1": avg(lambda r: distinct_n(r, 1)), "distinct_2": avg(lambda r: distinct_n(r, 2)),
            "repeated_4gram_rate": avg(repeated_ngram_rate), "empty_fraction": 1 - len(rows) / max(len(texts), 1)}


# ----------------------------------------------------------------------------- model-backed evals
def text_perplexity(model, tok, text: str, *, device: str, batch_size: int = 4) -> Dict[str, float]:
    import torch

    ids = tok.encode(text)
    wins = windows(ids, model.cfg.block_size)
    if not wins:
        raise ValueError("text is too short to evaluate")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(wins), batch_size):
            batch = wins[i:i + batch_size]
            width = max(len(w) for w in batch) - 1
            x = torch.full((len(batch), width), tok.pad_id, dtype=torch.long)
            y = torch.full((len(batch), width), -100, dtype=torch.long)
            for r, w in enumerate(batch):
                x[r, :len(w) - 1] = torch.tensor(w[:-1])
                y[r, :len(w) - 1] = torch.tensor(w[1:])
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)                           # mean over the non-ignored targets of this batch
            n = int((y != -100).sum())
            total_nll += float(loss) * n
            total_tokens += n
    n_bytes = len(text.encode("utf-8"))
    mean = total_nll / total_tokens
    return {"tokens": total_tokens, "bytes": n_bytes, "nll_per_token": mean, "perplexity": perplexity(mean),
            "bits_per_byte": bits_per_byte(total_nll, n_bytes), "bytes_per_token": n_bytes / max(len(ids), 1)}


def chat_loss(model, tok, conversations, *, device: str, batch_size: int = 8) -> Dict[str, float]:
    import torch

    from .sft import IGNORE, build_examples, pad_batch

    examples = build_examples(conversations, tok, model.cfg.block_size)
    if not examples:
        raise ValueError("no assistant turns to evaluate")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            xs, ys = pad_batch(examples[i:i + batch_size], tok.pad_id)
            x, y = torch.tensor(xs, device=device), torch.tensor(ys, device=device)
            _, loss = model(x, y)
            n = int((y != IGNORE).sum())
            total += float(loss) * n
            count += n
    return {"examples": len(examples), "assistant_tokens": count, "nll_per_token": total / count,
            "perplexity": perplexity(total / count)}


def sample_report(model, tok, prompts: Sequence[str], *, chat: bool, device: str, samples: int = 1,
                  max_new_tokens: int = 120, temperature: float = 0.7, repetition_penalty: float = 1.0) -> Dict[str, Any]:
    from .chat import generate_text

    outs = []
    for p in prompts:
        for _ in range(samples):
            outs.append((p, generate_text(model, tok, p, chat=chat, device=device, max_new_tokens=max_new_tokens,
                                          temperature=temperature, repetition_penalty=repetition_penalty)))
    return {"samples": outs, **degeneration_report([o for _, o in outs])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a ZivaWhite checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--text", default=None, help="held-out text file: perplexity and bits per byte")
    ap.add_argument("--chat", nargs="*", default=None, help="held-out conversation JSONL: assistant-only loss")
    ap.add_argument("--prompts", default=None, help="file with one prompt per line: generation + repetition metrics")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--raw", action="store_true", help="treat prompts as plain text continuation even for a chat model")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    a = ap.parse_args(argv)
    if not (a.text or a.chat is not None or a.prompts):
        ap.error("give at least one of --text, --chat, --prompts")

    from .checkpoint import load_model
    from .device import resolve_runtime
    from .tokenizer import tokenizer_or_exit

    model, ckpt, device, src = load_model(a.checkpoint, device=resolve_runtime().device)
    tok = tokenizer_or_exit(src, ckpt, a.tokenizer_dir)
    behavior = bool((ckpt.get("extra") or {}).get("behavior_tuned"))
    report: Dict[str, Any] = {"checkpoint": src, "step": ckpt.get("step"), "chat_tuned": behavior,
                              "params": model.cfg.total_params()}
    if a.text:
        with open(a.text, encoding="utf-8") as f:
            report["text"] = text_perplexity(model, tok, f.read(), device=device)
    if a.chat is not None:
        from .sft import load_conversations

        report["chat"] = chat_loss(model, tok, load_conversations(a.chat), device=device)
    if a.prompts:
        with open(a.prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        report["generation"] = sample_report(model, tok, prompts, chat=behavior and not a.raw, device=device,
                                             samples=a.samples, max_new_tokens=a.max_new_tokens,
                                             temperature=a.temperature, repetition_penalty=a.repetition_penalty)
    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return
    print(f"{src}  (step {report['step']}, {'chat-tuned' if behavior else 'base'}, {report['params'] / 1e6:.1f}M params)")
    if "text" in report:
        t = report["text"]
        print(f"text:  perplexity {t['perplexity']:.2f} | {t['bits_per_byte']:.3f} bits/byte | "
              f"{t['tokens']:,} tokens ({t['bytes_per_token']:.2f} bytes/token)")
    if "chat" in report:
        c = report["chat"]
        print(f"chat:  assistant-token perplexity {c['perplexity']:.2f} over {c['assistant_tokens']:,} tokens / {c['examples']} examples")
    if "generation" in report:
        g = report["generation"]
        print(f"gen:   distinct-1 {g['distinct_1']:.2f} | distinct-2 {g['distinct_2']:.2f} | repeated-4-gram rate "
              f"{g['repeated_4gram_rate']:.1%} | empty {g['empty_fraction']:.0%}")
        for p, o in g["samples"][:10]:
            print(f"  > {p}\n    {o}")


if __name__ == "__main__":
    main()
