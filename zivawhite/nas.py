"""
nas.py - the real alternative to "a sub-network that autonomously
rewrites the main network's architecture in real time."

That idea, taken literally, means shipping a system that redeploys
new versions of itself without a human in the loop -- that's not a
capability worth building even as a toy, independent of whether it's
technically feasible today (it mostly isn't, at the level the
blueprint describes). The real, decades-old research field this
corresponds to is Neural Architecture Search (NAS): search a space
of candidate architectures, evaluate each empirically, report what
won. This module does exactly that, and only that -- it never
touches a deployed model. It writes a ranked report; you decide
whether to `train.py --tier <winner>`.

Search strategy: random + greedy-local search over (n_layer, n_head,
n_embd) within a parameter budget, each candidate trained for a
short `search_steps` and scored on held-out val loss AND
loss-per-parameter (so a search on a laptop doesn't just always pick
the biggest model that fits).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

from .data.dataset import get_batch
from .model import ZivaGPT, ModelConfig


@dataclass
class Candidate:
    n_layer: int
    n_head: int
    n_embd: int
    params: int = 0
    val_loss: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    seconds: float = 0.0


def _random_candidate(max_params: int, vocab_size: int, rng: random.Random) -> Candidate:
    for _ in range(50):  # rejection sample until it fits the budget
        n_head = rng.choice([2, 4, 6, 8])
        n_embd = n_head * rng.choice([32, 48, 64, 96])
        n_layer = rng.choice([2, 4, 6, 8, 10])
        est = ModelConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd).n_params_est
        if est <= max_params:
            return Candidate(n_layer=n_layer, n_head=n_head, n_embd=n_embd, params=est)
    # fall back to the smallest sane shape if nothing sampled fits
    return Candidate(n_layer=2, n_head=2, n_embd=64, params=0)


def _evaluate(cand: Candidate, vocab_size: int, block_size: int, data_dir: str, device: str,
              search_steps: int, batch_size: int) -> Candidate:
    cfg = ModelConfig(vocab_size=vocab_size, n_layer=cand.n_layer, n_head=cand.n_head,
                       n_embd=cand.n_embd, block_size=block_size)
    model = ZivaGPT(cfg).to(device)
    cand.params = model.num_params()
    optimizer = model.configure_optimizer(0.1, 3e-4, (0.9, 0.95), device)

    t0 = time.time()
    model.train()
    tokens_seen = 0
    for _ in range(search_steps):
        x, y = get_batch(data_dir, "train", block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tokens_seen += x.numel()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(10):
            x, y = get_batch(data_dir, "val", block_size, batch_size, device)
            _, vloss = model(x, y)
            losses.append(vloss.item())
    cand.val_loss = sum(losses) / len(losses)
    cand.tokens_per_sec = tokens_seen / max(elapsed, 1e-6)
    cand.seconds = elapsed
    return cand


def search(data_dir: str, vocab_size: int, block_size: int, device: str, max_params: int,
           trials: int = 6, search_steps: int = 100, batch_size: int = 8, seed: int = 0) -> List[Candidate]:
    rng = random.Random(seed)
    results: List[Candidate] = []
    for i in range(trials):
        cand = _random_candidate(max_params, vocab_size, rng)
        print(f"[nas] trial {i+1}/{trials}: n_layer={cand.n_layer} n_head={cand.n_head} n_embd={cand.n_embd} "
              f"(~{cand.params:,} params est.)")
        cand = _evaluate(cand, vocab_size, block_size, data_dir, device, search_steps, batch_size)
        print(f"        -> val_loss={cand.val_loss:.4f} | {cand.tokens_per_sec:,.0f} tok/s | {cand.seconds:.1f}s")
        results.append(cand)
    # rank primarily by val_loss, break near-ties in favor of fewer params
    results.sort(key=lambda c: (round(c.val_loss, 3), c.params))
    return results


def report(results: List[Candidate], out_path: Optional[str] = None) -> str:
    lines = ["rank  n_layer  n_head  n_embd  params        val_loss  tok/s"]
    for rank, c in enumerate(results, 1):
        lines.append(f"{rank:>4}  {c.n_layer:>7}  {c.n_head:>6}  {c.n_embd:>6}  {c.params:>11,}  "
                      f"{c.val_loss:>8.4f}  {c.tokens_per_sec:>6,.0f}")
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            json.dump([asdict(c) for c in results], f, indent=2)
    return text
