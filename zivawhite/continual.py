"""
continual.py - the real alternative to "protects its knowledge using
lifelong learning."

The blueprint namedrops Elastic Weight Consolidation (Kirkpatrick et
al., 2017, PNAS) correctly -- it's a real, well-established technique,
not a hand-wave. This module implements it as it actually works: after
training on a dataset ("task A"), estimate each parameter's importance
via the diagonal of the Fisher information matrix (squared gradient of
the log-likelihood, averaged over a sample of task-A data). When you
later continue training on new data ("task B"), add a quadratic
penalty that discourages moving importance-weighted parameters far
from their task-A values. High-Fisher parameters (ones that mattered a
lot for task A) get pulled back harder; low-Fisher ones are free to
adapt. This is what actually connects to the "trains itself on
whatever it finds online, continuously" goal -- without it, each new
round of self-training data would just overwrite what came before.
"""
from __future__ import annotations

from typing import Dict

import torch

from .data.dataset import get_batch


@torch.no_grad()
def _snapshot_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def compute_fisher(model: torch.nn.Module, data_dir: str, block_size: int, batch_size: int,
                    device: str, n_batches: int = 50) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher information estimate over n_batches * batch_size sequences of
    task-A data. Does not touch the optimizer or update any weights.

    Each sequence gets its own backward pass. The Fisher is the average of SQUARED
    per-sample gradients; this used to square the gradient of a whole batch's mean
    loss instead, which averages gradients out before squaring and so underestimates
    the Fisher by roughly the batch size. (A sequence is still the unit, not a single
    token -- its loss is the mean over its tokens -- so values are not the paper's exact
    per-token Fisher.) This is `batch_size` times as many backward passes as before, and
    the larger values mean an `--ewc_lambda` tuned against old consolidation files will
    now bite harder: re-run `consolidate` and retune lambda."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    n_samples = 0
    for _ in range(n_batches):
        xb, yb = get_batch(data_dir, "train", block_size, batch_size, device)
        for i in range(xb.size(0)):
            model.zero_grad(set_to_none=True)
            _, loss = model(xb[i:i + 1], yb[i:i + 1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_samples += 1

    for n in fisher:
        fisher[n] /= max(1, n_samples)
    model.zero_grad(set_to_none=True)
    return fisher


def save_consolidation(path: str, model: torch.nn.Module, fisher: Dict[str, torch.Tensor]) -> None:
    torch.save({"fisher": fisher, "star_params": _snapshot_params(model)}, path)


def load_consolidation(path: str, map_location: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
    return torch.load(path, map_location=map_location, weights_only=False)


def ewc_penalty(model: torch.nn.Module, consolidation: Dict[str, Dict[str, torch.Tensor]],
                 lam: float) -> torch.Tensor:
    """Quadratic penalty pulling parameters back toward their task-A
    values, weighted by how important (high-Fisher) each one was."""
    fisher, star = consolidation["fisher"], consolidation["star_params"]
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n].to(p.device) * (p - star[n].to(p.device)).pow(2)).sum()
    return 0.5 * lam * penalty
