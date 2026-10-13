"""
latent.py - the real alternative to "the AI invents an alien language
to think in."

The blueprint's version of that idea is not implementable and, more
importantly, not something worth wanting: an internal representation
specifically designed to be opaque to the humans running it is an
interpretability problem, not a feature. The real, useful version of
"compress reasoning into a dense internal code instead of raw text"
already exists in the literature as a **discrete latent bottleneck**
(VQ-VAE, Oord et al. 2017; used for discrete "thought" representations
in several planning/world-model papers) -- and it's fully inspectable:
the codebook is just a matrix of vectors you can print, cluster, or
nearest-neighbor-decode at any time. Nothing is hidden from you by
construction.

This module is off by default (`ModelConfig.use_vq_bottleneck=False`).
Turn it on to experiment with routing the model's representation
through a fixed-size codebook before the output head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQBottleneck(nn.Module):
    def __init__(self, dim: int, num_codes: int = 512, commitment_cost: float = 0.25):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor):
        """z: (B, T, dim) -> (quantized, vq_loss, code_indices)"""
        flat = z.reshape(-1, self.dim)
        # squared L2 distance to every code, via ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        quantized = self.codebook(indices).view_as(z)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through estimator: forward uses the quantized value,
        # backward passes the gradient straight through as if this were identity
        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, indices.view(z.shape[0], z.shape[1])
