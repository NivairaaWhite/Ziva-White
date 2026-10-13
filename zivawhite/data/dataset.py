"""
np.memmap-backed batch sampling. The whole point of memmap here is
that `train.bin` is never fully loaded into RAM -- the OS pages in
only the bytes actually touched by the random block indices sampled
this step. That's what lets this scale from a 2MB toy shard on an
8GB laptop to a multi-GB/terabyte shard on a server without changing
a line of code (matches the terabyte-scale data-engine requirement
carried over from the AsterWhite build).
"""
from __future__ import annotations

import os

import numpy as np


def get_batch(data_dir: str, split: str, block_size: int, batch_size: int, device: str):
    import torch  # lazy: token_count() and the dataset-folder tooling must work without torch

    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) <= block_size:
        raise ValueError(f"{path} has only {len(data)} tokens, need > block_size={block_size}")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):  # covers plain "cuda" and per-rank "cuda:N" under torchrun
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def token_count(data_dir: str, split: str) -> int:
    path = os.path.join(data_dir, f"{split}.bin")
    return os.path.getsize(path) // np.dtype(np.uint16).itemsize
