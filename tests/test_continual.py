import numpy as np
import torch

from zivawhite.continual import compute_fisher, ewc_penalty, load_consolidation, save_consolidation
from zivawhite.model import ZivaGPT, ModelConfig


def _make_synthetic_shard(tmp_path, vocab_size=64, n_tokens=20_000):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    arr[: n_tokens - 200].tofile(data_dir / "train.bin")
    arr[n_tokens - 200 :].tofile(data_dir / "val.bin")
    return str(data_dir)


def test_ewc_penalty_zero_at_snapshot_and_grows_with_drift(tmp_path):
    data_dir = _make_synthetic_shard(tmp_path)
    cfg = ModelConfig.preset("tiny", vocab_size=64, block_size=32)
    model = ZivaGPT(cfg)

    fisher = compute_fisher(model, data_dir, cfg.block_size, batch_size=4, device="cpu", n_batches=5)
    assert any(f.sum().item() > 0 for f in fisher.values())

    ckpt_path = str(tmp_path / "consolidation.pt")
    save_consolidation(ckpt_path, model, fisher)
    consolidation = load_consolidation(ckpt_path)

    penalty_at_snapshot = ewc_penalty(model, consolidation, lam=1000.0)
    assert penalty_at_snapshot.item() == 0.0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    penalty_after_drift = ewc_penalty(model, consolidation, lam=1000.0)
    assert penalty_after_drift.item() > 0.0
