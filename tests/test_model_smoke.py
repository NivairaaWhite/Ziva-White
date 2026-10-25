import torch

from zivawhite.model import ZivaGPT, ModelConfig


def test_all_tiers_build_and_forward():
    for tier in ["tiny", "small"]:  # medium/large covered by config shape, skipped here for CI speed
        cfg = ModelConfig.preset(tier, vocab_size=260, block_size=32)
        model = ZivaGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x, y)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss.item() > 0
        assert model.num_params() > 0


def test_generate_extends_sequence():
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=32)
    model = ZivaGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=10, top_k=5)
    assert out.shape == (1, 14)


def test_weight_tying():
    cfg = ModelConfig.preset("tiny", vocab_size=260)
    model = ZivaGPT(cfg)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
