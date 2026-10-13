import torch

from zivawhite.model import ZivaGPT, ModelConfig


def test_vq_bottleneck_forward_backward():
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=32)
    cfg.use_vq_bottleneck = True
    cfg.vq_num_codes = 32
    model = ZivaGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    loss.backward()
    assert model.vq.codebook.weight.grad is not None
    assert model.last_vq_indices.shape == (2, 16)
    assert model.last_vq_indices.max().item() < cfg.vq_num_codes


def test_moe_forward_backward_routes_and_trains_all_experts():
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=32)
    cfg.use_moe = True
    cfg.moe_num_experts = 4
    cfg.moe_top_k = 2
    model = ZivaGPT(cfg)
    # enough tokens that, with random init, every expert gets picked at least once
    x = torch.randint(0, cfg.vocab_size, (8, 32))
    y = torch.randint(0, cfg.vocab_size, (8, 32))
    logits, loss = model(x, y)
    loss.backward()
    for e in range(cfg.moe_num_experts):
        assert model.blocks[0].mlp.experts_w1[e].weight.grad is not None
    assert model.blocks[0].mlp.last_aux_loss.item() >= 0


def test_moe_and_vq_can_combine():
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=32)
    cfg.use_moe, cfg.use_vq_bottleneck = True, True
    cfg.moe_num_experts, cfg.vq_num_codes = 2, 16
    model = ZivaGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss)
