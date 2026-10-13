"""KV-cached generation must match full recompute, and generate()'s new hooks must work."""
import torch

from zivawhite.model import ModelConfig, ZivaGPT


def _model(**over):
    torch.manual_seed(0)
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=64)
    for k, v in over.items():
        setattr(cfg, k, v)
    return ZivaGPT(cfg).eval()


def test_cached_logits_match_full_forward():
    model = _model()
    ids = torch.randint(0, 260, (1, 30))
    full, _ = model(ids)                       # logits for the last position, no cache
    cached, past = model.forward_cached(ids)   # prefill
    assert torch.allclose(full[:, -1], cached[:, -1], atol=1e-4)
    # now decode 5 more tokens one at a time and compare each step with a full recompute
    seq = ids
    for _ in range(5):
        nxt = torch.randint(0, 260, (1, 1))
        seq = torch.cat([seq, nxt], dim=1)
        cached, past = model.forward_cached(nxt, past)
        full, _ = model(seq)
        assert torch.allclose(full[:, -1], cached[:, -1], atol=1e-4)


def test_cached_greedy_generation_equals_uncached_generation():
    model = _model()
    prompt = torch.randint(0, 260, (1, 12))
    # greedy (top_k=1) so the comparison is deterministic; 12 + 30 stays inside block_size=64
    a = model.generate(prompt, max_new_tokens=30, temperature=1.0, top_k=1, use_cache=True)
    b = model.generate(prompt, max_new_tokens=30, temperature=1.0, top_k=1, use_cache=False)
    assert torch.equal(a, b)


def test_generation_can_run_past_the_context_window():
    model = _model()
    prompt = torch.randint(0, 260, (1, 40))
    out = model.generate(prompt, max_new_tokens=100, top_k=5)   # 140 tokens through a 64-token window
    assert out.shape == (1, 140)


def test_prompt_longer_than_the_window_does_not_crash():
    model = _model()
    out = model.generate(torch.randint(0, 260, (1, 200)), max_new_tokens=5, top_k=5)
    assert out.shape == (1, 205)


def test_stop_fn_ends_generation_early():
    model = _model()
    prompt = torch.randint(0, 260, (1, 5))
    out = model.generate(prompt, max_new_tokens=50, stop_fn=lambda idx: idx.size(1) >= 5 + 3)
    assert out.shape == (1, 8)


def test_stats_report_mean_top1_confidence():
    model = _model()
    stats = {}
    model.generate(torch.randint(0, 260, (1, 5)), max_new_tokens=6, top_k=5, stats=stats)
    assert 0.0 < stats["mean_top1"] <= 1.0


def test_generate_text_returns_only_new_text_for_a_base_model():
    from zivawhite.chat import generate_text
    from zivawhite.tokenizer import ByteLevelTokenizer

    model = _model()
    tok = ByteLevelTokenizer()
    out = generate_text(model, tok, "PROMPT-MARKER", chat=False, max_new_tokens=8, top_k=5)
    assert "PROMPT-MARKER" not in out and "<bos>" not in out


def test_generate_text_handles_a_prompt_far_longer_than_the_window_in_chat_mode():
    from zivawhite.chat import generate_text
    from zivawhite.tokenizer import ByteLevelTokenizer

    model = _model()
    out = generate_text(model, ByteLevelTokenizer(), "word " * 2000, chat=True, max_new_tokens=8, top_k=5)
    assert isinstance(out, str)  # used to raise (HTTP 500) once the prompt exceeded block_size


def test_moe_output_projections_get_the_scaled_init():
    cfg = ModelConfig.preset("tiny", vocab_size=260, block_size=32)
    cfg.use_moe = True
    torch.manual_seed(0)
    model = ZivaGPT(cfg)
    w2 = [p.std().item() for n, p in model.named_parameters() if "experts_w2" in n]
    w1 = [p.std().item() for n, p in model.named_parameters() if "experts_w1" in n or "experts_w3" in n]
    assert w2 and w1 and max(w2) < min(w1)   # scaled by 1/sqrt(2*n_layer), so clearly smaller
