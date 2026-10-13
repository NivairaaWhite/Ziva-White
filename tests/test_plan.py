"""plan.py: the compute arithmetic is checked against published training-compute figures."""
import math

import pytest

from zivawhite.modelconfig import ModelConfig
from zivawhite.plan import GPUS, flops_per_token, format_plan, human, nearest_shape, parse_count, plan


def test_parse_count_and_human():
    assert parse_count("125m") == 1.25e8 and parse_count("1.3B") == 1.3e9 and parse_count("20b") == 2e10
    assert parse_count("350,000") == 350000 and parse_count("2t") == 2e12
    assert human(1.5e9) == "1.5B" and human(125e6) == "125M" and human(900) == "900"


def test_the_6ND_rule_reproduces_published_training_compute():
    # GPT-3: 175B params, 300B tokens -> ~3.14e23 FLOPs (Brown et al. 2020, table D.1)
    assert 6 * 175e9 * 300e9 == pytest.approx(3.15e23, rel=0.01)
    # Chinchilla: 70B params, 1.4T tokens -> ~5.76e23 (Hoffmann et al. 2022)
    assert 6 * 70e9 * 1.4e12 == pytest.approx(5.9e23, rel=0.05)
    # with no attention term the planner is exactly 6N per token; the attention term only ever adds
    assert flops_per_token(1e9, 24, 2048, 0) == 6e9 and flops_per_token(1e9, 24, 2048, 2048) > 6e9


def test_time_for_a_given_model_and_token_budget():
    p = plan(preset="small", tokens=1e9, gpu="a100", n_gpus=1, mfu=0.4, context=512)
    n = p["params"]
    per_tok = 6 * n + 12 * 8 * 512 * 512
    assert p["tokens_per_second"] == pytest.approx(312e12 * 0.4 / per_tok, rel=1e-9)
    assert p["seconds"] == pytest.approx(1e9 / p["tokens_per_second"], rel=1e-9)
    assert p["mode"] == "time_for_tokens" and p["tokens_per_param"] == pytest.approx(1e9 / n)


def test_more_gpus_or_efficiency_means_proportionally_less_time():
    base = plan(preset="medium", tokens=5e9, gpu="rtx4090", n_gpus=1, mfu=0.3)["seconds"]
    assert plan(preset="medium", tokens=5e9, gpu="rtx4090", n_gpus=4, mfu=0.3)["seconds"] == pytest.approx(base / 4)
    assert plan(preset="medium", tokens=5e9, gpu="rtx4090", n_gpus=1, mfu=0.6)["seconds"] == pytest.approx(base / 2)
    assert plan(preset="medium", tokens=5e9, tflops=100.0, mfu=0.3)["peak_tflops"] == 100.0       # unlisted GPUs work


def test_tokens_affordable_in_a_time_budget():
    p = plan(params=125e6, hours=10, gpu="a100", mfu=0.35)
    assert p["mode"] == "tokens_for_budget" and p["tokens"] == pytest.approx(p["tokens_per_second"] * 36000)


def test_a_budget_alone_gives_the_chinchilla_optimal_size():
    p = plan(hours=100, gpu="a100", n_gpus=8, mfu=0.4)
    budget = 312e12 * 0.4 * 8 * 100 * 3600
    assert p["budget_flops"] == pytest.approx(budget)
    assert p["compute_optimal_params"] < math.sqrt(budget / 120)                           # attention FLOPs shrink the optimum a little
    assert p["shape_params"] == pytest.approx(p["compute_optimal_params"], rel=0.12)       # the concrete shape is close
    assert 17 <= p["tokens_per_param"] <= 25                                              # ~20 tokens/param by construction
    assert p["overtrained_alternative"]["params"] < p["shape_params"] / 2
    assert p["overtrained_alternative"]["tokens_per_param"] > 2 * p["tokens_per_param"]


def test_nearest_shape_returns_valid_models_near_the_target_size():
    for target in (25e6, 125e6, 350e6, 1.3e9, 3e9):
        cfg = nearest_shape(target, 2048)
        assert isinstance(cfg, ModelConfig) and abs(cfg.total_params() - target) / target < 0.15
        assert cfg.n_head % cfg.kv_heads == 0 and cfg.head_dim % 2 == 0
        assert 20 <= cfg.n_embd / cfg.n_layer <= 90, (target, cfg.n_layer, cfg.n_embd)   # not a 6-layer, 2560-wide oddity


def test_a_half_billion_parameter_model_is_deep_not_a_wide_stub():
    cfg = nearest_shape(500e6, 2048)
    assert cfg.n_layer >= 16


def test_memory_advice_flags_what_will_not_fit_and_never_blocks_anything():
    big = plan(preset="xl", tokens=1e9, gpu="t4")
    assert big["fits"] is False                                                  # 1.3B on a 16 GB T4 at batch 8
    assert plan(preset="tiny", tokens=1e8, gpu="t4")["fits"] is True
    assert big["memory_gb_per_gpu_grad_ckpt"] < big["memory_gb_per_gpu"]
    assert "does NOT fit" in format_plan(big)                                    # advice, in plain words


def test_bad_inputs_are_explained():
    with pytest.raises(ValueError, match="unknown gpu"):
        plan(preset="small", tokens=1e9, gpu="quantum9000")
    with pytest.raises(ValueError, match="mfu"):
        plan(preset="small", tokens=1e9, mfu=2.0)
    with pytest.raises(ValueError):
        plan(gpu="a100")
    with pytest.raises(ValueError, match="tokens= or hours="):
        plan(preset="small")


def test_every_listed_gpu_has_sane_numbers():
    assert all(0 < t < 3000 and 8 <= m <= 192 for t, m in GPUS.values())
