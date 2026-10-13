"""The ZivaGPT <-> LlamaForCausalLM mapping, proven without PyTorch.

Two independent numpy transformers are written below: `ziva_forward` follows zivawhite/model.py (interleaved
rotary pairs, fused [q|k|v] projection, repeat_interleave grouping) and `llama_forward` follows Hugging Face's
LlamaForCausalLM (rotate_half on split halves, separate q/k/v/o, repeat_kv, gate/up/down MLP). If the mapping in
hf_compat.py is right, converting the weights makes both produce the same logits to float64 precision."""
import numpy as np
import pytest

from zivawhite.hf_compat import (half_to_interleaved_perm, hf_config_dict, hf_to_zivagpt_state,
                                 interleaved_to_half_perm, reorder_heads, zivagpt_config_from_hf,
                                 zivagpt_to_hf_state)
from zivawhite.modelconfig import ModelConfig


# ----------------------------------------------------------------------------- shared numpy pieces
def rms(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(a):
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


def causal_attention(q, k, v):
    """q: (nh, T, hd), k/v: (nh, T, hd) already expanded to nh heads."""
    T, hd = q.shape[1], q.shape[2]
    att = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    att = np.where(np.tril(np.ones((T, T), bool)), att, -np.inf)
    return softmax(att) @ v


# -------------------------------------------------------------- ZivaGPT, transcribed from model.py
def ziva_rope(x, theta, pos_scale):
    """x: (nh, T, hd). model.py: x1, x2 = x[..., 0::2], x[..., 1::2]; stack([x1*c - x2*s, x1*s + x2*c]).flatten."""
    T, hd = x.shape[1], x.shape[2]
    freqs = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(T) * pos_scale, freqs)
    cos, sin = np.cos(ang)[None], np.sin(ang)[None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).reshape(x.shape)


def ziva_forward(sd, cfg, ids):
    hd, nh, nkv = cfg.head_dim, cfg.n_head, cfg.kv_heads
    n_rep = nh // nkv
    x = sd["tok_emb.weight"][ids]                                    # (T, d)
    T = len(ids)
    for i in range(cfg.n_layer):
        b = f"blocks.{i}."
        h = rms(x, sd[b + "ln1.weight"], cfg.norm_eps)
        qkv = h @ sd[b + "attn.qkv.weight"].T
        q, k, v = np.split(qkv, [nh * hd, nh * hd + nkv * hd], axis=-1)
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nkv, hd).transpose(1, 0, 2)
        v = v.reshape(T, nkv, hd).transpose(1, 0, 2)
        q = ziva_rope(q, cfg.effective_rope_theta, cfg.rope_position_scale)
        k = ziva_rope(k, cfg.effective_rope_theta, cfg.rope_position_scale)
        k, v = np.repeat(k, n_rep, axis=0), np.repeat(v, n_rep, axis=0)   # == torch.repeat_interleave(dim=1)
        y = causal_attention(q, k, v).transpose(1, 0, 2).reshape(T, nh * hd)
        x = x + y @ sd[b + "attn.proj.weight"].T
        h = rms(x, sd[b + "ln2.weight"], cfg.norm_eps)
        x = x + (silu(h @ sd[b + "mlp.w1.weight"].T) * (h @ sd[b + "mlp.w3.weight"].T)) @ sd[b + "mlp.w2.weight"].T
    x = rms(x, sd["ln_f.weight"], cfg.norm_eps)
    return x @ sd["tok_emb.weight"].T                                 # tied head


# ------------------------------------------------ Llama, transcribed from HF's LlamaForCausalLM
def llama_rope(x, theta, factor):
    """x: (nh, T, hd). HF: emb = cat(freqs, freqs); x*cos + rotate_half(x)*sin; rotate_half = cat(-x2, x1)."""
    T, hd = x.shape[1], x.shape[2]
    inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    inv_freq = inv_freq / factor                                       # HF "linear" rope scaling
    freqs = np.outer(np.arange(T), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb)[None], np.sin(emb)[None]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + np.concatenate([-x2, x1], axis=-1) * sin


def llama_forward(hf, hfcfg, ids):
    nh, nkv, hd = hfcfg["num_attention_heads"], hfcfg["num_key_value_heads"], hfcfg["head_dim"]
    n_rep = nh // nkv
    rs = hfcfg["rope_scaling"]
    factor = rs["factor"] if rs else 1.0
    x = hf["model.embed_tokens.weight"][ids]
    T = len(ids)
    for i in range(hfcfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms(x, hf[p + "input_layernorm.weight"], hfcfg["rms_norm_eps"])
        q = (h @ hf[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)
        k = (h @ hf[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        v = (h @ hf[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        q, k = llama_rope(q, hfcfg["rope_theta"], factor), llama_rope(k, hfcfg["rope_theta"], factor)
        # HF repeat_kv: (nkv, S, hd) -> expand (nkv, n_rep, S, hd) -> reshape (nkv*n_rep, S, hd)
        k = np.broadcast_to(k[:, None], (nkv, n_rep) + k.shape[1:]).reshape(nh, T, hd)
        v = np.broadcast_to(v[:, None], (nkv, n_rep) + v.shape[1:]).reshape(nh, T, hd)
        y = causal_attention(q, k, v).transpose(1, 0, 2).reshape(T, nh * hd)
        x = x + y @ hf[p + "self_attn.o_proj.weight"].T
        h = rms(x, hf[p + "post_attention_layernorm.weight"], hfcfg["rms_norm_eps"])
        gate, up = h @ hf[p + "mlp.gate_proj.weight"].T, h @ hf[p + "mlp.up_proj.weight"].T
        x = x + (silu(gate) * up) @ hf[p + "mlp.down_proj.weight"].T
    x = rms(x, hf["model.norm.weight"], hfcfg["rms_norm_eps"])
    return x @ hf["model.embed_tokens.weight"].T                       # tie_word_embeddings


# ------------------------------------------------------------------------------------ fixtures
def random_state(cfg, seed=0):
    rng = np.random.default_rng(seed)
    d, hd = cfg.n_embd, cfg.head_dim
    kv = cfg.kv_heads * hd

    def w(*shape, s=0.3):
        return rng.normal(0.0, s, shape)

    sd = {"tok_emb.weight": w(cfg.vocab_size, d), "ln_f.weight": 1 + w(d, s=0.1)}
    for i in range(cfg.n_layer):
        b = f"blocks.{i}."
        sd[b + "attn.qkv.weight"] = w(d + 2 * kv, d)
        sd[b + "attn.proj.weight"] = w(d, d)
        sd[b + "ln1.weight"], sd[b + "ln2.weight"] = 1 + w(d, s=0.1), 1 + w(d, s=0.1)
        sd[b + "mlp.w1.weight"], sd[b + "mlp.w3.weight"] = w(cfg.mlp_hidden, d), w(cfg.mlp_hidden, d)
        sd[b + "mlp.w2.weight"] = w(d, cfg.mlp_hidden)
    return sd


CONFIGS = {
    "mha":        dict(n_head=4, n_embd=32),
    "gqa_4to2":   dict(n_head=4, n_kv_head=2, n_embd=32),
    "gqa_8to2":   dict(n_head=8, n_kv_head=2, n_embd=64),
    "mqa":        dict(n_head=4, n_kv_head=1, n_embd=32),
    "linear_rope": dict(n_head=4, n_kv_head=2, n_embd=32, rope_scaling="linear", rope_scaling_factor=2.0),
    "ntk_rope":   dict(n_head=4, n_kv_head=2, n_embd=32, rope_scaling="ntk", rope_scaling_factor=4.0),
    "odd_mlp":    dict(n_head=4, n_kv_head=2, n_embd=32, intermediate_size=88, norm_eps=1e-6, rope_theta=500000.0),
}


def make(name):
    return ModelConfig(vocab_size=50, n_layer=3, block_size=16, **CONFIGS[name])


# ------------------------------------------------------------------------------------- the tests
@pytest.mark.parametrize("name", list(CONFIGS))
def test_converted_weights_reproduce_the_same_logits_in_the_llama_forward_pass(name):
    cfg = make(name)
    sd = random_state(cfg)
    ids = np.array([3, 17, 0, 42, 8, 8, 31, 5, 22, 49, 1, 7])
    ours = ziva_forward(sd, cfg, ids)
    hf = zivagpt_to_hf_state(sd, cfg)
    theirs = llama_forward(hf, hf_config_dict(cfg), ids)
    assert ours.shape == theirs.shape == (len(ids), cfg.vocab_size)
    np.testing.assert_allclose(theirs, ours, rtol=1e-9, atol=1e-9)


def test_without_the_row_reordering_the_two_conventions_really_do_disagree():
    """Guards the test itself: skipping the permutation must break equality, so the pass above means something."""
    cfg = make("gqa_4to2")
    sd = random_state(cfg)
    ids = np.arange(10)
    hf = zivagpt_to_hf_state(sd, cfg)
    hd = cfg.head_dim
    for i in range(cfg.n_layer):                      # undo the reordering -> naive copy of q/k rows
        qkv = sd[f"blocks.{i}.attn.qkv.weight"]
        hf[f"model.layers.{i}.self_attn.q_proj.weight"] = qkv[: cfg.n_head * hd]
        hf[f"model.layers.{i}.self_attn.k_proj.weight"] = qkv[cfg.n_head * hd: (cfg.n_head + cfg.kv_heads) * hd]
    wrong = llama_forward(hf, hf_config_dict(cfg), ids)
    assert np.abs(wrong - ziva_forward(sd, cfg, ids)).max() > 1e-3


@pytest.mark.parametrize("name", list(CONFIGS))
def test_export_then_import_is_lossless(name):
    cfg = make(name)
    sd = random_state(cfg)
    back = hf_to_zivagpt_state(zivagpt_to_hf_state(sd, cfg), cfg)
    assert set(back) == set(sd)
    for k in sd:
        np.testing.assert_array_equal(back[k], sd[k])


def test_permutations_are_inverses_and_group_evens_before_odds():
    for hd in (2, 8, 64, 128):
        p, q = interleaved_to_half_perm(hd), half_to_interleaved_perm(hd)
        assert sorted(p) == list(range(hd))
        assert [p[q[i]] for i in range(hd)] == list(range(hd))
        assert p[: hd // 2] == list(range(0, hd, 2)) and p[hd // 2:] == list(range(1, hd, 2))


def test_reorder_heads_touches_only_rows_within_each_head():
    w = np.arange(4 * 6 * 3).reshape(24, 3)            # 4 heads x head_dim 6
    out = reorder_heads(w, 4, 6, interleaved_to_half_perm(6))
    for h in range(4):                                  # each head still holds exactly its own rows
        assert sorted(out[h * 6:(h + 1) * 6, 0]) == sorted(w[h * 6:(h + 1) * 6, 0])
    with pytest.raises(ValueError):
        reorder_heads(w, 5, 6, interleaved_to_half_perm(6))


def test_exported_config_describes_the_model_exactly():
    cfg = ModelConfig.preset("small", vocab_size=12000)
    hf = hf_config_dict(cfg, bos_id=1, eos_id=2, pad_id=0, dtype="float16")
    assert hf["architectures"] == ["LlamaForCausalLM"] and hf["model_type"] == "llama"
    assert (hf["hidden_size"], hf["num_hidden_layers"], hf["num_attention_heads"], hf["num_key_value_heads"]) == (512, 8, 8, 2)
    assert hf["intermediate_size"] == 1408 and hf["head_dim"] == 64 and hf["tie_word_embeddings"] is True
    assert (hf["bos_token_id"], hf["eos_token_id"], hf["pad_token_id"]) == (1, 2, 0)
    assert hf["rope_scaling"] is None and hf["torch_dtype"] == "float16"


def test_exported_tensor_names_and_shapes_match_llama_for_every_layer():
    cfg = make("gqa_4to2")
    hf = zivagpt_to_hf_state(random_state(cfg), cfg)
    hd, d = cfg.head_dim, cfg.n_embd
    assert hf["model.layers.2.self_attn.q_proj.weight"].shape == (cfg.n_head * hd, d)
    assert hf["model.layers.2.self_attn.k_proj.weight"].shape == (cfg.kv_heads * hd, d)
    assert hf["model.layers.2.self_attn.v_proj.weight"].shape == (cfg.kv_heads * hd, d)
    assert hf["model.layers.0.mlp.gate_proj.weight"].shape == (cfg.mlp_hidden, d)
    assert hf["model.layers.0.mlp.down_proj.weight"].shape == (d, cfg.mlp_hidden)
    assert "lm_head.weight" not in hf                                    # tied: HF omits it
    assert len(hf) == 1 + cfg.n_layer * 9 + 1


def test_unsupported_models_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="MoE"):
        hf_config_dict(ModelConfig(n_head=2, n_embd=32, use_moe=True))
    with pytest.raises(ValueError, match="bias"):
        hf_config_dict(ModelConfig(n_head=2, n_embd=32, bias=True))
    base = {"model_type": "llama", "vocab_size": 100, "hidden_size": 64, "intermediate_size": 176,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 8192}
    assert zivagpt_config_from_hf(base).block_size == 2048              # training context is a choice, capped by default
    assert zivagpt_config_from_hf(base, block_size=512).block_size == 512
    with pytest.raises(ValueError, match="llama3"):
        zivagpt_config_from_hf({**base, "rope_scaling": {"rope_type": "llama3", "factor": 32.0}})
    with pytest.raises(ValueError, match="bias"):
        zivagpt_config_from_hf({**base, "model_type": "llama", "attention_bias": True})
    with pytest.raises(ValueError, match="untied"):
        zivagpt_config_from_hf({**base, "tie_word_embeddings": False})
    with pytest.raises(ValueError, match="model_type"):
        zivagpt_config_from_hf({**base, "model_type": "qwen2"})


def test_import_config_carries_the_shape_of_a_real_smollm2_style_model():
    # SmolLM2-135M-like numbers: 30 layers, hidden 576, 9 heads / 3 kv heads, MLP 1536, theta 100000, tied
    hf = {"model_type": "llama", "vocab_size": 49152, "hidden_size": 576, "intermediate_size": 1536,
          "num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3,
          "max_position_embeddings": 8192, "rope_theta": 100000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    cfg = zivagpt_config_from_hf(hf)
    assert (cfg.n_layer, cfg.n_head, cfg.kv_heads, cfg.head_dim, cfg.mlp_hidden) == (30, 9, 3, 64, 1536)
    assert cfg.rope_theta == 100000.0
    # round trip through the exporter's config: same architecture back out
    out = hf_config_dict(cfg)
    assert (out["hidden_size"], out["intermediate_size"], out["num_key_value_heads"], out["rope_theta"]) == (576, 1536, 3, 100000.0)
    # total parameters land where the real model's do (~135M) -> the mapping describes the right architecture
    assert 130e6 < cfg.total_params() < 140e6, cfg.total_params()
