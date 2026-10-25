"""
hf_compat.py - the exact mapping between ZivaGPT and Hugging Face's `LlamaForCausalLM`.

ZivaGPT *is* a Llama-architecture model (RMSNorm pre-norm, rotary embeddings, SwiGLU, grouped-query
attention, no biases, tied embeddings), so a trained checkpoint can be written out as a standard
Llama checkpoint. That one fact is what makes everything else work without rebuilding it here:

    transformers / PEFT / TRL     load it with AutoModelForCausalLM, no custom code, no trust_remote_code
    vLLM, TGI, SGLang             serve it (they all implement LlamaForCausalLM)
    llama.cpp -> GGUF -> Ollama   convert_hf_to_gguf.py reads the same folder (see export_hf.py)

and, in reverse, an open Llama-architecture checkpoint can be imported as a starting point.

The one non-obvious part is the rotary convention. ZivaGPT rotates the INTERLEAVED pairs
(x0,x1), (x2,x3), ...; Hugging Face's Llama rotates the SPLIT halves (x_i, x_{i+d/2}). Both compute the
same function if the rows of the q and k projections are reordered inside every head: even rows first,
then odd rows. That reordering (and its inverse) is `interleaved_to_half_perm` below, and it is proven
against an independent numpy reimplementation of both conventions in tests/test_hf_compat.py.

Everything here works on numpy arrays AND torch tensors (only slicing, reshape and fancy indexing),
and imports neither, so it can be tested without PyTorch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .modelconfig import ModelConfig


# ------------------------------------------------------------------------------ rotary permutation
def interleaved_to_half_perm(head_dim: int) -> List[int]:
    """new[j] = old[perm[j]]: all even positions, then all odd ones."""
    return list(range(0, head_dim, 2)) + list(range(1, head_dim, 2))


def half_to_interleaved_perm(head_dim: int) -> List[int]:
    """Inverse of interleaved_to_half_perm."""
    perm = interleaved_to_half_perm(head_dim)
    inv = [0] * head_dim
    for new_pos, old_pos in enumerate(perm):
        inv[old_pos] = new_pos
    return inv


def reorder_heads(w, n_heads: int, head_dim: int, perm: List[int]):
    """Reorder the rows of a (n_heads*head_dim, ...) projection matrix inside every head."""
    rows = w.shape[0]
    if rows != n_heads * head_dim:
        raise ValueError(f"expected {n_heads * head_dim} rows ({n_heads} heads x {head_dim}), got {rows}")
    rest = tuple(w.shape[1:])
    return w.reshape(n_heads, head_dim, *rest)[:, perm].reshape(rows, *rest)


def _cat(parts):
    if type(parts[0]).__module__.split(".")[0] == "numpy":
        import numpy as np

        return np.concatenate(parts, axis=0)
    import torch

    return torch.cat(parts, dim=0)


def _same(a, b) -> bool:
    try:
        return tuple(a.shape) == tuple(b.shape) and bool((a == b).all())
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------- weights
def zivagpt_to_hf_state(sd: Dict[str, Any], cfg: ModelConfig) -> Dict[str, Any]:
    """ZivaGPT state_dict -> LlamaForCausalLM state_dict (lm_head omitted: embeddings are tied)."""
    _check_exportable(cfg)
    hd = cfg.head_dim
    q_size, kv_size = cfg.n_head * hd, cfg.kv_heads * hd
    perm = interleaved_to_half_perm(hd)
    out: Dict[str, Any] = {"model.embed_tokens.weight": sd["tok_emb.weight"]}
    for i in range(cfg.n_layer):
        b, h = f"blocks.{i}.", f"model.layers.{i}."
        qkv = sd[b + "attn.qkv.weight"]
        q, k, v = qkv[:q_size], qkv[q_size:q_size + kv_size], qkv[q_size + kv_size:]
        out[h + "self_attn.q_proj.weight"] = reorder_heads(q, cfg.n_head, hd, perm)
        out[h + "self_attn.k_proj.weight"] = reorder_heads(k, cfg.kv_heads, hd, perm)
        out[h + "self_attn.v_proj.weight"] = v
        out[h + "self_attn.o_proj.weight"] = sd[b + "attn.proj.weight"]
        out[h + "input_layernorm.weight"] = sd[b + "ln1.weight"]
        out[h + "post_attention_layernorm.weight"] = sd[b + "ln2.weight"]
        out[h + "mlp.gate_proj.weight"] = sd[b + "mlp.w1.weight"]   # silu(gate) * up, then down
        out[h + "mlp.up_proj.weight"] = sd[b + "mlp.w3.weight"]
        out[h + "mlp.down_proj.weight"] = sd[b + "mlp.w2.weight"]
    out["model.norm.weight"] = sd["ln_f.weight"]
    return out


def hf_to_zivagpt_state(hf: Dict[str, Any], cfg: ModelConfig) -> Dict[str, Any]:
    """LlamaForCausalLM state_dict -> ZivaGPT state_dict. Refuses untied output embeddings."""
    _check_exportable(cfg)
    hd = cfg.head_dim
    inv = half_to_interleaved_perm(hd)
    emb = hf["model.embed_tokens.weight"]
    if "lm_head.weight" in hf and not _same(hf["lm_head.weight"], emb):
        raise ValueError("this checkpoint has a separate (untied) lm_head; ZivaGPT always ties the output "
                         "embedding to the input embedding, so it cannot represent it faithfully")
    out: Dict[str, Any] = {"tok_emb.weight": emb}
    for i in range(cfg.n_layer):
        b, h = f"blocks.{i}.", f"model.layers.{i}."
        q = reorder_heads(hf[h + "self_attn.q_proj.weight"], cfg.n_head, hd, inv)
        k = reorder_heads(hf[h + "self_attn.k_proj.weight"], cfg.kv_heads, hd, inv)
        out[b + "attn.qkv.weight"] = _cat([q, k, hf[h + "self_attn.v_proj.weight"]])
        out[b + "attn.proj.weight"] = hf[h + "self_attn.o_proj.weight"]
        out[b + "ln1.weight"] = hf[h + "input_layernorm.weight"]
        out[b + "ln2.weight"] = hf[h + "post_attention_layernorm.weight"]
        out[b + "mlp.w1.weight"] = hf[h + "mlp.gate_proj.weight"]
        out[b + "mlp.w3.weight"] = hf[h + "mlp.up_proj.weight"]
        out[b + "mlp.w2.weight"] = hf[h + "mlp.down_proj.weight"]
    out["ln_f.weight"] = hf["model.norm.weight"]
    return out


def _check_exportable(cfg: ModelConfig) -> None:
    if cfg.use_moe or cfg.use_vq_bottleneck:
        raise ValueError("MoE / VQ-bottleneck models are experimental ZivaGPT variants with no LlamaForCausalLM "
                         "equivalent; only dense models can be exported or imported")
    if cfg.bias:
        raise ValueError("bias=True models are not supported by the exporter (Llama checkpoints here are bias-free)")


# ---------------------------------------------------------------------------------------- config
def hf_config_dict(cfg: ModelConfig, *, bos_id: Optional[int] = None, eos_id: Optional[int] = None,
                   pad_id: Optional[int] = None, dtype: str = "float32") -> Dict[str, Any]:
    """The config.json of the exported LlamaForCausalLM."""
    _check_exportable(cfg)
    rope_scaling = None
    max_pos = cfg.block_size
    if cfg.rope_scaling == "linear" and cfg.rope_scaling_factor > 1.0:
        rope_scaling = {"rope_type": "linear", "type": "linear", "factor": float(cfg.rope_scaling_factor)}
        max_pos = int(cfg.block_size / cfg.rope_scaling_factor)  # HF: original length; the factor extends it
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.n_embd,
        "intermediate_size": cfg.mlp_hidden,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "num_key_value_heads": cfg.kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": max_pos,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.effective_rope_theta,   # NTK scaling is just a larger theta
        "rope_scaling": rope_scaling,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "use_cache": True,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "torch_dtype": dtype,
    }


def zivagpt_config_from_hf(hf: Dict[str, Any], *, vocab_size: Optional[int] = None,
                           block_size: Optional[int] = None) -> ModelConfig:
    """Build the ModelConfig for importing a Llama-architecture checkpoint, or explain precisely why it
    cannot be represented. `block_size` defaults to min(max_position_embeddings, 2048): the context you
    TRAIN at is your choice (and your memory's), not the 128k a model may advertise."""
    problems = []
    if hf.get("model_type") != "llama":
        problems.append(f"model_type is {hf.get('model_type')!r}; only 'llama' is supported "
                        "(Llama-2/3, SmolLM2 and other Llama-architecture models)")
    if hf.get("attention_bias") or hf.get("mlp_bias"):
        problems.append("it uses attention/MLP biases (e.g. Qwen2); ZivaGPT is bias-free")
    if hf.get("hidden_act", "silu") != "silu":
        problems.append(f"activation {hf.get('hidden_act')!r} is not SwiGLU/silu")
    rs = hf.get("rope_scaling")
    if rs:
        kind = rs.get("rope_type", rs.get("type"))
        problems.append(f"rope_scaling={kind!r} is not supported here (Llama-3.1/3.2 use 'llama3' scaling). "
                        "Pick a model with plain rotary embeddings, e.g. SmolLM2 or Llama-2-architecture models")
    if hf.get("tie_word_embeddings") is False:
        problems.append("untied output embeddings (e.g. Llama-3-8B); ZivaGPT ties them")
    if problems:
        raise ValueError("cannot import this model:\n  - " + "\n  - ".join(problems))
    n_head = int(hf["num_attention_heads"])
    kv = int(hf.get("num_key_value_heads") or n_head)
    ctx = int(hf.get("max_position_embeddings", 2048))
    return ModelConfig(
        vocab_size=int(vocab_size or hf["vocab_size"]),
        n_layer=int(hf["num_hidden_layers"]),
        n_head=n_head,
        n_kv_head=None if kv == n_head else kv,
        n_embd=int(hf["hidden_size"]),
        intermediate_size=int(hf["intermediate_size"]),
        block_size=int(block_size or min(ctx, 2048)),
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
    )
