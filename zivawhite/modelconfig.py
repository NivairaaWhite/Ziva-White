"""
modelconfig.py - the shape of a ZivaGPT, with no torch import.

Lives apart from model.py so that sizing, the compute planner, the HF exporter's config
mapping and the tests can all work without PyTorch installed.

Model size is ALWAYS an explicit choice. Nothing here (or anywhere else) looks at your RAM or VRAM
to decide how big your model may be: pick a named preset, and/or override any number:

    ModelConfig.preset("medium")                              # a named starting point
    ModelConfig.preset("small", n_layer=12, n_embd=768)       # a preset with some numbers changed
    ModelConfig(n_layer=20, n_head=16, n_kv_head=4, n_embd=1280, block_size=4096)   # fully custom

The only hardware-aware piece is `estimate_train_memory_gb`, which is *advice* (train.py prints a
warning if it thinks you will run out of memory); it never blocks or shrinks anything.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 512          # max context length
    dropout: float = 0.0
    bias: bool = False             # biases in Linear/Norm -> off is faster & regularizes
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5          # RMSNorm epsilon (Llama 1e-5, Qwen2 1e-6): matters when importing weights
    # SwiGLU inner width. None = the 8/3 * n_embd rule (rounded up to 64). Set it to run a model whose
    # width follows a different rule (e.g. Llama-3's 3.5x, or any imported checkpoint).
    intermediate_size: Optional[int] = None
    # Grouped-query attention: n_kv_head key/value heads shared by n_head query heads. None = n_head,
    # i.e. classic multi-head attention (what every checkpoint trained before this field exists uses,
    # and the state_dict layout is identical in that case). Fewer KV heads = a KV cache that is
    # n_head/n_kv_head times smaller, so longer contexts and bigger batches fit at inference.
    n_kv_head: Optional[int] = None
    # Context extension beyond block_size the model was trained at. "linear" = position interpolation
    # (positions divided by the factor); "ntk" = NTK-aware (theta scaled, positions untouched). Both
    # work best after a short fine-tune on long sequences; neither is free quality.
    rope_scaling: Optional[str] = None
    rope_scaling_factor: float = 1.0
    use_vq_bottleneck: bool = False   # experimental: see latent.py
    vq_num_codes: int = 512
    vq_commitment_cost: float = 0.25
    use_moe: bool = False              # experimental: sparse mixture-of-experts MLP, see model.py:MoESwiGLU
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_loss_coef: float = 0.01

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.n_head < 1 or self.n_layer < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError(f"head size n_embd/n_head = {self.n_embd // self.n_head} must be even (rotary embeddings rotate pairs)")
        if self.n_kv_head is not None and (self.n_kv_head < 1 or self.n_head % self.n_kv_head != 0):
            raise ValueError(f"n_head ({self.n_head}) must be a multiple of n_kv_head ({self.n_kv_head})")
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        if self.rope_scaling not in (None, "linear", "ntk"):
            raise ValueError(f"rope_scaling must be None, 'linear' or 'ntk', not {self.rope_scaling!r}")
        if self.rope_scaling_factor < 1.0:
            raise ValueError("rope_scaling_factor must be >= 1.0")

    # ------------------------------------------------------------------- derived quantities
    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        """SwiGLU inner width: `intermediate_size` if given, else 8/3 * n_embd (so the 3 matrices cost what
        a 4x GELU MLP would) rounded up to a multiple of 64."""
        if self.intermediate_size:
            return int(self.intermediate_size)
        hidden = int(8 * self.n_embd / 3)
        return ((hidden + 63) // 64) * 64

    @property
    def moe_hidden(self) -> int:
        hidden = int(8 * self.n_embd / 3 / max(1, self.moe_top_k))
        return max(64, ((hidden + 63) // 64) * 64)

    @property
    def effective_rope_theta(self) -> float:
        if self.rope_scaling == "ntk" and self.rope_scaling_factor > 1.0:
            hd = self.head_dim
            return self.rope_theta * self.rope_scaling_factor ** (hd / max(hd - 2, 1))
        return self.rope_theta

    @property
    def rope_position_scale(self) -> float:
        return 1.0 / self.rope_scaling_factor if self.rope_scaling == "linear" else 1.0

    @property
    def n_params_est(self) -> int:
        """Rough non-embedding parameter count (the classic 12*L*d^2, minus what GQA saves).
        Kept as the NAS size filter; use count_params() for the exact number."""
        d, kv = self.n_embd, self.kv_heads * self.head_dim
        return 12 * self.n_layer * d * d - 2 * self.n_layer * d * (d - kv)

    def count_params(self) -> int:
        """Exact non-embedding parameter count (matches ZivaGPT.num_params(); VQ bottleneck not counted)."""
        d, kv = self.n_embd, self.kv_heads * self.head_dim
        attn = d * (d + 2 * kv) + d * d
        if self.use_moe:
            mlp = d * self.moe_num_experts + self.moe_num_experts * 3 * d * self.moe_hidden
        else:
            mlp = 3 * d * self.mlp_hidden
        norms = 2 * d
        bias = 0
        if self.bias:
            bias = (d + 2 * kv) + d + (0 if self.use_moe else 2 * self.mlp_hidden + d)
        return self.n_layer * (attn + mlp + norms + bias) + d  # + final norm

    def total_params(self) -> int:
        """Everything, with the input and output embedding tied (counted once)."""
        return self.count_params() + self.vocab_size * self.n_embd

    def kv_cache_bytes(self, seq_len: Optional[int] = None, batch: int = 1, bytes_per_el: int = 2) -> int:
        """Inference KV-cache size: this is what GQA shrinks."""
        s = seq_len or self.block_size
        return 2 * self.n_layer * batch * s * self.kv_heads * self.head_dim * bytes_per_el

    # --------------------------------------------------------------------------- building
    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        """Tolerant of keys this version does not know (a checkpoint written by newer code)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def preset(cls, tier: str, vocab_size: int = 32000, block_size: Optional[int] = None, gqa: bool = True,
               **overrides) -> "ModelConfig":
        """A named starting point. `overrides` replace any field (None values are ignored, so CLI flags
        that were not passed can be forwarded as-is). gqa=False gives plain multi-head attention, e.g. to
        match an older checkpoint."""
        if tier not in PRESETS:
            raise ValueError(f"unknown tier {tier!r}, expected one of {list(PRESETS)}")
        cfg = dict(PRESETS[tier])
        if block_size is not None:
            cfg["block_size"] = block_size
        if not gqa:
            cfg["n_kv_head"] = None
        user_set = {k: v for k, v in overrides.items() if v is not None}
        cfg.update(user_set)
        # a preset's KV-head count only makes sense for the head count it was written for
        if "n_head" in user_set and "n_kv_head" not in user_set:
            kv = cfg.get("n_kv_head")
            if kv is not None and cfg["n_head"] % kv != 0:
                cfg["n_kv_head"] = None
        return cls(vocab_size=vocab_size, **cfg)


# Named sizes, all explicit choices (see module docstring). Parameter counts are for a 32k vocabulary:
#   tiny ~2M body | small ~23M | medium ~265M | large ~790M | xl ~1.24B   (+ vocab * n_embd for the tied embedding)
PRESETS: Dict[str, Dict] = {
    "tiny":   dict(n_layer=4,  n_head=4,  n_kv_head=2, n_embd=192,  block_size=256),
    "small":  dict(n_layer=8,  n_head=8,  n_kv_head=2, n_embd=512,  block_size=512),
    "medium": dict(n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, block_size=1024),
    "large":  dict(n_layer=32, n_head=24, n_kv_head=6, n_embd=1536, block_size=2048),
    "xl":     dict(n_layer=28, n_head=32, n_kv_head=8, n_embd=2048, block_size=2048),
}

# Per-preset training defaults (batch per process, grad-accum steps, total steps). Used only when you
# do not pass --batch_size/--grad_accum/--max_steps; custom sizes fall back to the "small" row.
TRAIN_DEFAULTS: Dict[str, Dict[str, int]] = {
    "tiny":   dict(batch_size=8,  grad_accum=8, max_steps=3000),
    "small":  dict(batch_size=16, grad_accum=4, max_steps=10000),
    "medium": dict(batch_size=32, grad_accum=2, max_steps=30000),
    "large":  dict(batch_size=64, grad_accum=1, max_steps=60000),
    "xl":     dict(batch_size=32, grad_accum=4, max_steps=100000),
}


def estimate_train_memory_gb(cfg: ModelConfig, batch_size: int, grad_checkpoint: bool = False) -> float:
    """ROUGH peak GPU memory for one training process, in GB (expect +-40%): fp32 master weights + grads +
    AdamW moments (16 bytes/param), mixed-precision activations (~34*s*b*d bytes per layer with fused
    attention, per Korthikanti et al. 2022; ~2*s*b*d with activation checkpointing plus one live layer),
    and the logits/cross-entropy buffers. Advice only; never used to refuse anything."""
    states = cfg.total_params() * 16
    s, b, d = cfg.block_size, batch_size, cfg.n_embd
    per_layer = 34 * s * b * d
    acts = (2 * s * b * d * cfg.n_layer + per_layer) if grad_checkpoint else per_layer * cfg.n_layer
    logits = 10 * b * s * cfg.vocab_size
    return (states + acts + logits) * 1.1 / (1024 ** 3)
