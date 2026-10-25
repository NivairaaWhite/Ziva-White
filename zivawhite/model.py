"""
ZivaGPT: a from-scratch, owned (not a wrapped third-party checkpoint)
decoder-only transformer. Same architecture at every scale — only the
numbers in ModelConfig change between an 8GB laptop and a multi-GPU box.

Design choices (see docs/MODEL.md for the reasoning + sources):
  - Rotary position embeddings (RoPE)  -> no learned position table,
    extrapolates better to longer context than it was trained on.
  - Grouped-query attention (GQA)      -> n_kv_head key/value heads shared by n_head
    query heads: a KV cache n_head/n_kv_head times smaller (what lets long
    contexts and big batches fit at inference) for ~no quality cost. With
    n_kv_head == n_head it is plain multi-head attention and the weights are
    laid out exactly as before, so older checkpoints load unchanged.
  - RMSNorm + pre-norm                 -> the standard for training
    stability at every scale, cheaper than LayerNorm.
  - SwiGLU MLP                         -> better quality per parameter
    than plain GELU-MLP at negligible extra cost.
  - Weight-tied input/output embedding -> saves a full embedding
    matrix worth of parameters, matters a lot at the "tiny" tier.
  - Flash-attention via
    torch.nn.functional.scaled_dot_product_attention when available,
    falls back to a manual implementation automatically otherwise
    (e.g. older torch / CPU-only paths).
"""
from __future__ import annotations

import math
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ModelConfig lives in a torch-free module (sizing, planning and the HF exporter use it without
# PyTorch); it is re-exported here so `from zivawhite.model import ModelConfig` keeps working.
from .modelconfig import ModelConfig, PRESETS  # noqa: F401


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device, dtype, position_scale: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float() * position_scale  # position_scale < 1: linear interpolation
    angles = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); `offset` = absolute position of x's first token
    # (non-zero when decoding with a KV cache, so new tokens get their real positions)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[offset: offset + x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[offset: offset + x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:           # real checks, not asserts: `python -O` strips asserts
            raise ValueError(f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})")
        if cfg.n_head % cfg.kv_heads != 0:
            raise ValueError(f"n_head ({cfg.n_head}) must be a multiple of n_kv_head ({cfg.kv_heads})")
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.kv_heads
        self.n_rep = self.n_head // self.n_kv_head          # query heads per key/value head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_size = self.n_head * self.head_dim            # == n_embd
        self.kv_size = self.n_kv_head * self.head_dim
        # One fused projection laid out [ q | k | v ]. When n_kv_head == n_head this is byte-for-byte the
        # layout older checkpoints use ((3, n_head, head_dim) along the output dim).
        self.qkv = nn.Linear(cfg.n_embd, self.q_size + 2 * self.kv_size, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self._flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past=None, return_cache: bool = False):
        """Training calls this with just (x, cos, sin). For generation, `past` is
        the (k, v) cached from earlier tokens and `return_cache=True` also returns
        this call's (k, v) so the next step only has to process its new token.
        Cached keys are stored already rotated, at their absolute positions, and with only
        the n_kv_head key/value heads (that is where GQA saves memory)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)      # (B, n_head,    T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)   # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        offset = 0 if past is None else past[0].size(2)
        q, k = apply_rope(q, cos, sin, offset), apply_rope(k, cos, sin, offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        S = k.size(2)  # total keys = cached + new
        present = (k, v)  # the compact (un-repeated) heads are what gets cached

        if self.n_rep > 1:
            # each KV head serves n_rep CONSECUTIVE query heads (same grouping as Llama/HF `repeat_kv`)
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # New queries attend to every cached key and causally to the new ones. A single new
        # token needs no mask; with no cache, SDPA's own is_causal handles it (training path).
        def causal_mask() -> Optional[torch.Tensor]:
            if T == 1:
                return None
            return torch.ones(T, S, device=x.device, dtype=torch.bool).tril(diagonal=S - T)

        if self._flash:
            if past is None:
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
                )
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(), dropout_p=0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = causal_mask()
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return (y, present) if return_cache else y


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_hidden  # 8/3 * n_embd rounded up to a multiple of 64 (kernel friendly)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoESwiGLU(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer -- the architecture
    behind Mixtral/DeepSeek/Grok-style models, in miniature. Each token
    is routed to its top-k highest-scoring experts (out of N small
    SwiGLU experts) instead of going through one big dense MLP, so
    capacity scales with `n_experts` while compute per token stays
    fixed at `top_k` experts. A standard load-balancing auxiliary loss
    (Shazeer et al. 2017 / Switch Transformer, Fedus et al. 2021) is
    returned alongside the output -- without it, routers collapse onto
    a handful of favorite experts and the rest go untrained, which is
    the classic MoE failure mode.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.moe_num_experts
        self.top_k = cfg.moe_top_k
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"moe_top_k ({self.top_k}) must be between 1 and moe_num_experts ({self.n_experts})")
        self.gate = nn.Linear(cfg.n_embd, self.n_experts, bias=False)
        # each expert is a smaller SwiGLU so total MoE params stay comparable to one dense MLP
        hidden = int(8 * cfg.n_embd / 3 / max(1, self.top_k))
        hidden = max(64, ((hidden + 63) // 64) * 64)
        self.experts_w1 = nn.ModuleList(nn.Linear(cfg.n_embd, hidden, bias=cfg.bias) for _ in range(self.n_experts))
        self.experts_w2 = nn.ModuleList(nn.Linear(hidden, cfg.n_embd, bias=cfg.bias) for _ in range(self.n_experts))
        self.experts_w3 = nn.ModuleList(nn.Linear(cfg.n_embd, hidden, bias=cfg.bias) for _ in range(self.n_experts))
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.reshape(-1, C)                              # (N, C), N = B*T
        gate_logits = self.gate(flat)                         # (N, n_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_probs, top_idx = gate_probs.topk(self.top_k, dim=-1)  # (N, top_k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renormalize the chosen experts

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            expert_ids = top_idx[:, slot]                      # (N,) which expert each token picked in this slot
            weights = top_probs[:, slot].unsqueeze(-1)          # (N, 1)
            for e in range(self.n_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                tok = flat[mask]
                h = F.silu(self.experts_w1[e](tok)) * self.experts_w3[e](tok)
                out[mask] += weights[mask] * self.experts_w2[e](h)

        # load-balancing loss: encourage uniform routing across experts
        # (fraction of tokens routed to expert e) * (mean gate prob for expert e), summed, scaled by n_experts
        importance = gate_probs.mean(dim=0)
        with torch.no_grad():
            load = torch.zeros(self.n_experts, device=x.device)
            load.scatter_add_(0, top_idx.reshape(-1), torch.ones_like(top_idx, dtype=x.dtype).reshape(-1))
            load = load / load.sum().clamp_min(1.0)
        self.last_aux_loss = self.n_experts * (importance * load).sum()

        return self.dropout(out.view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MoESwiGLU(cfg) if cfg.use_moe else SwiGLU(cfg)
        self.is_moe = cfg.use_moe

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past=None, return_cache: bool = False):
        if return_cache:
            a, kv = self.attn(self.ln1(x), cos, sin, past=past, return_cache=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), kv
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class ZivaGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.vq: Optional["VQBottleneck"] = None
        if cfg.use_vq_bottleneck:
            from .latent import VQBottleneck  # local import: keeps latent.py fully optional

            self.vq = VQBottleneck(cfg.n_embd, cfg.vq_num_codes, cfg.vq_commitment_cost)
        self.last_vq_indices: Optional[torch.Tensor] = None

        # Activation checkpointing (train.py --grad_ckpt): recompute each block's activations in the
        # backward pass instead of storing them. ~30% slower, a large cut in activation memory.
        self.gradient_checkpointing = False

        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.effective_rope_theta, "cpu", torch.float32,
                                   position_scale=cfg.rope_position_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections, GPT-2 recipe
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight") or ".experts_w2." in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """targets may contain -100 at positions that must not be learned from (prompt tokens, padding):
        F.cross_entropy ignores that index. sft.py/dpo.py rely on this."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos.to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin.to(device=x.device, dtype=x.dtype)
        use_ckpt = self.gradient_checkpointing and self.training and not self.cfg.use_moe
        for block in self.blocks:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.ln_f(x)

        vq_loss = None
        if self.vq is not None:
            x, vq_loss, self.last_vq_indices = self.vq(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            if vq_loss is not None:
                loss = loss + vq_loss
            if self.cfg.use_moe:
                aux = sum(b.mlp.last_aux_loss for b in self.blocks if b.mlp.last_aux_loss is not None)
                loss = loss + self.cfg.moe_aux_loss_coef * aux
            return logits, loss
        # inference: only the last position needs a logit
        logits = self.head(x[:, [-1], :])
        return logits, None

    def hidden_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits for EVERY position, (B, T, vocab). Used by dpo.py to score whole responses."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos.to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        if self.vq is not None:
            x, _, self.last_vq_indices = self.vq(x)
        return self.head(x)

    @torch.no_grad()
    def forward_cached(self, idx: torch.Tensor, past=None):
        """Inference-only forward that reuses a KV cache. `past` is None for a
        fresh pass over `idx`, or the list of per-layer (k, v) a previous call
        returned, in which case `idx` holds only the NEW tokens. Returns
        (logits for the last position, new cache). Same maths as forward(), but
        generating N tokens costs O(N) token-forwards instead of O(N^2)."""
        B, T = idx.shape
        offset = 0 if past is None else past[0][0].size(2)
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence length {offset + T} > block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)
        cos = self.rope_cos.to(device=x.device, dtype=x.dtype)
        sin = self.rope_sin.to(device=x.device, dtype=x.dtype)
        present = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cos, sin, past=None if past is None else past[i], return_cache=True)
            present.append(kv)
        x = self.ln_f(x)
        if self.vq is not None:
            x, _, self.last_vq_indices = self.vq(x)
        return self.head(x[:, [-1], :]), present

    # ------------------------------------------------------------------------------ sampling
    @staticmethod
    def _sample(logits: torch.Tensor, idx: torch.Tensor, temperature: float, top_k: Optional[int],
                top_p: Optional[float], min_p: Optional[float], repetition_penalty: float,
                repetition_window: int) -> torch.Tensor:
        """Pick the next token id, shape (B, 1), from last-position `logits` (B, vocab).
        Order: repetition penalty -> temperature -> top-k -> top-p -> min-p -> sample."""
        if repetition_penalty and repetition_penalty != 1.0:
            # CTRL / HF-style: tokens already seen (the last `repetition_window` of prompt + reply) get
            # their logit pushed toward "less likely". Small models fall into loops without this.
            recent = idx[:, -repetition_window:] if repetition_window else idx
            seen = logits.gather(1, recent)
            seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
            logits = logits.scatter(1, recent, seen)
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            remove = cum > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

        if min_p:
            # keep tokens whose probability is at least min_p * (the best token's probability)
            probs = F.softmax(logits, dim=-1)
            keep = probs >= min_p * probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(~keep, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def _decode_steps(self, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: Optional[int],
                      top_p: Optional[float], min_p: Optional[float], repetition_penalty: float,
                      repetition_window: int, stats: Optional[dict], use_cache: bool) -> Iterator[torch.Tensor]:
        """Generator behind generate() and generate_stream(): yields one (B, 1) tensor of new token ids per
        step and keeps its own running copy of the sequence. The caller decides when to stop."""
        self.eval()
        block = self.cfg.block_size
        past = None
        top1_sum, top1_n = 0.0, 0
        for _ in range(max_new_tokens):
            if use_cache:
                if past is None or past[0][0].size(2) >= block:
                    # First step, or the cache has filled the window: re-prefill from
                    # the most recent tokens. After a full cache keep only half the
                    # window so this cost is spread over block//2 following steps.
                    keep = max(1, block - 1) if past is None else max(1, block // 2)
                    logits, past = self.forward_cached(idx[:, -keep:])
                else:
                    logits, past = self.forward_cached(idx[:, -1:], past)
            else:
                logits, _ = self(idx if idx.size(1) <= block else idx[:, -block:])
            logits = logits[:, -1, :]
            if stats is not None:
                top1_sum += F.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
                top1_n += 1
                stats["mean_top1"] = top1_sum / top1_n   # kept current, so it is right even if the caller stops early
            next_id = self._sample(logits, idx, temperature, top_k, top_p, min_p, repetition_penalty,
                                   repetition_window)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        stop_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        stats: Optional[dict] = None,
        use_cache: bool = True,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        repetition_window: int = 256,
    ) -> torch.Tensor:
        """Sample up to `max_new_tokens` tokens after `idx`; returns idx with the
        new tokens appended (prompt included).

        stop_fn(idx) -> bool is called after every new token and ends generation
        early when it returns True. If `stats` is a dict it gets "mean_top1": the
        average probability the model gave its single most likely token over the
        generated steps (a rough confidence signal). use_cache=False falls back to
        recomputing the whole context every step (slower; kept for testing).
        min_p / repetition_penalty (>1 discourages repeats) default to off."""
        for next_id in self._decode_steps(idx, max_new_tokens, temperature, top_k, top_p, min_p,
                                          repetition_penalty, repetition_window, stats, use_cache):
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            if stop_fn is not None and stop_fn(idx):
                break
        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        stop_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        stats: Optional[dict] = None,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        repetition_window: int = 256,
    ) -> Iterator[int]:
        """Like generate() for a single sequence (batch size 1), but yields each new token id as soon as
        it is sampled, so a server can stream text. The token that triggers a stop IS yielded (then the
        stream ends), exactly mirroring what generate() would have appended."""
        if idx.size(0) != 1:
            raise ValueError("generate_stream handles one sequence at a time")
        for next_id in self._decode_steps(idx, max_new_tokens, temperature, top_k, top_p, min_p,
                                          repetition_penalty, repetition_window, stats, True):
            idx = torch.cat([idx, next_id], dim=1)
            yield int(next_id[0, 0])
            if eos_token_id is not None and int(next_id[0, 0]) == eos_token_id:
                return
            if stop_fn is not None and stop_fn(idx):
                return

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple, device_type: str):
        # Weight decay on every matrix (dim >= 2: linear layers AND the tied embedding, as in nanoGPT);
        # none on norm gains and biases (dim < 2).
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # startswith, not ==: rt.device is "cuda:N" (not bare "cuda") for
        # every rank once torchrun sets more than one process running --
        # an exact-match check here would silently disable fused AdamW for
        # every multi-GPU run, which is exactly the kind of thing DDP
        # wiring should not quietly regress.
        use_fused = device_type.startswith("cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused if use_fused else None) \
            if use_fused else torch.optim.AdamW(groups, lr=lr, betas=betas)
