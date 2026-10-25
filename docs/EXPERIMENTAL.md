<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
Optional modules and the online-learning kit.

# Experimental modules — real alternatives to the AGI blueprint

A later round of this build worked from an uploaded "AI-Generating
Algorithm" concept doc (self-rewriting code, an emergent language
between AI nodes that bypasses human language, hardware that
self-migrates to future chips). None of that is implementable as
described, and some of it — code that redeploys itself without a
human reviewing it, or a representation deliberately designed to be
opaque to the people running it — isn't something worth building even
as a toy. What follows are the real, working, bounded research
techniques that correspond to each idea, all of them actually run and
tested (numbers below are from real runs, not projections):

| Blueprint idea | What actually exists for this | File |
|---|---|---|
| "generates its own internal language" | discrete latent bottleneck (VQ-VAE) — a fully inspectable codebook, opt-in, off by default | `latent.py` |
| "Architect Node rewrites the network live" | bounded Neural Architecture Search: trains N candidates for a short budget, ranks them, writes a report — never touches a deployed model | `nas.py` |
| "protects knowledge via lifelong learning" | Elastic Weight Consolidation (Kirkpatrick et al. 2017) — real Fisher-information-weighted penalty against catastrophic forgetting | `continual.py` |
| "self-profiling JIT tests the silicon" | empirical autotuning — measures actual tokens/sec on this machine at a few batch sizes, picks the fastest measured one | `device.py:benchmark_and_tune` |
| (not in the blueprint — added because it's the real architecture behind several current frontier models) | sparse Mixture-of-Experts feed-forward layer with top-k routing and a load-balancing loss, opt-in | `model.py:MoESwiGLU` |

Two rows that used to be here — "take capabilities from other AI
models" (`tools.py`'s function-calling loop) and entropy-based dynamic
byte patching (`bytepatch.py`) — are gone; see "Removed: the AI-API
wrapper" above and "Removed: the byte-patching prototype" below for
why each one specifically didn't earn a place in the current build.

Try what's left:

```bash
# architecture search -- report only, deploys nothing
python -m zivawhite.app nas --data_dir data/processed --trials 6

# empirical hardware autotuning
python -m zivawhite.app tune

# continual learning: consolidate what a checkpoint learned, then
# keep training on new data without stomping on it
python -m zivawhite.app consolidate --checkpoint runs/ziva/checkpoint.pt --data_dir data/task_a --out runs/task_a.pt
python -m zivawhite.app train --resume --ewc runs/task_a.pt --ewc_lambda 1000
```

For the VQ bottleneck and MoE layer, set the flag on `ModelConfig`
before building the model (see `tests/test_experimental_architecture.py`
for working examples of both, including combining them):

```python
cfg = ModelConfig.preset("tiny", vocab_size=12000)  # matches the bundled tokenizer
cfg.use_moe, cfg.moe_num_experts, cfg.moe_top_k = True, 4, 2
cfg.use_vq_bottleneck, cfg.vq_num_codes = True, 512
```

### Removed: the byte-patching prototype

An earlier round built `bytepatch.py`: a small, isolated, opt-in
prototype of entropy-based dynamic byte patching (the published,
narrower piece of a larger research-notes upload — Meta's Byte Latent
Transformer), kept deliberately out of `model.py`/`ZivaGPT` so it
carried zero risk to the default BPE path. It was real and it worked —
verified end to end (contiguous patch spans covering 100% of input,
bit-identical batched-vs-standalone padding, loss dropping 5.66 → 0.13
over 60 steps on a short passage) — but it was never benchmarked
against BPE at matched parameter count, which its own module docstring
named as the actual open question.

That benchmark was then run for real
(`scripts/benchmark_tokenizers.py`, since removed along with the rest
of this): at a matched ~4M total parameters, 300 training steps, one
CPU core, bits-per-byte (the metric that's actually comparable between
a subword tokenizer and a byte-level patcher — see that script's own
docstring for why raw loss-per-token isn't) —

| | bits/byte (lower is better) | throughput |
|---|---|---|
| BPE (this project's default) | **2.373** | 11.0 KB(text)/s |
| entropy-based byte patching | 4.094 | 12.4 KB(text)/s |

BPE won decisively — roughly 70% better compression of the actual
text at the same parameter budget, in a short run where neither model
is close to converged. Byte-patching also gained a real generation
capability during this round (`train.py --arch bytepatch`, trainable
end to end with checkpointing, though still no `ask`/serving support —
autoregressive generation for entropy-based patches needs to decide
patch boundaries from bytes that don't exist yet, a genuinely open
design problem, not unwired glue code), but with the measured result
showing it underperforming the existing default rather than improving
on it, both the prototype and its training path have been removed
(`bytepatch.py`, `scripts/benchmark_tokenizers.py`, the
`--arch bytepatch` flag, and their tests). This paragraph is the
record of what was tried and measured, not the prototype itself — a
short, single-book, single-CPU-core, 300-step run is real evidence
against keeping it, not proof dynamic patching can never work here;
revisiting it would mean rerunning that same benchmark script's
methodology at a larger scale, not resurrecting the removed code
unchanged.

## Managing updates over time — `scripts/online_learning/`

A self-contained kit (feedback log -> drift monitoring -> gated
promotion decisions -> version registry) for the actual online-learning
management problem: safely folding a stream of new interactions into
the model without it silently regressing or forgetting. Batch
acceptance/edit/reject stats and the drift score are real, computed
metrics. The one piece that started as a deliberate stub —
`run_eval_suite()`, scoring real model output against a fixed eval
set — is now wired to a real model:

```bash
cd scripts/online_learning/scripts
python3 build_replay_buffer.py     # rehearsal sample from YOUR conversations in datasets/
python3 online_learning_loop.py --checkpoint ../../../runs/ziva/checkpoint.pt
```

The kit ships with no data. Your `feedback_log.jsonl` and `eval_benchmark.jsonl` go in
`datasets/_eval/` (ignored by training) or the kit's own `data/` folder; formats are in its README.

Scores with a free, offline, lexical-overlap heuristic (a crude "did it
get worse" signal; the old `--judge llm` option was removed along with the
API connectors). Run
with no `--checkpoint` at all and it behaves exactly like the original
stub (`eval_score` stays `null`, every decision is logged as
`heuristic_pending_real_eval`) — see
`scripts/online_learning/README.md` for the full picture, including
why a lightly-trained checkpoint scoring low is the eval working
correctly, not a bug.
