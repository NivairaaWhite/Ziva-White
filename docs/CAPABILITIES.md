# What it takes to work like Llama, Qwen and other Ollama-class assistants

An honest map of every capability those assistants have, what each one actually requires, and where ZivaWhite
stands on it. Statuses mean exactly this:

| tag | meaning |
|---|---|
| **TESTED** | built, and its logic is covered by tests that were run (pure Python, or a numpy reference implementation) |
| **WRITTEN, RUN IT** | built with tests, but the tests need PyTorch / FastAPI / a GPU. Run `pytest -q` on your machine before trusting it |
| **PARTIAL** | some of it is here; the missing part is named |
| **NOT BUILT** | not here, with the reason and the way to get it |

## 1. The thing code cannot change: scale

Quality in language models is driven, in this order, by **how many tokens the model trained on, how many parameters
it has, and how good the data is.** The software around it matters far less. Reported figures for models people run
in Ollama:

| model | parameters | pretraining tokens |
|---|---|---|
| SmolLM2-135M / 360M / 1.7B | 0.135 / 0.36 / 1.7 B | 2 T / 4 T / 11 T (its paper puts the 1.7B run at roughly $250,000 of GPU time) |
| Qwen2.5-1.5B | 1.5 B | 18 T |
| Llama-3.2-1B | 1 B | 9 T (distilled from a larger model) |
| **your ZivaWhite `small`** | 0.025 B | well under 0.01 T |

What a GPU budget buys (`python -m zivawhite.app plan ...`, 35% efficiency assumed, so edit `--mfu` for your card):

| budget | compute-optimal model | tokens |
|---|---|---|
| 1 x A100, 1 day | ~250 M | ~4 B |
| 1 x A100, 1 week | ~660 M | ~13 B |
| 8 x A100, 1 month | ~4 B | ~80 B |

Even those are 100-1000x fewer tokens than the models above. Two honest conclusions:

1. **A model trained from scratch here will be a good *learning and research* model, not a replacement for Qwen or
   Llama.** It can be fluent and useful within what it saw; it will not know much and will often be wrong.
2. **The way to get Ollama-class behaviour on one machine is to start from open weights**, then use everything
   ZivaWhite adds on top (chat tuning, tool use, DPO, evaluation, serving). That is what `import-hf` is for.

```bash
huggingface-cli download HuggingFaceTB/SmolLM2-360M --local-dir models/smollm2-360m      # Apache-2.0, plain Llama architecture
python -m zivawhite.app import-hf --src models/smollm2-360m --out runs/base
python -m zivawhite.app sft --data data/chat/*.jsonl --init_from runs/base/checkpoint.pt --tokenizer_dir runs/base/tokenizer --out_dir runs/chat
```

`import-hf` accepts plain Llama-architecture checkpoints (SmolLM2, Llama-2-style models). It refuses, with the reason, Qwen2
(attention biases), Llama-3.1/3.2 ("llama3" rope scaling), untied embeddings (Llama-3-8B) and vocabularies over 65,536
(Llama-3's 128k). The imported weights keep their original license, which then governs what you publish. This path is
**WRITTEN, RUN IT**: the weight mapping is proven exact against a numpy reimplementation of both architectures
(`tests/test_hf_compat.py`), but I have not run it on a real downloaded checkpoint.

You choose: *from scratch* (you own every weight; quality is bounded by your compute) or *from open weights* (far more
capable; the starting weights are someone else's). Both are supported end to end.

## 2. Capability by capability

### Foundation

| capability | what it needs | status |
|---|---|---|
| **A modern architecture** (RoPE, RMSNorm, SwiGLU, grouped-query attention, flash-style attention) | code | **TESTED**: all present. GQA is new; equivalence with the Llama formulation is proven in numpy for MHA, GQA, MQA, linear and NTK rope. The torch attention code itself needs **WRITTEN, RUN IT** |
| **Scale**: 100M-3B parameters | compute + data | **PARTIAL**: any size is configurable (`tiny`..`xl` or custom); `plan` tells you the cost. Multi-GPU is DDP (`torchrun`); **FSDP / ZeRO sharding is NOT BUILT** (needed beyond ~1B on 24 GB cards; `--grad_ckpt` buys a lot of room first) |
| **A general tokenizer** (code, numbers, other languages) | train a BPE on broad text | **TESTED** (the data path): `data tokenizer`. The bundled 12k vocabulary is English books and will bottleneck a general model. Vocabulary is capped at 65,536 by the uint16 shard format |
| **Large clean pretraining data** | web-scale text, deduplicated and quality-filtered | **PARTIAL**: streaming FineWeb into shards is built (`data text`, memory-flat). **Deduplication, quality classifiers and data-mixing schedules are NOT BUILT**; these are a large part of why SmolLM2 works |
| **Code ability** | code in the pretraining mix (e.g. The Stack, which is gated and license-sensitive) | **NOT BUILT**: no code recipe is included. Add one in `data/hf_prepare.py` once you have decided the licensing |
| **Multilingual / Indic** | multilingual tokenizer + data | **NOT BUILT**: the pipeline is language-agnostic, no data is bundled |
| **Long context** (32K-128K) | long-sequence training; memory for attention and KV cache | **PARTIAL**: rope `linear`/`ntk` scaling + `--block_size` for long fine-tuning, and GQA shrinks the KV cache. 128K is not realistic at this scale; ring/context-parallel attention is **NOT BUILT** |

### Assistant behaviour

| capability | what it needs | status |
|---|---|---|
| **Instruction following / chat** | SFT on conversations, assistant-only loss | **TESTED** (data path) / **WRITTEN, RUN IT** (trainer): `sft` keeps system/user/assistant/tool roles, teaches `<eos>`, epochs + validation + early stopping. Data: `data chat --recipe smol-smoltalk` (built for sub-1B models) |
| **System-prompt adherence** | a system role in the format, and data that uses it | **TESTED**: `<|system|>` is its own role in training, serving and the exported chat template (the old dataset-folder path folds system text into the user turn). Adherence quality still depends on data and scale |
| **Preference alignment (DPO)** | pairs of better/worse answers; a chat-tuned model first | **TESTED** (pair handling) / **WRITTEN, RUN IT** (trainer + loss). `dpo --pairs` for existing data, or `dpo --from_sft` builds self-play pairs (real reply vs the model's own, the SPIN method) when you have none |
| **RLHF with a reward model, RLVR / GRPO** | reward model or programmatic verifier, plus fast batched generation | **NOT BUILT**. DPO is the practical route at this scale. RLVR is a natural next step because calculator/math answers are verifiable, but it needs high-throughput generation (vLLM on the exported model) |
| **Safety / refusals** | safety data, red-teaming, evaluation | **NOT BUILT**. Nothing here makes a model safe. Treat any published model as unaligned and say so (the generated model card does) |
| **Reasoning / chain of thought** | CoT data, ideally RL | **PARTIAL**: `data chat --recipe numinamath-cot`. Switchable "thinking modes" and RL-trained reasoning are **NOT BUILT**, and small models gain little from CoT |
| **Tool use** | a call protocol, a safe executor, training data | **TESTED**: `tools.py` (sandboxed calculator that refuses code injection and resource bombs, clock, optional web; bounded loop) and `tool-data` (verifiable synthetic conversations). Reliability at small scale is limited. **Grammar/JSON-constrained decoding, which would help a lot, is NOT BUILT**; parallel calls are NOT BUILT. Glaive/xLAM converters are best-effort (schemas unverified: `inspect` first) |
| **Memory / retrieval (RAG)** | embeddings + a vector store | **PARTIAL**: the router grounds answers in live web results; no vector store |
| **Vision / audio** | multimodal training | **NOT BUILT**: the model is text-only (the repo's voice layer is speech-to-text / text-to-speech around it) |

### Serving and ecosystem: "works like Ollama"

The single most useful fact: **ZivaGPT is a Llama-architecture model, so a trained checkpoint can be exported as a
standard `LlamaForCausalLM`.** That one export unlocks the whole ecosystem without rebuilding any of it.

| capability | route | status |
|---|---|---|
| Hugging Face `transformers`, PEFT/LoRA, TRL | `export` -> `AutoModelForCausalLM.from_pretrained(folder)` | the mapping is **TESTED** (numpy oracle); the load test is **WRITTEN, RUN IT** (`export --verify`) |
| **Ollama** | the exported folder includes a `Modelfile`; Ollama imports Llama-architecture safetensors folders (`FROM .`) or GGUF | **PARTIAL**: Modelfile written from Ollama's documented syntax, not run against Ollama here. Known snag below |
| llama.cpp / GGUF / 4-bit quantization | `convert_hf_to_gguf.py` on the export | **PARTIAL**: same snag below |
| vLLM / TGI (high-throughput serving, batching) | serve the exported folder | **NOT BUILT here, deliberately**: use vLLM; this repo serves one generation at a time |
| OpenAI-compatible API with streaming | `serve` -> `/v1/chat/completions`, `/v1/models` | **TESTED** (streaming decoder, router, locks) / **WRITTEN, RUN IT** (FastAPI endpoint tests) |
| Quantization | int8 via `compress`; 4-bit via GGUF after export | **PARTIAL** |
| Speculative decoding, paged KV cache, continuous batching | serving engines | **NOT BUILT**: vLLM provides them |

**The known snag for a from-scratch model:** llama.cpp's converter recognises a BPE tokenizer by hashing its output,
so a tokenizer you trained yourself triggers `BPE pre-tokenizer was not recognized`. The converter prints a hash;
mapping it to `gpt-2` (your tokenizer uses the plain GPT-2 byte-level regex) in `get_vocab_base_pre()` is the
documented one-line fix. I confirmed the mechanism in llama.cpp's documentation and issue threads but did not run
it. A model imported from SmolLM2 keeps SmolLM2's recognised tokenizer, so it avoids the problem.

### Evaluation and training operations

| capability | status |
|---|---|
| Perplexity and **bits per byte** (tokenizer-independent) on held-out text; assistant-only chat loss; repetition metrics | **TESTED** (metrics) / **WRITTEN, RUN IT** (model-backed): `eval` |
| Standard benchmarks (MMLU, HellaSwag, GSM8K, ...) | **NOT BUILT**: run EleutherAI's `lm-evaluation-harness` on the exported folder. Small models score near chance on most |
| Experiment tracking (Weights & Biases / TensorBoard) | **NOT BUILT**: logs go to stdout |
| Overfitting guards (`--max_epochs`, `--patience`, `--dropout`, final-step validation) | **WRITTEN, RUN IT**: this fixed the cause of the chat-tuning overfitting (below) |
| Safe checkpoints (no pickle execution), safetensors | **WRITTEN, RUN IT** |

## 3. Why chat fine-tuning overfit

I can see the project's default pipeline, not the settings of your own Colab run, so this is the most likely cause
rather than a diagnosis of your run. The default chat fine-tune (`autolearn.chat_train_steps`) runs a **fixed 100 steps
regardless of dataset size**. On the
`small` tier a step is about 33,000 tokens (16 x 4 sequences x 512 tokens), so the run read about 3.3 million tokens in
total: roughly 30 passes over a 100K-token chat set, and hundreds over anything smaller. Worse, validation only ran when
`step % eval_interval == 0`, and a 100-step run never reached step 100, so **no validation loss was ever measured** and
the final (memorised) weights were saved as "best". Fixed in `train.py`: an epoch cap, early stopping, a final-step
evaluation and dropout; `autolearn` passes them by default.

## 4. About the checklists you pasted

- **You do not need `trl`, `peft`, DeepSpeed, vLLM, bitsandbytes, Triton or custom CUDA** to do what this project
  does. It has its own SFT/DPO trainers, and PyTorch's `scaled_dot_product_attention` already dispatches to
  FlashAttention-2 kernels on Ampere or newer GPUs (a T4 gets a memory-efficient kernel instead). Those tools matter in
  specific situations: **vLLM** for serving throughput (use it on the exported model), **FSDP/DeepSpeed** for >1B
  models on small GPUs, **LoRA (`peft`)** if you fine-tune a model too big to train fully.
- **"DPO instead of supervised fine-tuning"** is backwards. DPO is a second stage that refines a model that can
  already follow the chat format; on a base model it has nothing to refine.
- **"Switch to bfloat16 on a T4"**: a T4 has no native bf16, so fp16 + a loss scaler (already what `device.py`
  does) is correct. bf16 is for Ampere and newer.
- **Dataset list**: `smol-smoltalk` (not the full SmolTalk) is the set built for sub-1B models; Hugging Face used it
  and then DPO on UltraFeedback for SmolLM2-Instruct, the same stage order as here. The Nectar "400k prompts" figure
  I could not confirm, so no recipe is included. Many of these sets were generated by commercial LLMs; see the
  license notes in `docs/PUBLISHING.md` before you publish anything trained on them.

## 5. The architectures in the research you shared, and what to do with each

Checked against the papers' own abstracts: *Attention Is All You Need* (arXiv 1706.03762), *The Llama 3 Herd of Models*
(2407.21783), *DeepSeek-V3 Technical Report* (2412.19437: 671B total / 37B active parameters, Multi-head Latent Attention,
DeepSeekMoE, auxiliary-loss-free load balancing, multi-token prediction, 14.8T tokens, 2.788M H800 GPU-hours), and *Gemini 1.5*
(2403.05530: a sparse mixture-of-experts model with context windows of millions of tokens). Two caveats on the summary: "the
Transformer replaced sequential with parallel processing" is true of *training*; **generation is still one token at a time**,
which is why a KV cache (and GQA, which shrinks it) matters. And **GPT-4 being a mixture-of-experts is third-party reporting,
not something OpenAI has confirmed**; its GPT-4 technical report withholds architecture details. GPT-4o is described by
OpenAI as one model trained end to end across text, vision and audio.

| idea | what it buys | here |
|---|---|---|
| Self-attention Transformer | the foundation | **TESTED / WRITTEN, RUN IT**: it is the model |
| Dense decoder, Llama-3 style (RMSNorm, RoPE, SwiGLU, GQA) | simple, stable, the most quality per byte of memory at small scale | **Built; this is exactly ZivaWhite's architecture**, and why the export to `LlamaForCausalLM` is exact |
| Mixture-of-Experts (DeepSeek-V3, Gemini 1.5) | more total parameters for the same per-token compute | **PARTIAL**: an experimental `use_moe` layer exists (top-k routing + load-balancing loss). Not exportable (the Llama format has no experts; Mixtral would be the target). **Recommendation: stay dense.** All experts must sit in memory, so at the sizes you can train on one machine you pay the memory of a big model for little of the benefit; MoE earns its keep at DeepSeek's scale |
| Multi-head Latent Attention (MLA) | compresses the KV cache into a low-rank latent: smaller than GQA | **NOT BUILT, on purpose.** GQA already gives a 2-4x smaller cache and exports to a standard Llama that vLLM/Ollama run; MLA needs DeepSeek's own architecture in every serving tool, so you would lose that. Worth it only for long-context serving at scale |
| Auxiliary-loss-free expert balancing | stable MoE training | **NOT BUILT** (MoE-only) |
| Multi-token prediction | a denser training signal; enables speculative decoding | **NOT BUILT**. A contained addition (extra prediction heads) if you want it later |
| FP8 training | ~2x throughput on H100-class GPUs | **NOT BUILT** (needs that hardware) |
| Million-token context (Gemini 1.5) | whole codebases / hours of video in a prompt | **NOT FEASIBLE** here: needs purpose-built infrastructure. `linear`/`ntk` rope scaling reaches a few multiples of the trained length at best |
| Native multimodality (GPT-4o, Gemini) | one network over text, images and audio | **NOT BUILT**: text-only |

**Putting the scale in perspective.** 2.788M GPU-hours is about 318 years on a single GPU. At an illustrative $2 per
GPU-hour that is roughly $5.6M, which the paper calls cost-effective, and which is still thousands of times a personal
budget. Its other lesson applies to you: most of the gains came from data, training recipe and post-training (SFT +
RL), not from exotic attention. For one machine, the highest-value moves remain the ones in section 6.

## 6. Suggested order of work

1. `plan` for your GPU, so the model size and token count are decisions, not guesses.
2. Decide: from scratch (125M-300M, `configs/general_*.yaml`) or import SmolLM2-360M / 1.7B.
3. Train a tokenizer (from scratch only), pretrain / import, then `sft` on `smol-smoltalk` + `tool-data`.
4. `dpo` (with `--sft_alpha 0.1`), then `eval` before and after each stage so you can see whether it helped.
5. `export --verify`, test in `transformers`, then Ollama, then publish with an honest card.
6. Next real capabilities, in order of payoff: **constrained (JSON-grammar) decoding for tools, data deduplication and
   quality filtering, FSDP for larger models, RLVR on verifiable tasks.**
