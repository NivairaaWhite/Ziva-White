<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
The model, how it is trained, and how to scale that up.

# The model half — ZivaGPT

A decoder-only transformer with the Llama architecture, same at every size: RoPE position encoding (with optional
linear / NTK scaling to stretch the context), RMSNorm, SwiGLU MLP, **grouped-query attention** (a KV cache several
times smaller at inference), weight-tied input/output embeddings, `scaled_dot_product_attention` (PyTorch dispatches
this to FlashAttention-2 kernels on Ampere or newer GPUs and to a memory-efficient kernel on older ones such as the
T4; a safe manual fallback otherwise), activation checkpointing (`--grad_ckpt`), a KV-cached sampler with
temperature / top-k / top-p / min-p / repetition penalty, and a token streamer for servers.

**Model size is always your choice.** Nothing looks at your RAM or VRAM to decide how big your model may be.
Pick a named preset and/or set any of the numbers (`modelconfig.py` has the exact parameter counts):

| preset | body params | + tied embedding (32k vocab) | typical use |
|--------|-------------|------------------------------|-------------|
| tiny   | ~2M         | ~8M                          | experiments, CPUs |
| small  | ~23M        | ~39M                         | first real runs (your 25M MHA model is this size) |
| medium | ~265M       | ~298M                        | one 24-80GB GPU |
| large  | ~790M       | ~840M                        | one 80GB GPU, or several smaller ones (`torchrun`) |
| xl     | ~1.24B      | ~1.3B                        | multi-GPU, with `--grad_ckpt` |

```bash
python -m zivawhite.train --tier medium --max_steps 20000
python -m zivawhite.train --tier small --n_layer 12 --n_embd 768 --n_head 12 --n_kv_head 4 --block_size 2048
python -m zivawhite.app plan --preset medium --tokens 10b --gpu rtx4090 --n_gpus 2     # how long? (advice only)
python -m zivawhite.app plan --gpu a100 --hours 168                                    # what size can a week afford?
```

Hardware only decides *device and precision* (a T4 has no native bf16, so it gets fp16 + a loss scaler: that is
correctness, not a limit). If a memory estimate says a run probably will not fit, `train.py` prints a warning and
carries on; it never shrinks or refuses anything. Checkpoints trained before grouped-query attention existed (no
`n_kv_head`) load unchanged.

`train.py` handles the laptop case specifically: small physical
batch size + heavy gradient accumulation so an 8GB machine never
runs out of RAM, autocast (bf16/fp16) only turns on when CUDA is
actually available, and checkpoints are written atomically
(`checkpoint.py`: write to a temp file, `os.replace`) so a killed
process never corrupts your last saved state.

**Expected quality, honestly:** the "tiny" preset is a real transformer,
not a toy — but 2-6M parameters trained for a few hundred steps
produces garbled output. Public
benchmarks for this exact architecture family (nanoGPT-style, char
tokenization, tiny-shakespeare) show ~5000 steps (roughly 20-60
minutes on a laptop CPU) gets to a val loss around 1.6-1.7 and
recognizably-Shakespeare-shaped text. Point it at more/better data
and more steps for anything beyond a first experiment.

## From base model to assistant (what 0.4 adds)

```text
tokenizer -> pretrain -> sft -> (tool data) -> dpo -> eval -> export -> serve / publish
```

```bash
# 0. Know what you can afford
python -m zivawhite.app plan --gpu a100 --hours 72

# 1. A general vocabulary (the bundled 12k tokenizer is English books; code, numbers and other languages need more).
#    Pass it ONCE, to the first training run (step 2/3). That run saves it in runs/<run>/tokenizer/ and every
#    later step (sft, dpo, eval, export, ask, serve) finds it there by itself: no --tokenizer_dir again.
pip install -r requirements-publish.txt
python -m zivawhite.app data tokenizer --recipe fineweb --max_docs 300k --vocab_size 32000 --out tokenizers/gen32k

# 2. Pretrain on streamed web text (writes shards directly; no giant text file)
python -m zivawhite.app data text --recipe fineweb --tokenizer_dir tokenizers/gen32k --out_dir data/processed --max_tokens 5b
python -m zivawhite.app train --config configs/general_125m.yaml --data_dir data/processed --tokenizer_dir tokenizers/gen32k --out_dir runs/base

# 3. Teach it to chat (assistant-only loss, system/tool roles, epochs + early stopping)
python -m zivawhite.app data chat --recipe smol-smoltalk --out data/chat/smol.jsonl --max_rows 100k
python -m zivawhite.app tool-data --out data/chat/tool_use.jsonl --n 3000       # verifiable calculator/clock conversations
python -m zivawhite.app sft --data data/chat/*.jsonl --init_from runs/base/checkpoint.pt --out_dir runs/chat

# 4. Prefer better answers (second stage: needs a chat-tuned model)
python -m zivawhite.app data prefs --recipe ultrafeedback --out data/prefs/uf.jsonl --max_rows 20k
python -m zivawhite.app dpo --pairs data/prefs/uf.jsonl --init_from runs/chat/checkpoint.pt --sft_alpha 0.1 --out_dir runs/dpo

# 5. Measure, then use it
python -m zivawhite.app eval --checkpoint runs/dpo/checkpoint.pt --text data/heldout.txt --prompts prompts.txt
python -m zivawhite.app agent "What is 17% of 2340?" --checkpoint runs/dpo/checkpoint.pt
python -m zivawhite.app serve                        # includes an OpenAI-compatible /v1/chat/completions (streaming)

# 6. Publish: Hugging Face, Ollama, vLLM (see docs/PUBLISHING.md)
python -m zivawhite.app export --checkpoint runs/dpo/checkpoint.pt --out exports/my-model --verify
```

Or start from open weights instead of random numbers (`import-hf`, see `docs/CAPABILITIES.md`: it is the one step
that closes most of the gap to Ollama-class models). What each stage can and cannot do, honestly, is in
[`docs/CAPABILITIES.md`](CAPABILITIES.md); publishing is in [`docs/PUBLISHING.md`](PUBLISHING.md).

**Overfitting guards** (why a fine-tune used to memorise its data): `--max_epochs`, `--patience`, `--dropout`, and a
final-step validation, so a short run is measured too. `checkpoint_best.pt` only moves when validation loss improves,
and that is what `ask`/`serve` load.

**Safety of weights:** checkpoints load with `torch.load(weights_only=True)` first (no pickle code execution);
`export` writes safetensors; `ZIVAWHITE_ALLOW_PICKLE=0` forbids the pickle fallback entirely.

## Data pipeline and scaling to huge datasets

`data/dataset.py` reads training shards via `np.memmap` — the OS
pages in only the bytes touched by the block indices sampled *this
step*, so `train.bin` is never fully loaded into RAM. The same code,
unmodified, works whether that file is a few MB
or hundreds of gigabytes on a server. `data/folder.py` turns your datasets folder into corpora and
`data/ingest.py` tokenizes them into those shards (and still serves the legacy web queue and
`fetch_bulk`).

## Multi-GPU, for real — `torchrun` + `DistributedDataParallel`

This used to be a documented "concrete next step," honestly flagged as
not done. It's wired now, and — unlike most of this project's other
CPU-sandbox-verified claims — actually tested live as a real
distributed run, not just single-GPU-plus-code-review: a real 2-process
run over the `gloo` backend (this project's own build/test sandbox has
no GPU; `gloo` is PyTorch's CPU-compatible distributed backend, so this
exercises the real `torch.distributed` machinery, not a mock of it).
`device.py:resolve_runtime()` reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
directly from `torchrun`'s own environment — no `--ddp` flag to
remember, no guessing:

```bash
# single GPU (unchanged) or CPU laptop -- exactly as before
python -m zivawhite.train --tier medium

# one machine, 4 GPUs
torchrun --standalone --nproc_per_node=4 -m zivawhite.train --tier large

# two machines, 4 GPUs each (run on each machine, only --node_rank differs)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=<node0-ip> --master_port=29500 -m zivawhite.train --tier large
```

`--batch_size` is **per process** (standard PyTorch convention):
effective global batch = `batch_size * grad_accum * world_size`,
printed at startup so it's never ambiguous. Only rank 0 prints
progress, evaluates, and saves checkpoints (every rank's weights are
identical after each step's gradient all-reduce, so there's nothing to
gain from more than one rank doing it); checkpoints save with plain,
un-prefixed keys (`raw_model.state_dict()`, never the DDP wrapper's),
so a model trained under `torchrun` loads with **zero code changes**
anywhere else in the project — `ask`, `serve`, `generate` don't need to
know a checkpoint came from a distributed run. Verified directly: ran
the 2-process job above, confirmed the saved checkpoint has no
`module.` key prefix, then loaded and served it through the completely
ordinary single-process `ask` path.

Two real regressions this wiring would otherwise have introduced, found
and fixed along the way: `model.py:configure_optimizer` and
`data/dataset.py:get_batch` both checked `device == "cuda"` — an exact
string match that's true for single-GPU (`"cuda"`) but false for every
rank under `torchrun` (`"cuda:0"`, `"cuda:1"`, ...), which would have
silently disabled fused AdamW and pinned-memory transfers on every
multi-GPU run. Both now check `device.startswith("cuda")`.
