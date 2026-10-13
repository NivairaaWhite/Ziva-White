# Changelog

Newest first. Everything from "Earlier history" down was moved here verbatim from
the README, where it had grown to ~1,000 lines and buried the usage docs.

## Revision 0.4.0: a general-purpose pipeline: no hardware-picked size, GQA, SFT/DPO, tools, Hugging Face/Ollama export

**Important: what was and was not verified.** This revision was written in a sandbox where PyTorch,
FastAPI and `tokenizers` could not be installed (no package-index access). Everything that does not need them was run:
529 tests pass (338 before), the 10 test files that need torch fail to import exactly as they did before this revision
(same files, same reason), and 5 new tests that need torch/FastAPI/transformers are skipped (1 other skip pre-dates this). **Run `pytest -q` on a machine with torch before trusting the model-side changes** (`model.py`,
`train.py`, `checkpoint.py`, `sft.py`, `dpo.py`, `export_hf.py`, `import_hf.py`, the FastAPI endpoints). The numerics that
matter most (the Llama weight/rotary mapping, GQA grouping) are proven against independent numpy implementations in
`tests/test_hf_compat.py`, but the torch code itself has not been executed.

**Changed: model size is never inferred from hardware**
- `device.py` still detects device and precision (CUDA/MPS/CPU, fp16 vs bf16: correctness) but no longer picks a
  "tier" from VRAM/RAM. Size is a named preset and/or any explicit numbers: `--tier tiny|small|medium|large|xl`,
  `--n_layer --n_head --n_kv_head --n_embd --intermediate_size --block_size --rope_theta`. With no `--tier` the default
  starting point is `small`. If a rough memory estimate says a run will not fit, `train.py` prints a warning and continues.
- New torch-free `modelconfig.py` (`ModelConfig`, `PRESETS`, exact parameter counts, `estimate_train_memory_gb`).
  `from zivawhite.model import ModelConfig` still works. New preset `xl` (~1.3B).
- `benchmark_and_tune` takes an explicit `tier=`; its CUDA sync check now matches `cuda:N`, not only `cuda`.

**Added: model**
- **Grouped-query attention** (`n_kv_head`). With `n_kv_head == n_head` the state dict is laid out exactly as before, so
  every existing checkpoint loads unchanged. New presets use GQA (`--no_gqa` for plain multi-head).
- RoPE scaling (`rope_scaling` linear/ntk), activation checkpointing (`--grad_ckpt`), `intermediate_size` and `norm_eps`
  (needed to import real open models), `hidden_logits()`, `generate_stream()`, `min_p` and `repetition_penalty` sampling.

**Added: training and post-training**
- `train.py`: `--max_epochs`, `--patience` (DDP-safe early stop), `--dropout`, `--global_batch_tokens`, `--eval_iters`,
  eval under autocast, tokens/s in the log, and a final-step validation.
- `sft.py`: supervised fine-tuning that learns only the assistant's reply, one example per assistant turn (so the model
  learns to emit `<eos>`), system and tool roles kept, epochs, validation split, early stopping.
- `dpo.py`: Direct Preference Optimization with precomputed reference log-probabilities, `--sft_alpha`, and a self-play
  pair builder (`--from_sft`: real reply vs the model's own, the first round of SPIN) for when you have no preference data.
- `tools.py`: bounded read-only tool use (sandboxed AST calculator, clock, opt-in web), a strict call parser, the agent
  loop, and `generate_tool_conversations()` (verifiable synthetic tool-use data). `app.py agent`, `app.py tool-data`.
- `evaluate.py` (perplexity, bits per byte, assistant-only chat loss, repetition metrics), `plan.py` (compute planner).

**Added: ecosystem**
- `hf_compat.py`, `export_hf.py`, `import_hf.py`: export as a standard `LlamaForCausalLM` (safetensors, tokenizer, chat
  template, generated model card, Ollama `Modelfile`, `--verify`, `--push`), and import plain Llama-architecture models
  (e.g. SmolLM2) as a starting point. The interleaved-to-split-half rotary permutation is the non-obvious part.
- `server.py`: OpenAI-compatible `POST /v1/chat/completions` (with `stream: true` SSE) and `GET /v1/models`; the router
  gained `chat` / `chat_stream` and configurable sampling (`generation:` in `configs/ziva.yaml`).
- `checkpoint.py`: `torch.load(weights_only=True)` first with a loud, switch-off-able pickle fallback
  (`ZIVAWHITE_ALLOW_PICKLE=0`); `save_safetensors` / `.safetensors` checkpoints accepted everywhere.
- `data/hf_prepare.py`: recipes for FineWeb, open-phi textbooks, SmolTalk / smol-smoltalk, OpenHermes, NuminaMath,
  UltraFeedback, Glaive and xLAM; `inspect`; streaming shard writer; a wrong-schema guard; a provenance file per output;
  tokenizer training. `tokenizer.py`: `train_from_iterator`, `encode_batch`, extra special tokens, `tokenizer.json` loading.
- `chat.py`: `system` and `tool` roles, streaming decoder (`stream_text`), `HF_CHAT_TEMPLATE` (tested byte-for-byte against
  `build_chat_prompt`). `render_turn` used to turn any non-user role into an assistant turn silently; it now raises.
- Configs `general_125m|300m|1b.yaml`; docs `docs/CAPABILITIES.md`, `docs/PUBLISHING.md`.

**Fixed**
- **Chat fine-tuning could memorise its data and never notice.** The default fine-tune ran a fixed 100 steps regardless of
  dataset size (~3.3M tokens: ~30 passes over a 100K-token set), and validation only ran at `step % eval_interval == 0`,
  so a run shorter than `eval_interval` never measured validation loss and saved its final weights as "best".
  `autolearn` now passes `--max_epochs`, `--patience` and `--dropout` (config keys `chat_*`, `text_*`).
- Best-checkpoint snapshots recorded the 0-based step index; they now record steps completed.
- Checkpoints were loaded with `weights_only=False` (arbitrary code execution from an untrusted `.pt`).

**Not built (see `docs/CAPABILITIES.md`)**: FSDP/ZeRO, constrained (JSON) decoding, data deduplication/quality filtering,
RLHF/GRPO, safety tuning, MLA, multi-token prediction, multimodality, benchmark-harness integration, experiment tracking.

## Revision 0.3.0: learn from a folder, creator identity, voice, device control, background running

**Removed (all bundled sample / smoke-test material)**
- `zivawhite/data/samples/` (the 25-book corpus and the 88-example behaviour set),
  `data/pretraining_corpus.py`, `data/behavior.py`, `scripts/quickstart.py|sh`, the hardcoded Gutenberg
  bootstrap URL list, the hardcoded autolearn topic list, and the synthetic data generators in
  `scripts/online_learning/`. The CLI commands that depended on them (`bootstrap`, `pretrain`,
  `finetune-behavior`) are gone; their job is now done by the datasets folder.
- The pytest files named `test_*_smoke.py` are ordinary unit tests and were kept.

**Learning from a folder you fill (`datasets/`)**
- `autolearn` now watches `datasets/` (`paths.datasets`) instead of a fixed topic list. `data/folder.py` reads
  txt/md/rst/html/pdf and jsonl/json/csv/tsv in the common conversation shapes (messages, ShareGPT, Alpaca,
  prompt/response, question/answer), plus "User:/Assistant:" transcripts; cleans, de-duplicates, and builds a text
  corpus and a conversation corpus with a manifest. Change detection is by corpus hash, so only the kind that
  changed is retrained, touching a file retrains nothing, and a crashed run is retried. Files still being copied
  are waited for; unreadable files are reported and skipped; too-small corpora wait and say how much more is needed.
- Training runs as a managed low-priority child process, throttled while the machine is busy or on battery, and
  forwards its output to the log. `train.py --extra_steps` trains N more steps without loading the checkpoint first.
  The new model is hot-loaded.
- Web intake is optional and writes allow-listed pages into `datasets/text/web/` as ordinary files with provenance;
  topics and URLs come from your files. The license allow-list is not configurable.

**Creator identity**
- `identity:` in `configs/ziva.yaml` (creator **Mahasvin S S**, alias **Snow White**). Questions about the assistant
  or its creator are answered deterministically from it, and the same facts are generated as training examples.

**Voice and wake words** (`zivawhite/voice/`): energy VAD with adaptive noise floor, offline STT (Vosk or
faster-whisper), OS/pyttsx3/custom TTS, wake-word matching with a phonetic fallback, a conversation state machine
(wake, acknowledge, follow-up window, stop phrases), and the web UI's browser mic, hands-free mode and spoken replies.

**Device control** (`zivawhite/control/`): fixed command grammar, Windows/macOS/Linux backends, app allow-list,
spoken confirmation for lock/sleep/restart/shutdown (enforced in the controller), power actions off by default,
persistent timers and reminders, optional Home Assistant bridge, and HTTP access rules (`access.py`).

**Background running**: a supervisor with restart/backoff/give-up, a heartbeat status file, rotated logs,
`daemon start|stop|restart|status|logs`, and login autostart for systemd, launchd and Task Scheduler.

**Other**
- All settings moved to `configs/ziva.yaml` with a git-ignored local override and startup validation.
- Web UI: `innerHTML` replaced with `textContent` throughout (server-supplied sources and reasons were inserted unescaped).
- `/v1/assistant`, `/v1/assistant/info`, `/v1/assistant/events`, `/v1/voice/stop`, `/v1/datasets`, `/v1/status`.
- `import zivawhite`, the config, the dataset tooling, voice and device control no longer import torch.
- Fixed while testing: a malformed PDF raising a library-specific error escaped the per-file guard; malformed
  URLs like `javascript:alert(1)` made URL validation raise instead of reject.

## Revision: serving, data pipeline and cleanup pass

Found by reading the code (no GPU, no torch in the review sandbox, so the pure-Python
parts were exercised and the torch parts were checked by arithmetic and by new tests
you can run: `python -m pytest`).

**Why "hello" returned book text**
- Inference never used the chat format the model was fine-tuned on. New `zivawhite/chat.py`
  is the single place the format is defined; the training-data renderer and every inference
  path (router, `/v1/generate`, `infer.py`) now use it. Chat-tuned checkpoints are prompted
  as `<|user|>\nhello\n<|assistant|>\n` and stop when the reply ends.
- `finetune-behavior` overwrote the base `checkpoint.pt` and could never replace
  `checkpoint_best.pt` (its val loss was measured on different data than the stored best).
  It now warm-starts from the base (`train.py --init_from`), writes to `runs/ziva_behavior/`,
  and the best gate is reset whenever data changes (`--reset_best`; implied for fresh and
  warm-started runs). `ask`/`serve` look in `runs/ziva_behavior/` first.
- `ask` / `serve` without `--checkpoint` loaded no model at all. They now find one.
- `resolve_serving_checkpoint` swapped in `checkpoint_best.pt` for ANY filename in the folder;
  now only for `checkpoint.pt`.

**Generation**
- KV cache (`ZivaGPT.forward_cached`), O(N) instead of O(N^2) token-forwards per reply.
- `generate` takes `stop_fn`, returns confidence stats, and `generate_text` returns only the
  new text (no echoed prompt, no leaked `<bos>`; special tokens are registered on load).
- Prompts longer than the context window are shortened instead of crashing (HTTP 500).
  Retrieved web text is trimmed to fit before the writer prompt is built.
- Router confidence is the mean top-1 probability over the tokens actually generated, not a
  separate forward pass over the whole prompt.
- Freshness routing no longer fires on "electric current" or "score a goal".
- If auto-routing picks web but retrieval returns nothing (no search backend configured),
  the local model answers instead of a dead end.
- Generation is serialized with a lock (the server runs handlers in a threadpool).
- int8 checkpoints from `compress` can now be loaded by `ask`/`serve`.

**Data pipeline**
- `<eos>` goes after each document, not each line (books are hard-wrapped, so it used to be
  one `<eos>` per ~13 tokens). Documents are marked with `<|doc...|>` lines.
- Validation is a set of whole random documents, not the tail of the file.
- Tokens stream to disk instead of living in a Python list (RAM no longer scales with corpus).
- Behavior data uses a smaller validation floor so the 88-example set isn't 26% validation.
- `ingest` / `autolearn` write to `data/processed_web/`, no longer over the base shards.

**Training**
- `--init_from`, `--reset_best`, `--schedule constant`, `--lr_horizon`, `--config file.yaml`
  (the YAMLs in `configs/` were never read before).
- Resuming or warm-starting rebuilds the model from the checkpoint's own config, so a Colab
  "small" checkpoint can be resumed on a laptop whose auto-tier is "tiny".
- `keep_training.py` / `autolearn.py` no longer restart the cosine schedule every chunk.
- bf16 is only chosen on compute capability >= 8.0 (a T4 emulates it, slower than fp16).
- MoE `experts_w2` now gets the scaled residual init like every other output projection.
- EWC Fisher uses per-sequence gradients (batch-mean gradients underestimated it ~batch-size
  times). Re-run `consolidate` and retune `--ewc_lambda` if you have old files.

**Security**
- `serve` binds 127.0.0.1 by default; CORS defaults to localhost origins only.
- `X-Forwarded-For` is ignored unless `ZIVAWHITE_TRUST_PROXY=1`; API-key check is constant-time;
  the rate limiter forgets idle clients.
- Page fetches refuse localhost / private / metadata addresses and re-check every redirect
  (SSRF); robots.txt is fetched with the bot's User-Agent and a timeout, per RFC 9309
  (missing = allowed, server error / unreachable = refused); downloads are size-capped.
- Allow-list matching uses the URL's hostname, not `netloc`.

**Removed (dead code)**
- Unused: `router.ConnectorResult`, `nas.copy`, `sources.Callable`, `nas._random_candidate`'s
  `block_size`, `latent.decode_indices`, `latent.codebook_usage`, `compress.load_quantized`
  (superseded by `checkpoint.load_model`).
- `online_learning_loop.py --judge llm` and `--route api` (needed connectors that no longer exist).
- `start.command` was a byte-for-byte copy of `start.sh`; it is now a 1-line wrapper.

**Not changed, on purpose**
- NAS / VQ / MoE / EWC / compress were not moved into an `experimental/` folder: they have
  tests, CLI commands and imports, and moving them risks breaking things for no behavior gain.
- A 25M-parameter model trained on a few million tokens will still be a weak assistant.
  See README "Troubleshooting".

## Earlier history (moved from README)

## If you hit "typed hello, got nothing" before this round

That was a real bug, reproduced and fixed here, not a hardware
limitation. `data/ingest.py:build_shards()` computed the train/val
split as a flat `0.0005` fraction of the corpus — fine for a
multi-million-token corpus, but the bundled offline bootstrap sample
is only ~156K tokens, so that fraction handed validation ~78 tokens
against a `block_size` of 256. `train.py`'s first evaluation pass
(step 240 by default) tried to pull a 256-token batch out of a
78-token file, raised `ValueError`, and the process died before ever
writing a checkpoint — which is why the server then came up with no
model loaded and "hello" got nothing back. Reproduced those exact
numbers (155,695 train / 78 val) firsthand, then fixed `build_shards`
to floor the validation split at `max(block_size, 2000)` tokens
regardless of corpus size, and added a non-fatal skip in
`train.py:evaluate()` as a second line of defense. Also real, found
and fixed the same session: `.env` was never actually loaded anywhere
in the codebase (no `python-dotenv`, no `load_dotenv()` call) — the
usual "add a `.env` file and restart" fix genuinely wouldn't have
worked before this round; and the Anthropic connector defaulted to an
old model string and always sent a non-default `temperature`, which
Claude Sonnet 5 now hard-rejects with an HTTP 400. All three are fixed
and re-verified below.

Also fixed this round, a second and separate reason output could look
like noise even once training ran without crashing: see "If output
still looked like noise even after training ran cleanly" below.

## If output still looked like noise even after training ran cleanly

A different bug from the crash above, real and separate: every entry
point that didn't get an explicit `--tokenizer_dir` silently fell back
to a byte-level tokenizer (`ByteLevelTokenizer`, one token per raw
byte, vocab_size=260) — `tokenizer.py` supported real BPE the whole
time, but nothing actually trained or pointed at one by default,
anywhere, including `cmd_bootstrap`'s call to `build_shards`. That's
not incorrect, just a brutal handicap on a CPU training budget: the
model has to spend its limited capacity re-deriving "which byte
sequences form a word" from scratch, on every run, before it can even
start learning language structure. Fixed by training a real BPE
tokenizer (vocab_size=12000) once, offline, on the full bundled
25-book corpus, and shipping it at `data/tokenizer/` as the default
`load_tokenizer()` falls back to. Measured effect on the exact bundled
bootstrap sample: 155,695 byte-level tokens for the same text BPE
covers in 47,889 — roughly 3.25x fewer tokens for identical content,
so a fixed `block_size` now covers proportionally more real text. See
`data/tokenizer/README.md` for the regeneration command and why not to
retrain over it once you have real checkpoints.

## What was actually run to verify this build

- `pytest tests/` — 6/6 passed (model builds + forward pass at every
  tier, generation extends the sequence correctly, weight tying
  holds, router picks `web` for a freshness query and `api` when no
  local model is loaded, explicit route override works).
- Fetched a real public-domain sample, tokenized it (1,154,816
  train tokens), trained 60 real optimizer steps on CPU — loss
  5.6060 → 4.9349 → 4.4808, checkpoint saved and reloaded correctly.
- Generated from that checkpoint and asked the router a question
  through it end-to-end (both `--route local` and the freshness-auto
  path, the latter correctly falling through to "no search backend
  configured" instead of crashing).
- Queued a fetched Wikipedia-domain page through the router,
  confirmed it landed in `data/queue/*.jsonl`, then ran
  `ingest --commit` and confirmed the resulting `manifest.jsonl`
  correctly recorded `source: wikipedia, license: CC BY-SA 4.0`.
- Started the FastAPI server and confirmed `/v1/health` and
  `/v1/generate` respond correctly.
- VQ bottleneck: real forward+backward pass, confirmed gradients reach
  the codebook and code-usage is inspectable.
- MoE layer: real forward+backward with 4 experts/top-2 routing,
  confirmed gradients reach every expert and the load-balancing loss
  computes.
- Tool calling: a full multi-turn loop against a mocked connector —
  model requests `calculator(6*7)`, tool executes, result is fed back,
  model returns the final answer — resolved correctly in 2 turns.
- NAS: a real 2-3 trial search, each candidate actually trained for
  10-15 steps and evaluated on held-out data, correctly ranked by
  measured val loss.
- EWC: computed real Fisher information from actual gradients, saved
  and reloaded a consolidation file, confirmed the penalty is exactly
  0 at the snapshot point and provably positive after parameters
  drift — then ran a real `consolidate` → `train --resume --ewc` CLI
  pipeline end to end.
- Autotuner: measured real tokens/sec on this machine at several batch
  sizes and picked the empirically fastest one via `--autotune`.
- 14/14 pytest tests passing across all of the above.
- Windows/launcher round: found and fixed a real bug (`/v1/ask` was
  returning a 422 because `AskBody`/`GenBody` were defined *inside*
  `cmd_serve()` — with `from __future__ import annotations` active,
  FastAPI couldn't resolve those local-scope model classes; moved
  them to module level, reproduced the failure and the fix in
  isolation before applying it, then re-verified `/v1/health`,
  `/v1/ask` on both an explicit route and the freshness-auto path,
  and `/v1/generate`, all against a real running server).
- `launch.py` run directly: correctly skipped bootstrap when a
  checkpoint already existed, bound to `127.0.0.1`, waited for
  `/v1/health` before reporting ready; run a second time while the
  first instance was still up, correctly moved to the next free port
  (8001) instead of crashing.
- `python -m zivawhite.app bootstrap` run with no network access
  available to it beyond localhost: tokenized the bundled sample and
  trained a real checkpoint, confirming the offline first-run path
  actually works offline.
- `scripts/quickstart.py`: dependency install, network fetch,
  tokenization, training, and generation steps each run and confirmed
  individually (a full 500-step run just doesn't fit this sandbox's
  single CPU core inside one tool call — the training loop itself was
  already separately verified extensively above).

## This round: the reported bug, plus a full merge and re-verification

- **Reproduced the exact reported crash** firsthand:
  `python -m zivawhite.app bootstrap` tokenized the bundled sample
  into 155,695 train / 78 val tokens — the exact numbers reported —
  confirming `build_shards`'s flat `val_fraction` as the root cause,
  not a hardware or PyTorch environment issue.
- **Fixed and re-verified**: same bootstrap run now produces 2,049 val
  tokens (safely above `block_size`), and a `train.py --eval_interval 4`
  run through the old crash point now logs a clean
  `[eval] step 4 val_loss ...` instead of raising `ValueError`.
- **`.env` loading**: confirmed nothing in the codebase called
  `load_dotenv()` before this round (grepped the whole tree). After
  adding it to `zivawhite/__init__.py`, wrote a throwaway `.env`
  with a fake key in a clean directory and confirmed
  `AnthropicConnector().api_key` actually picks it up.
- **Merged and tested `pretrain`**: ran `pretrain --stage stage0` and
  `--stage stage1` end to end — corpus cleaning + re-tokenization
  through ZivaGPT's own tokenizer, real training steps, checkpoint
  saved. Confirmed loss actually decreases (5.60 → 4.46 over 60 steps
  on the 4-book stage1 corpus).
- **Merged and tested `finetune-behavior`**: ran it against a bootstrap
  checkpoint — 88 examples correctly rendered to `<|user|>`/`<|assistant|>`
  text, tokenized (30,696 train tokens), resumed training from the
  existing checkpoint's step count, completed without error.
- **Merged and tested the online-learning kit**
  (`scripts/online_learning/`): fixed a stale path in
  `build_replay_buffer.py` left over from the original standalone zip's
  folder layout, then ran the full pipeline — `build_replay_buffer.py`,
  `validate_dataset.py`, and `online_learning_loop.py` both with no
  `--checkpoint` (confirmed identical heuristic-only behavior to the
  original stub) and with a real checkpoint via `--checkpoint`
  (confirmed real per-example generation + scoring, producing genuine,
  low-but-real `eval_score`s consistent with a lightly-trained tiny
  model, not a fabricated number).
- **Full end-to-end HTTP round trip with a real trained checkpoint**:
  started `serve`, confirmed `/v1/health` reports
  `local_model_loaded: true`, then `POST /v1/ask {"query": "hello"}`
  returned real generated text (not empty, not a crash) — word-shaped
  fragments ("the", "and", "he", "was") rather than pure noise, showing
  the additional training steps are doing something real even at this
  scale. Also confirmed a freshness-triggering query
  ("what is the weather today") correctly routes to `"web"` and falls
  through cleanly to the local model when no search backend is
  configured, instead of crashing.
- Anthropic connector: confirmed the new default model string and that
  `temperature` is no longer sent unless explicitly requested.
- Full `pytest tests/` suite: still 14/14 passing after every change
  above.

## This round: open up the API layer, add image input, map the AGI checklist

- **The actual "not limited to a fixed API list" fix**: confirmed
  `_decide`/`_run_api`/`_run_local`/`_run_web` all hardcoded
  `("anthropic", "openai")` before this round (grepped every occurrence).
  Replaced with `_api_connector_names()`, a dynamic, configurable list.
  Wrote a regression test (`test_router_routes_to_a_connector_outside_the_old_hardcoded_pair`)
  proving a connector registered under a name outside that old pair now
  actually gets used — it would have been silently unreachable before.
- **`GenericConnector` tested against a real HTTP server, not just
  mocked**: spun up a local `http.server` instance returning a
  Cohere-shaped response (`{"message": {"content": [{"text": ...}]}}`),
  wired it in with `CUSTOM_API_URL`/`CUSTOM_API_BODY_TEMPLATE`/
  `CUSTOM_API_RESPONSE_PATH` env vars only — no code — and confirmed
  both the connector directly and a full `ZivaWhite()` instance (zero
  connectors passed in, built entirely from `.env`) found it and routed
  a real query to it correctly, auth header included.
- **`GeminiConnector`**: request/response shape verified against
  Google's current documented `generateContent` format (fetched their
  docs directly rather than relying on training-data memory, since
  Gemini's API surface changed at least once during this project
  already). Unit-tested with mocked HTTP: role translation
  (`assistant` → `model`), text extraction, and safety-block handling
  (empty `candidates` + a `blockReason` surfaces as a clean error, not
  a silent empty success).
- **Multimodal image input**: added `load_image_b64()` (path/bytes/dict
  → normalized base64) and wired it through Anthropic, OpenAI-compatible,
  and Gemini connectors' `call()` methods, plus `ask(image=...)`,
  `zivawhite.app ask --image`, and `POST /v1/ask`'s `image_base64`
  field. Confirmed `ask(image=...)` bypasses local/web routing
  entirely (there's no vision-capable local model) and reaches a
  connector's `call()` with the image payload intact; confirmed the
  CLI path fails with a clear, specific message
  ("no image-capable API connector available...") rather than an empty
  response or a crash when no vision-capable connector is configured.
- Caught and fixed a bug introduced mid-round: `ask()`'s image branch
  was unconditionally overwriting `_run_api`'s specific failure reason
  with a generic one, masking exactly the message a user would need to
  self-diagnose a missing API key. Fixed before it shipped.
- 16 new tests added (`tests/test_connectors.py`), full suite: 30/30
  passing.

## This round: production-hardened server, image upload in the web UI

- **`security.py`**: bearer-token auth and per-client rate limiting,
  both opt-in/off-by-default. Live-tested against a running server, not
  just unit-tested: confirmed a request with no token gets a clean 401
  once `ZIVAWHITE_SERVER_API_KEY` is set, the correct token gets
  through, and exactly 3 requests succeed then a 4th gets 429 +
  `Retry-After` with `ZIVAWHITE_RATE_LIMIT=3/60` — the limiter's
  actual counting, not just that it returns *a* 429 eventually.
- **Structured errors end to end**: confirmed an empty `query` and an
  out-of-range `max_new_tokens` both come back as clean 422s with the
  specific pydantic validation detail, a 404 comes back as
  `{"error": "Not Found"}` instead of Starlette's default shape, and
  (by code review of the handler, since forcing a genuine 500 needs a
  real unhandled exception) any other exception is caught and logged
  server-side rather than reaching the client as a stack trace.
- **Web UI**: added image attach/drag-drop/paste with a thumbnail
  preview and a remove button, an API-key settings panel that
  auto-appears when `/v1/version` reports `auth_required: true`, and
  HTTP-status-aware error messages (401 vs. a validation error vs. an
  unreachable server now say different, useful things instead of one
  generic failure message). Verified: extracted the `<script>` block
  and syntax-checked it with `node --check`, verified every HTML tag is
  balanced with a small parser, then actually served the page and
  confirmed a full image round trip through `/v1/ask` end to end (200
  bytes in, a clean "no image-capable connector configured" response
  out, matching a real environment with no API keys set).
- 11 new tests (`tests/test_security.py`), full suite: 41/41 passing.

## This round: renamed to Ziva White, real BPE by default, CPU efficiency, "only gets better" checkpoints, compression, and an experimental byte-patching prototype

- **Renamed** `MavelyaWhite` → **Ziva White** throughout: package
  (`zivawhite/`), classes (`ZivaGPT`, `ZivaWhite`), env vars
  (`ZIVAWHITE_*`), CLI, README, web UI. Full suite re-verified passing
  after the rename, not just after the feature work.
- **Real BPE tokenizer by default** — the actual root cause of output
  looking like noise even on a well-behaved training run (a separate,
  second bug from the crash documented earlier in this README): see
  "If output still looked like noise even after training ran cleanly"
  above. Trained a real vocab_size=12000 BPE tokenizer on the full
  bundled 25-book corpus and made it the default everywhere via
  `tokenizer.py:load_tokenizer`, zero flags required. Verified live:
  re-ran bootstrap and confirmed 47,889 tokens where the old byte-level
  path produced 155,695 for the identical file; a fresh model's initial
  loss came out at 9.40, matching `ln(vocab_size)=ln(12000)=9.39` almost
  exactly, confirming the 12000-vocab is actually what's wired in, not
  just documented as if it were.
- **CPU efficiency, measured not assumed**: `device.py` now runs a
  sub-second matmul benchmark at startup comparing fp32 against
  bfloat16 autocast on the actual CPU in front of it, and only switches
  if bf16 measurably wins — same empirical-autotuning philosophy
  `benchmark_and_tune` already used, extended to dtype selection.
  `train.py` wires the result into real training (bf16 needs no
  GradScaler, unlike fp16, so none is added for it). Also explicitly
  sets `torch.set_num_threads()` to all visible cores rather than
  trusting ambient environment defaults. Measured on this build's own
  (CPU-constrained) test machine: bf16 autocast was faster and was
  used automatically; a real training run completed with a sane,
  decreasing loss curve, not a hang or a crash.
- **The served model can only get better, not just "keep training"**:
  `train.py` now tracks `checkpoint_best.pt`, rewritten only when
  validation loss actually improves (inference-only — no optimizer
  state, so it's also smaller and faster to load than the rolling
  checkpoint). `checkpoint.py:resolve_serving_checkpoint` makes
  ask/serve/generate prefer it automatically. Verified live: trained to
  step 32, confirmed `checkpoint.pt` sat at step 32 while
  `checkpoint_best.pt` correctly held the step-30 weights (the best
  validation loss actually seen), and confirmed serving resolves to the
  best one, not the latest one. `scripts/keep_training.py` runs
  training in indefinite, resumable, interrupt-safe chunks so this can
  just be left running.
- **Compression**: `zivawhite compress` dynamically int8-quantizes a
  checkpoint's Linear layers for CPU inference (PyTorch's dynamic
  quantization is CPU-only anyway — a genuine fit, not a borrowed
  GPU-serving trick). Verified live: a real checkpoint went from
  48,938,895 bytes to 13,377,336 bytes (3.7x smaller, combining
  quantization with dropping optimizer state), generation got faster,
  and val perplexity moved from 7045.5 to 7080.6 — a small, honestly
  measured cost, not a claimed free lunch. The quantized model
  round-trips correctly through save/load and still supports
  `.generate()` identically.
- **`bytepatch.py`** — a new, small, genuinely-working experimental
  prototype of entropy-based dynamic byte patching, in response to
  research notes provided this round. See the dedicated section above
  ("Entropy-based dynamic byte patching") for what it is, what it
  isn't, and what was actually run to verify it (`tests/test_bytepatch.py`,
  5 new tests). Deliberately isolated from `model.py` — off by default,
  zero risk to the reliable path.
- 46/46 tests passing (41 previous + 5 new for `bytepatch.py`).

## This round: a real audit (asked to find what's broken/missing and "make it an AGI")

On "make it an AGI": no, and the reasoning hasn't changed since the
existing "What the AGI capability list maps to here" section above was
written against an earlier version of this same ask — nothing in the
newly re-attached research notes changes that. Building toward general
intelligence isn't a feature this or any other codebase can add; it's
an open problem nobody has solved at any scale. What follows instead
is the real, concrete thing that *was* asked for and *is* answerable:
an actual audit, using a real run the person did on their own machine
(Windows, 8 cores, 23.9GB RAM) as evidence, not just re-reading the
code in the abstract.

The good news the log/screenshot confirmed first: the BPE fix from last
round works on a real machine, not just in a sandbox. Bootstrap
tokenized the identical 47,889/2,049 train/val split, training ran
300 real steps (loss 9.50 → 3.52, val_loss 4.84), `checkpoint_best.pt`
fired correctly ("[best] new best val_loss 4.8392"), and the actual
generated text — real English words, character names, punctuation,
Shakespeare-plausible fragments — is a different category of output
entirely from the byte-garbage this project started this thread with.

Three real bugs found and fixed from that same evidence:

- **`resolve_runtime()` was re-running its hardware benchmark on every
  call, including every `/v1/health` poll.** Last round's CPU
  dtype-autodetection benchmark is cheap once, but `app.py`'s
  `/v1/health` handler calls `resolve_runtime()` fresh on *every
  request*, and `launch.py` polls that endpoint every 0.4s while
  waiting for the server — which is exactly why the log shows health
  checks taking 5-6+ seconds each and `launch.py` printing "server
  didn't respond in time" even though the server was actually fine.
  Fixed with `@functools.lru_cache` on `resolve_runtime` — verified
  directly: first call ~0.02-0.25s, every call after that ~0.00s.
- **The local model was being used as a RAG answer-writer, and asked
  direct questions, with no awareness that it had never been
  instruction-tuned.** This is the actual reason "hello" produced a
  Shakespeare continuation instead of a greeting response, and why the
  "run" query's log shows the model being fed a full
  "Using ONLY the information below, answer concisely..." template and
  producing more Shakespeare instead of following it — `bootstrap`
  only pretrains on raw corpus text; nothing about that teaches a model
  to recognize "hello" as a greeting or follow an instruction. Not a
  crash, not incorrect code — a real category mismatch that was
  invisible to the user. Fixed: `finetune-behavior` now stamps
  `behavior_tuned: true` into the checkpoint (`train.py
  --behavior_tuned`); the router reads it (`router.py:load_local_model`)
  and (a) says so honestly in `reason` when asking a base model
  directly, and (b) stops using an untuned local model as the RAG
  writer — returning raw search context instead of instruction-shaped
  garbage. `cmd_bootstrap` now also prints a direct pointer to
  `finetune-behavior` when it finishes. 6 new tests
  (`tests/test_router_smoke.py`) cover both the honesty and the
  RAG-writer gating, verified against both a tuned and an untuned
  in-memory model.
- **`ask()` was silently discarding the backend's own reason and
  replacing it with just the routing rationale.** This is a pre-existing
  bug, not one introduced last round — the reason strings that reached
  the person, in the screenshot and every prior round's testing, were
  only ever `_decide()`'s short routing phrases ("local model
  available, no freshness signal", "explicit override"). Anything more
  specific a backend computed — a confidence-based API fallback, and
  now the base-model honesty above — was computed and then thrown
  away before it ever reached the caller, directly undercutting the
  web UI's own "nothing is hidden" claim. Fixed by combining them
  (`"{routing rationale} -> {what actually happened}"`) instead of
  overwriting; a route change (e.g. local's confidence-fallback to
  API) correctly keeps only the backend's reason, since the original
  routing rationale no longer applies to what actually ran.

What's genuinely working, unprompted and re-verified this round: BPE
tokenization end to end, bootstrap/pretrain/finetune-behavior/serve/ask
CLI paths, `checkpoint_best.pt` tracking, `compress`, device
auto-detection, the web UI's routing display (now actually showing
real reasons, per the fix above), and the full 51-test suite. What's
still explicitly a small-scale/smoke-test-only piece, unchanged and
already honestly labeled as such before this round: the 88-example
behavior dataset (`data/samples/behavior/README.md`) — running
`finetune-behavior` will make "hello" get an actual greeting-shaped
reply instead of Shakespeare, but on a dataset this size, "coherent
assistant" is not a realistic bar to expect it to clear; that needs a
real (much larger) instruction dataset, which is a data problem, not a
code bug.

51/51 tests passing (46 previous + 5 new router tests this round —
see `tests/test_router_smoke.py` for the added base/tuned-model cases).

## This round: removed the AI-API wrapper, removed artificial model limits, removed the bundled smoke-test data, wired real multi-GPU training, automated self-training, benchmarked and then removed an underperforming prototype

Four asks arrived across this round, and they're reported together
honestly rather than smoothed into a single tidy narrative, because
two of them were declined (with reasons) and that matters as much as
what shipped:

**Wiring requested and delivered as asked:**

- **Multi-GPU, for real.** `train.py`/`device.py` now read
  `torchrun`'s `RANK`/`LOCAL_RANK`/`WORLD_SIZE` directly and wrap the
  model in `DistributedDataParallel` — see "Multi-GPU, for real"
  above. Verified with an actual 2-process run over `gloo` (this
  project's sandbox has no GPU), not just single-GPU testing plus code
  review of the untested multi-GPU branch. Found and fixed two real
  regressions the device-string change would otherwise have silently
  introduced (exact `== "cuda"` checks in `configure_optimizer`/
  `get_batch` that are false for `"cuda:0"`).
- **Byte-patching vs. BPE, actually benchmarked.** A previous round's
  prototype (`bytepatch.py`) was explicitly left "not yet benchmarked
  against BPE" as its own honestly-stated open question. This round
  ran that benchmark for real, matched parameter count, bits-per-byte
  as the metric (see "Removed: the byte-patching prototype" above for
  the actual numbers and methodology).
- **"Remove the smoke test data, remove limits on the model, remove
  the wrapper feature, make training faster, connect it to the
  internet to train itself."** All five, literally: the bundled
  offline sample corpus and the `stage0` curriculum stage are gone;
  `AskBody`/`GenBody`'s artificial length/token caps are gone (see
  "Removed: the artificial limits on the model"); the AI-API wrapper
  (Anthropic/OpenAI/Gemini/generic connectors, `tools.py`) is gone
  (see "Removed: the AI-API wrapper"); `cudnn.benchmark`, an opt-in
  `torch.compile` flag, and the multi-GPU wiring above address
  training speed; `autolearn.py` is the "trains itself on the
  internet" loop, automated end to end rather than requiring a human
  to run `ask` then `ingest --commit` by hand.

**Declined, with reasons, rather than executed as literally asked:**

- **"Merge micrograd and tinygrad into something better than
  PyTorch."** Measured directly from the two repositories provided:
  micrograd is 154 lines total (a scalar-only autograd engine,
  explicitly scoped by its own author as a teaching tool); tinygrad is
  243,037 lines (a real multi-year compiler/runtime project that
  doesn't itself claim to beat PyTorch on performance). "Better than
  PyTorch" isn't a gap either closes, or that a rewrite closes in one
  sitting, for anyone. What was actually built instead:
  `nanotensor.py`, a small array-valued autograd engine combining one
  real idea from each (micrograd's graph-and-backward-closure
  mechanism, generalized from scalars to broadcasting-aware numpy
  tensors; a tinygrad-style "measure a few real kernel candidates on
  this machine and keep the fastest" matmul search), verified against
  PyTorch's own autograd on every op. Then a follow-up request to
  "merge everything" — now including the actual PyTorch source
  (`pytorch-main.zip`) — was also declined: that zip is itself
  ~189,000 lines *without* the git submodules (oneDNN, FBGEMM, cutlass,
  NCCL, ...) a real PyTorch build requires, so it doesn't even compile
  standalone, and more fundamentally, PyTorch isn't a missing
  ingredient to merge in — every file in this project already runs on
  it. `nanotensor.py` was kept as the tested, isolated module it was,
  not wired anywhere near `model.py`/`train.py`.
- **Both prototypes removed on request, once evidence said they
  weren't earning their place.** Asked directly: "if these prototypes
  don't improve my model, remove them." They didn't (byte-patching
  measurably underperformed BPE at matched parameters; `nanotensor.py`
  was never connected to the model at all — it couldn't be without
  losing GPU support, DDP, attention, and everything else `ZivaGPT`
  actually needs), so both are gone: `bytepatch.py`, `nanotensor.py`,
  `scripts/benchmark_tokenizers.py`, `train.py --arch bytepatch` and
  every test file for any of it. This section, and "Removed: the
  byte-patching prototype" above, are what's left as the honest record
  of having tried — not the code itself.

**Everything else touched this round for consistency, once the
wrapper/smoke-data removal made them stale:** `launch.py` and
`scripts/quickstart.sh`/`.py` (bootstrap is now a live internet fetch,
no offline fallback left); `.env.example` (no more wrapper API keys);
`zivawhite/webui/index.html` (no image-attach UI, no `api`/`tools`
routes in the dropdown); `data/pretraining_corpus.py` and
`data/samples/pretraining_corpus/SOURCES.md` (`tinyshakespeare.txt`
and `stage0` removed, book counts corrected).

Verified this round: a real single-process training run end to end
(loss 9.48 → 4.86 over 40 steps, `--resume` and `--compile` both
exercised); a real 2-process `torchrun` run producing a checkpoint
that loads with zero code changes through the ordinary single-process
`ask` path; a live HTTP server round-trip against `/v1/health` and
`/v1/ask` confirming the removed image fields and removed length caps
actually behave as documented; the byte-patch-vs-BPE benchmark's real
numbers; `nanotensor.py`'s gradients matched against PyTorch's own
autograd before it was removed. 30/30 tests passing in the final
state (down from a peak of 44 with both prototypes' test files still
present — the drop is entirely removed test files for removed code,
not a coverage regression on anything that shipped).
