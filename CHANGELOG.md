# Changelog

Newest first. History before 0.3.0 is in [docs/CHANGELOG_EARLIER.md](docs/CHANGELOG_EARLIER.md) (moved there verbatim).

## Unreleased: models keep their own tokenizer; review fixes (security, robustness, packaging)

### A model finds the tokenizer it was trained with (the "12k is hardcoded" problem)
- **Cause.** Nothing recorded which tokenizer a model was trained with, and every later command loaded the bundled
  12,000-token English one unless `--tokenizer_dir` was repeated on each of them (`serve` and the web UI could not
  even be told). A model trained with another vocabulary refused to load. A wrong `--tokenizer_dir` path silently
  became the bundled tokenizer.
- Every training run (`train`, `sft`, `dpo`) now saves its tokenizer into `runs/<run>/tokenizer/` and stores a
  fingerprint of it in each checkpoint. `ask`, `serve`, the web UI, `generate`, `eval`, `sft`, `dpo`, `export` and
  autolearn's later rounds find that tokenizer by themselves (`tokenizer.resolve_tokenizer`): explicit flag, then the
  one saved with the checkpoint, then `autolearn.tokenizer_dir`, then the bundled one only if the checkpoint was
  trained with it. A candidate of the wrong size or fingerprint is never used; when nothing fits you get a message
  naming what the checkpoint needs and where it looked.
- A `--tokenizer_dir` that holds no tokenizer is now an error instead of a silent switch to the bundled one.
- Continuing or warm-starting a run keeps its vocabulary, and refuses a different one (even of the same size).
- New: `python -m zivawhite.app tokenizer --vocab_size N` learns a vocabulary from your datasets folder (before the
  first training run); `inspect` shows which tokenizer a checkpoint uses; `Config.problems()` reports an
  `autolearn.tokenizer_dir` that holds no tokenizer.
- Docs: README gained "Where to put your training data" (with a map of where everything ends up and which
  tokenizer a model uses); `datasets/README.md` and the tokenizer README say the same.

### Security
- **DNS rebinding** bypassed the device-control check (`Origin == Host` is true when the attacker controls both).
  Every request's `Host` must now be `localhost`, an IP address, or listed in `ZIVAWHITE_ALLOWED_HOSTS`; the
  same-origin shortcut requires it too. (`webguard.HostGuardMiddleware`, `access.host_ok`)
- **A pending "are you sure?" could be confirmed by an unauthorised caller.** `Assistant.handle` re-checks
  `allow_control` on the "yes"; HTTP session ids are now issued by the server (`access.SessionRegistry`) and
  namespaced (`http:<id>`), so a client can no longer pick `voice` or `cli`. The web UI keeps the id it is given.
- **Pickle is opt-in.** Checkpoints load with `weights_only=True` and are refused otherwise unless
  `ZIVAWHITE_ALLOW_PICKLE=1` (it defaulted to on). `continual.load_consolidation` no longer uses
  `weights_only=False`. `compress` stores int8 models as plain tensors (int8 weights + a scale each) instead of a
  pickled module; old pickled int8 files still load only with the opt-in.
- Failed logins are rate limited: the limiter now runs before the key check, and a wrong-key lock-out
  (`ZIVAWHITE_AUTH_FAIL_LIMIT`, default 10 failures/60 s) answers 429 before the key is even compared.
- `robots.txt` is fetched with the same SSRF protection as a page (redirects followed by hand, every hop checked,
  size capped). Fetched pages report the URL that actually answered, and the training-queue licence check uses
  that URL, not the one that was asked for.
- Search API errors can no longer carry the key or a URL (keys that providers require in the URL are scrubbed).
- Web UI: a hash-based `Content-Security-Policy` (no `unsafe-inline`), `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy` on every response.
- `load_dotenv` no longer walks up parent directories: one explicit `.env` (see `_env.py`).
- Custom TTS: a hung command is killed after `voice.tts.timeout_s` (it held the speaking lock forever), and spoken
  text beginning with `-` can no longer be read as an option by the TTS program.
- `preexec_fn` removed from the training subprocess (it can deadlock in a multithreaded process); the child's
  priority is lowered from the parent with `os.setpriority`.

### Server robustness
- `limits.max_new_tokens` (default one context window), `limits.max_body_bytes` (413), `limits.lock_timeout_s` (503 +
  `Retry-After`): one request can no longer hold the model forever and stall `/v1/health`. `/v1/chat/completions`
  streaming answers 503 as a real status when busy.
- The streaming lock covers only the generation: a slow or stalled client no longer blocks everyone (the model runs
  in a worker that releases it when it finishes, and stops when the client disconnects).
- `route` is `auto | local | web` (anything else is a 422; it used to mean "web").
- Local-answer confidence: text that is looping no longer counts as confident (mean top-1 probability is high for
  repetition); `router.confidence_floor`, `router.loop_threshold`, `router.freshness_years` and
  `router.freshness_patterns` are configurable, and the freshness year window follows the clock (it was a fixed
  2025-2039).

### Calculator
- `(-8)**0.5` no longer crashes with `TypeError`: non-real results give a readable error. One size rule
  (`bit_length()`, about 1000 digits) is used for everything, so `factorial()` accepts exactly the arguments whose
  result is allowed (0..449; it advertised 500) and `99999**1000` no longer leaks Python's "4300 digits" message.

### Search fallback
- One adapter per provider (`SEARCH_API_PROVIDER`: `brave`, `serpapi`, `google_cse`, `generic`); Brave's key goes in
  a header. The `generic` fallback now sends `Authorization: Bearer <key>` instead of a `key=` URL parameter.
  Bing's search API was shut down by Microsoft on 2025-08-11 and is no longer mentioned as an option; Google's
  Custom Search API is closed to new customers and ends on 2027-01-01.
- `.env.example` no longer ships `SEARXNG_URL` uncommented (health reported web search as configured with nothing
  running); the stale `app.py` pointer is fixed.

### Training and checkpoints
- Checkpoints are `fsync`ed before the rename (and the directory after it); the `.bak` is a hard link instead of a
  full copy of the checkpoint.
- Periodic checkpoints record the next step to run (they recorded the step just finished, so a resume repeated one).
- A non-finite loss or gradient skips the update, and three in a row stop the run **without** overwriting the last
  good `checkpoint.pt`. Grad-norm is logged; the loss is summed on the device (one host sync per step, not one per
  micro-step). DDP with a mixture-of-experts model sets `find_unused_parameters`.
- `model.py` validates with exceptions instead of `assert` (stripped by `python -O`).

### Packaging, docs, CI
- The wheel is now complete: web UI, bundled tokenizer and default settings are packaged (`configs/` ships as
  `zivawhite.defaults`), runtime dependencies are declared with upper bounds, and a private `ziva.local.yaml` can
  never be included. An installed copy works in the folder you run it from (`ZIVAWHITE_HOME` overrides).
- `.github/workflows/ci.yml`: tests on Ubuntu and Windows (Python 3.10/3.12) plus a wheel-contents check.
- README (45 KB) and CHANGELOG (43 KB) split: reference material moved to `docs/` (new: `INSTALL.md`, `SECURITY.md`).
- The `notebooks/` folder (a Colab GPU notebook) was removed.
- The bot `User-Agent` no longer advertises an invented `example.invalid` URL; set `ZIVAWHITE_BOT_CONTACT` to add yours.

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
