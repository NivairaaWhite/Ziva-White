# Ziva White

A local assistant built from scratch, created by **Mahasvin S S (Snow White)**. It owns a trainable
transformer (`ZivaGPT`) and read-only access to live web search. Every answer comes from this
project's own weights; the web only grounds an answer in current information.

What it does, beyond answering questions:

- **Learns from a folder you fill.** Drop files into `datasets/`; it trains on them automatically,
  in the background. No sample data ships, and nothing about it is hardcoded. See
  [Where to put your training data](#where-to-put-your-training-data) below.
- **Knows who made it.** "Who created you?" is answered from `configs/ziva.yaml`, not left to a
  small model to improvise.
- **Voice in and out, with wake words.** Fully offline when you use the local voice loop
  ("hey Ziva", "hey Snow White", or your own).
- **Controls the device.** Volume, brightness, media, apps, screenshots, lock/sleep, timers, optional
  smart-home. A fixed command grammar, never the language model, decides what runs.
- **Works in the background.** A supervised daemon keeps the watcher, timers and voice loop going,
  restarts what crashes, and can start at login.

Change history: [CHANGELOG.md](CHANGELOG.md). Everything else is in [docs/](docs/) (index at the end of this page).

## Get going

**One click:** Windows `start.bat`, macOS `start.command` (right-click → Open the first time),
Linux `./start.sh`. The first run creates a `.venv` and installs dependencies, then opens the web UI.

**Or by hand:**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                  # optional: web search, API key, smart home

python -m zivawhite.app datasets --init               # creates datasets/text and datasets/chat
#   ...copy your files in (next section)...
python -m zivawhite.app run                           # web UI + API + background jobs
```

(`pip install .` without `-e` also works, outside a checkout: the web UI, the bundled tokenizer and the default
settings are packaged with it, and it works in the folder you run it from. See [docs/INSTALL.md](docs/INSTALL.md).)

Until you've given it data there is no local model. Identity answers, device commands and timers
work immediately; questions are answered once the first training round finishes (see
`datasets` / `daemon status` for progress). A tiny model trained on a little data will be fluent and
often wrong: more data, more steps or a bigger `--tier` help, and the web route is there for facts.

## Where to put your training data

**Put it in the `datasets/` folder, next to this README.** That is the only place the model learns from;
nothing is bundled and nothing is downloaded for you.

```
datasets/
├── text/       prose that teaches the LANGUAGE   (books, articles, notes, docs, code, a saved site)
├── chat/       conversations that teach it to ANSWER   (question in, reply out)
├── _sources/   optional: topics.txt / urls.txt for the (off-by-default) web intake
└── _eval/      optional: held-out files you do not want trained on
```

```bash
python -m zivawhite.app datasets --init     # creates the folders above (safe to run any time)
python -m zivawhite.app datasets            # what it sees, what changed, what still needs training
```

* **What goes where.** Plain files: `.txt .md .rst .html .pdf` (and `.log`) in `text/`; conversations in `chat/` as
  `.jsonl`, `.json`, `.csv` or plain `User:/Assistant:` transcripts. `.jsonl/.json/.csv/.tsv` files are classified
  by what is *inside* them, wherever they sit. Put **text first, then chat data**: a model must read before it can
  be taught to answer. Formats, examples and limits: [datasets/README.md](datasets/README.md).
* **Different folder?** Set `paths.datasets` in `configs/ziva.local.yaml`, or `ZIVAWHITE_DATASETS=D:/my-data`
  in `.env`.
* **Then do nothing.** While it runs it notices new or changed files within ~30 s, rebuilds only the corpus that
  changed, and trains in the background (low priority, only while the machine is idle and plugged in). A corpus
  that is too small waits and says how much more it needs. The new model is loaded without a restart.
* **Removing a file** stops it being used in *future* training; it does not make a trained model forget it.
  To start from scratch, delete `runs/` and `data/datasets/`.

### Where everything ends up

| Folder | What is in it | Safe to delete? |
|---|---|---|
| `datasets/` | **your** training files | no: this is your data |
| `data/datasets/` | corpora and tokenized shards built from your files, plus what was already trained on | yes (rebuilt) |
| `runs/ziva/` | the base model (`checkpoint.pt`, `checkpoint_best.pt`) **and its `tokenizer/`** | deleting it forgets the model |
| `runs/ziva_behavior/` | the chat-tuned model (preferred by `ask` / `serve`) **and its `tokenizer/`** | rebuilt from `runs/ziva/` + `chat/` |
| `data/tokenizer/` | only if you trained your own vocabulary (below) | only before training with it |
| `configs/ziva.local.yaml` | your settings | no |

### Which tokenizer (vocabulary) does my model use?

A tokenizer turns text into the numbers a model reads, so it is **part of the model**, and a model is only
meaningful with the tokenizer it was trained with.

* A **new** model starts from the tokenizer bundled with the package (a general-English one, 12,000 tokens).
  That is a default for the first run, not a fixed setting.
* **Every training run saves the tokenizer it used** next to its checkpoints (`runs/<run>/tokenizer/`) and records
  its fingerprint inside them. After that, `ask`, `serve`, the web UI, `generate`, `eval`, `sft`, `dpo`, `export`
  and later training rounds all find *that* tokenizer by themselves: you never repeat `--tokenizer_dir` for a model
  you trained. Nothing is silently paired with the bundled one, and a wrong or missing tokenizer stops with a
  message that says what the model needs and where it looked. `python -m zivawhite.app inspect` shows which one a
  checkpoint uses.
* **To use your own vocabulary** (a language other than English, or very unusual text), before the first training run:

  ```bash
  python -m zivawhite.app tokenizer --vocab_size 8000     # learns from datasets/ -> data/tokenizer
  ```

  then put `autolearn: {tokenizer_dir: data/tokenizer}` in `configs/ziva.local.yaml`. It needs roughly 150
  characters of text per vocabulary entry. A model that already exists keeps its vocabulary; a different one means
  training a new model (start a new run, or delete `runs/`).

## Everything is configured in one file

`configs/ziva.yaml` is the single source of truth: the datasets location, training step counts and
throttling, the creator, wake words, voice engines, the device-control allow-list, the server's limits, the
daemon's paths and intervals. **Don't edit it.** Create `configs/ziva.local.yaml` (git-ignored) containing only
the keys you want to change; it is deep-merged over the defaults (lists are replaced, not appended).
`ZIVAWHITE_CONFIG` points at a different override file; `ZIVAWHITE_DATASETS` moves the datasets
folder. A typo (a bad port, an unknown job, a sensitivity of 5, a tokenizer folder that holds no tokenizer) is
reported at startup, not an hour into a background run.

```yaml
# configs/ziva.local.yaml: example
voice:
  enabled: true
  wake:
    words: ["hey computer", "computer", "hey ziva"]
device_control:
  allow_power_actions: true        # lets "shut down" through (still asks for a yes)
  apps:
    browser: {windows: "chrome", macos: "Google Chrome", linux: "firefox"}
autolearn:
  train_args: ["--tier", "small"]
```

## The creator

`identity:` in the config holds the assistant's name, its creator (**Mahasvin S S**, alias **Snow
White**) and the exact wording of the answers. Two things use it:

1. **Deterministic answers**: "who created you", "who made you", "who is your developer", "who are
   you", "who is Mahasvin S S" and similar are answered from the config before the model is consulted,
   because a small model will happily invent a creator. Phrasing is anchored on *you/your*, so "who made
   the Eiffel Tower" still goes to the model.
2. **Training examples**: the same facts, in the phrasings listed under `identity.training.questions`,
   are generated as chat examples and mixed into conversation fine-tuning (only when you have chat data).

Change the name in the config and both follow; no code edit, and the next training round re-learns it.

## Documentation

| Read | For |
|---|---|
| [datasets/README.md](datasets/README.md) | training-data formats, what is ignored, what happens after you drop a file |
| [docs/INSTALL.md](docs/INSTALL.md) | installing as a package, where files live, reproducible installs, CI |
| [docs/SECURITY.md](docs/SECURITY.md) | what protects the server and the device, and what does not |
| [docs/HTTP_API.md](docs/HTTP_API.md) | the HTTP API, limits, hardening |
| [docs/VOICE.md](docs/VOICE.md), [docs/DEVICE_CONTROL.md](docs/DEVICE_CONTROL.md), [docs/BACKGROUND.md](docs/BACKGROUND.md) | voice and wake words, controlling the machine, the daemon |
| [docs/MODEL.md](docs/MODEL.md) | the model, the full train → chat → export pipeline, multi-GPU |
| [docs/ROUTER.md](docs/ROUTER.md) | local model vs live web, web search setup, what the web may be used for |
| [docs/PUBLISHING.md](docs/PUBLISHING.md) | Hugging Face, Ollama, llama.cpp, vLLM |
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md), [docs/CAPABILITIES_MAP.md](docs/CAPABILITIES_MAP.md), [docs/EXPERIMENTAL.md](docs/EXPERIMENTAL.md) | an honest capability map; optional modules |
| [CHANGELOG.md](CHANGELOG.md) | what changed |

## Troubleshooting

Start with `python -m zivawhite.app inspect`: it prints which checkpoint will be loaded, its size and
step, and whether it is chat-tuned. Then `daemon status` and `datasets`.

- **"No local model yet."** Put training files in `datasets/` (`datasets` shows what it sees). If it
  says the data is too small, add more or lower `autolearn.min_train_tokens`.
- **"hello" returns book-style text.** The loaded model is a base model: it continues text. Add
  conversations under `datasets/chat/`; autolearn writes the chat-tuned model to `runs/ziva_behavior/`,
  which is preferred over `runs/ziva/`.
- **Training never starts.** `autolearn --status` shows what it thinks; the throttle waits while the
  machine is busy or on battery (`autolearn --once --no-throttle` overrides for one run).
- **Output is noise, or "trained with a vocab of N tokens but the tokenizer in use has M".** If M is 260, the
  `tokenizers` package isn't installed and the byte-level fallback was used: `pip install tokenizers`. Otherwise
  the model's own tokenizer could not be found. A model carries its tokenizer in `runs/<run>/tokenizer/`
  (training saves it there); `inspect` prints where it is looking. If that folder is gone, copy the tokenizer the
  model was trained with back to `runs/<run>/tokenizer/`, or pass `--tokenizer_dir <folder>`.
- **`tokenizer_dir ... holds no tokenizer`.** `autolearn.tokenizer_dir` points at a folder without `vocab.json` +
  `merges.txt`. Create one with `python -m zivawhite.app tokenizer`, or set it back to `null`.
- **HTTP 400 "Invalid Host header".** The server only answers to `localhost` and IP addresses unless told
  otherwise (a defence against DNS rebinding). To reach it by a name such as `my-pc.local`, set
  `ZIVAWHITE_ALLOWED_HOSTS=my-pc.local` in `.env`.
- **HTTP 503 "model is busy" / 429 / 413 / 400 "max_new_tokens ... over this server's limit".** One request
  generates at a time, a wrong API key repeated too often is locked out for a minute, bodies over 1 MB are
  refused, and no request may ask for more new tokens than one context window. All four are settings: see
  `limits:` in `configs/ziva.yaml` and [docs/HTTP_API.md](docs/HTTP_API.md).
- **"could not be loaded with the safe loader".** Checkpoints load with torch's `weights_only` loader so a shared
  `.pt` file cannot run code. An int8 file from an *old* `compress` pickled a whole module and is refused:
  re-run `compress` on the float checkpoint (or, only for a file you made yourself, set
  `ZIVAWHITE_ALLOW_PICKLE=1`).
- **The wake word is never recognised.** `voice --test-mic`, then add the spelling you see to
  `voice.wake.words`. If nothing is recognised at all, check `--devices` and the OS microphone permission.
- **"Voice unavailable".** The message says what is missing: `pip install -r requirements-voice.txt`,
  PortAudio on Linux, or the Vosk model folder.
- **Web queries return "No web results".** Set `SEARXNG_URL` (or `SEARCH_API_URL` + `SEARCH_API_KEY`) in
  `.env`; SearxNG needs its JSON format enabled. Until then auto-routed web queries fall back to the
  local model.
- **Resuming on different data.** The "best checkpoint" gate compares validation loss, which is only
  meaningful on the same data; autolearn passes `--reset_best` for you.

## Running it on Windows

Everything here is plain Python + PyTorch; nothing is Linux/macOS-specific. Use `.venv\Scripts\activate`
(cmd) or `.venv\Scripts\Activate.ps1` (PowerShell), not the Unix `source` form; `start.bat` does all of
that for you. Hardware detection queries Windows directly (`GlobalMemoryStatusEx` through `ctypes`) when
`psutil` isn't available. Autostart uses a Task Scheduler logon task (`daemon install`).

## Layout

```
launch.py               cross-platform launcher (create folders -> run -> open browser)
start.bat / start.sh / start.command   one-click entry points (start.command just runs start.sh)
configs/ziva.yaml       every setting, documented; override in configs/ziva.local.yaml
datasets/               YOUR training data goes here (README.md explains the formats)
zivawhite/
  config.py             layered config loading + validation
  identity.py           assistant name + creator; deterministic answers; training examples
  assistant.py          identity -> device control (with confirmation) -> model; event feed
  autolearn.py          the datasets-folder watcher/trainer (+ optional web intake)
  data/folder.py        scan, parse, clean, dedupe, build corpora, track what was trained
  data/tokenizer_train.py  learn a vocabulary from your datasets folder (`app tokenizer`)
  control/              intents.py (grammar), backends.py (Windows/macOS/Linux), controller.py, timers.py, smarthome.py
  voice/                vad.py, stt.py (Vosk/faster-whisper), tts.py, wake.py, audio.py, loop.py
  daemon.py             supervisor, start/stop/status/logs, autostart for systemd/launchd/Task Scheduler
  runtime.py            builds everything from the config; model hot-reload
  server.py, access.py  HTTP API + web UI; who may control the device over HTTP (Host check, server-issued sessions)
  webguard.py           request guards: Host allow-list (DNS rebinding), body-size limit, security headers + CSP
  model.py              ZivaGPT, the transformer (KV-cached generation)
  chat.py               chat format + reply stopping, shared by training data and inference
  tokenizer.py          BPE tokenizer: bundled default, saved with every run, found again by checkpoint (or the byte-level fallback)
  _env.py               which .env is loaded (one explicit place; never searches parent folders)
  errors.py             shared exceptions (ModelBusy)
  device.py             device + precision detection (CUDA/MPS/CPU, fp16 vs bf16). Model size is never inferred from it
  modelconfig.py        ModelConfig + presets + exact parameter counts + memory estimate (no torch needed)
  hf_compat.py          the exact ZivaGPT <-> LlamaForCausalLM weight/config mapping (proven in tests/test_hf_compat.py)
  export_hf.py          export a Hugging Face folder (safetensors, tokenizer, chat template, model card, Ollama Modelfile), --push
  import_hf.py          start from an open Llama-architecture model (e.g. SmolLM2) instead of from scratch
  sft.py, dpo.py        assistant-only fine-tuning; Direct Preference Optimization (+ self-play pair building)
  tools.py              bounded, read-only tool use (calculator, clock, web) + verifiable synthetic tool-use data
  evaluate.py, plan.py  perplexity / bits-per-byte / repetition metrics; compute planner (what can I afford?)
  data/hf_prepare.py    stream Hugging Face datasets into shards / chat JSONL / preference pairs / a trained tokenizer
  checkpoint.py         atomic, fsynced, crash-safe save/load; safe (weights_only) loading
  compress.py           int8 quantization, stored as plain tensors (no pickle)
  router.py             ZivaWhite: local model + web search/fetch
  train.py              training loop, same script every tier, torchrun-aware
  app.py                the CLI
  security.py           opt-in API key, rate limiting, wrong-key lock-out
  webui/index.html      browser UI: chat, mic, hands-free, spoken replies
  connectors/           web_search / web_fetch only
scripts/online_learning/  feedback -> drift/eval/promotion management loop (see docs/EXPERIMENTAL.md)
docs/                   everything that used to make this README 45 KB
tests/                  pytest suite
```
