# Ziva White

A local assistant built from scratch, created by **Mahasvin S S (Snow White)**. It owns a trainable
transformer (`ZivaGPT`) and read-only access to live web search. Every answer comes from this
project's own weights; the web only grounds an answer in current information.

What it does, beyond answering questions:

- **Learns from a folder you fill.** Drop files into `datasets/`; it trains on them automatically,
  in the background. No sample data ships, and nothing about it is hardcoded: see
  [datasets/README.md](datasets/README.md).
- **Knows who made it.** "Who created you?" is answered from `configs/ziva.yaml`, not left to a
  small model to improvise.
- **Voice in and out, with wake words.** Fully offline when you use the local voice loop
  ("hey Ziva", "hey Snow White", or your own).
- **Controls the device.** Volume, brightness, media, apps, screenshots, lock/sleep, timers, optional
  smart-home. A fixed command grammar, never the language model, decides what runs.
- **Works in the background.** A supervised daemon keeps the watcher, timers and voice loop going,
  restarts what crashes, and can start at login.

Change history lives in [CHANGELOG.md](CHANGELOG.md).

## Get going

**One click:** Windows `start.bat`, macOS `start.command` (right-click → Open the first time),
Linux `./start.sh`. The first run creates a `.venv` and installs dependencies, then opens the web UI.

**Or by hand:**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                  # optional: web search, API key, smart home

python -m zivawhite.app datasets --init               # creates datasets/text and datasets/chat
#   ...copy your files in (see datasets/README.md)...
python -m zivawhite.app run                           # web UI + API + background jobs
```

Until you've given it data there is no local model. Identity answers, device commands and timers
work immediately; questions are answered once the first training round finishes (see
`datasets` / `daemon status` for progress). A tiny model trained on a little data will be fluent and
often wrong: more data, more steps or a bigger `--tier` help, and the web route is there for facts.

## Everything is configured in one file

`configs/ziva.yaml` is the single source of truth: the datasets location, training step counts and
throttling, the creator, wake words, voice engines, the device-control allow-list, the daemon's
paths and intervals. **Don't edit it.** Create `configs/ziva.local.yaml` (git-ignored) containing only
the keys you want to change; it is deep-merged over the defaults (lists are replaced, not appended).
`ZIVAWHITE_CONFIG` points at a different override file; `ZIVAWHITE_DATASETS` moves the datasets
folder. A typo (a bad port, an unknown job, a sensitivity of 5) is reported at startup, not an hour
into a background run.

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

## Learning from the datasets folder

```bash
python -m zivawhite.app datasets            # what's there, what changed, what still needs training
python -m zivawhite.app autolearn --once    # one scan/train round now
python -m zivawhite.app autolearn           # keep watching (what the daemon does)
```

Text teaches the language, conversations teach it to answer. Text trains the base model; conversations
fine-tune a copy of it, so your chat data never overwrites the base. Only the corpus that changed is
retrained; too-small corpora wait and say how much more they need; training runs as a low-priority
child process only while the machine is idle and on mains power; a crash is retried next round; the new
model is hot-loaded. Details, formats and limits: [datasets/README.md](datasets/README.md).
Removing a file does not make a trained model forget it.

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

## Voice, wake words

Pipeline: microphone → voice-activity detection → **offline** speech-to-text → wake word → assistant →
text-to-speech. Only utterances that *begin* with a wake word are acted on; everything else is dropped
without leaving the process.

**Setup (one manual step: a speech model):**

```bash
pip install -r requirements-voice.txt
# Linux also: sudo apt install libportaudio2 espeak-ng
# download a Vosk model from https://alphacephei.com/vosk/models (vosk-model-small-en-us is ~40 MB),
# unpack it so that models/vosk/model/ contains its files, then:
python -m zivawhite.app voice --devices       # pick your microphone (set voice.stt.input_device)
python -m zivawhite.app voice --say "hello"   # test the speaker
python -m zivawhite.app voice --test-mic      # see what it hears and which words count as a wake word
python -m zivawhite.app voice                 # talk to it; Ctrl+C to stop
```

To have it listen all the time, set `voice.enabled: true`; the daemon then keeps the voice loop alive.

- **Wake words** are `voice.wake.words` (defaults: "hey ziva", "ziva", "hey snow white", "snow white").
  Offline recognisers have never heard a made-up name and write down what it sounds like, so matching
  scores both spelling and a rough phonetic key ("siva", "zeva" and "shiva" wake it; "diva" and "eva" don't).
  If yours keeps writing something else, run `--test-mic`, see the spelling, and add it to the list.
  `voice.wake.sensitivity` is the minimum match score.
- **Conversation:** "hey Ziva, what time is it" works in one breath; "hey Ziva" alone gets the `ack_phrase`
  and the next utterance is the command; after a reply you can keep talking without the wake word for
  `follow_up_s`. "stop" / "be quiet" silences speech.
- **Half-duplex by design:** it doesn't listen while it speaks (so it can't wake itself), which means you
  can't interrupt by talking over it; use the Stop control in the web UI.
- **Engines:** speech-to-text `vosk` (small, always-on friendly) or `faster_whisper` (more accurate,
  heavier). Speech output uses your OS voice (Windows SAPI, macOS `say`, Linux spd-say/espeak), `pyttsx3`, or
  your own command (`voice.tts.engine: custom`, e.g. a neural voice).
- **In the browser** the web UI has a mic button, a hands-free mode with the same wake words, and spoken
  replies. That uses your browser's speech recognition, which in Chrome/Edge sends audio to the vendor's
  cloud. It is convenient, not private or offline; use the local voice loop for that.

## Device control

Say it or type it; both take the same path. Matched by a fixed grammar (`control/intents.py`):

| Say | Does |
|---|---|
| volume up / down / "set the volume to 40" / mute / unmute | system volume |
| brightness up / down / "set the brightness to 70" | screen brightness |
| play, pause, next song, previous track | media keys / players |
| take a screenshot | saves a PNG under `device_control.screenshot_dir` |
| open calculator / open the terminal app | only apps listed in `device_control.apps` |
| go to youtube dot com / open website example.com | http(s) only, real hostnames only |
| lock the screen / sleep / restart / shut down | asks "are you sure?" first |
| what time is it / the date / battery / CPU usage | answers directly |
| set a timer for 10 minutes / remind me in an hour to call mom | timers that survive restarts and speak when due |
| turn off the living room light | Home Assistant, if you enable it and map names |

**How it is kept safe:**

- The grammar only ever sees *your* words. Model output and fetched web text are never parsed as
  commands, so a hallucination or a hostile page cannot lock your screen.
- Patterns are anchored: "what is the volume of a sphere" does nothing.
- "open X" launches only apps in the config's allow-list. Commands are argument lists, never shell
  strings. URLs must be http(s) with a real hostname, no credentials or ports.
- Lock, sleep, restart and shutdown always ask for a spoken or clicked yes (enforced inside the controller,
  not just the UI). It expires after `confirm_ttl_s`, any other message cancels it, and it is per session.
  Restart and shutdown are refused outright unless `allow_power_actions: true`.
- Over HTTP, control works only for loopback callers or ones with the API key, and never for a request
  carrying another website's `Origin` (so a page open in your browser can't press buttons). Requests that
  pass through a reverse proxy are never treated as local; use an API key there.
- Platform notes: Linux needs the usual tools (`pactl`/`wpctl`/`amixer`, `brightnessctl`, `playerctl`,
  a screenshot tool); a missing one is named in the reply. macOS brightness needs `brew install brightness`;
  lock uses a System Events keystroke, which needs Accessibility permission. On Windows volume is exact only
  with `pycaw` (otherwise approximated with volume keys), and brightness works on laptop panels.

## Running in the background

```bash
python -m zivawhite.app run                # foreground: web UI + API + jobs (Ctrl+C stops)
python -m zivawhite.app daemon start       # the same, detached; no terminal needed
python -m zivawhite.app daemon status      # state, each job, model, learning progress
python -m zivawhite.app daemon logs -n 100
python -m zivawhite.app daemon stop        # clean shutdown; force-kills the process tree only if it hangs
python -m zivawhite.app daemon install     # start at login: systemd user service / LaunchAgent / Task Scheduler
python -m zivawhite.app daemon uninstall
python -m zivawhite.app run --no-server    # jobs only, no web server (e.g. a voice-only box)
```

Jobs (`background.jobs`): **autolearn** (the datasets watcher), **timers**, and **voice** (only if
`voice.enabled`). Each runs supervised: a crash restarts it with exponential backoff, repeated crashes
mark it `failed` instead of looping forever, and one that can't work on this machine (no microphone, no
speech model) is marked `unavailable` with the reason and not retried. State goes to
`runs/daemon_status.json` every few seconds; logs to `runs/daemon.log` with size rotation. Timers set
through `command` fire only while something is running.

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
- **Output is noise, or "vocab ... != tokenizer" errors.** The `tokenizers` package isn't installed, so
  the byte-level fallback (vocab 260) was used. `pip install tokenizers`.
- **The wake word is never recognised.** `voice --test-mic`, then add the spelling you see to
  `voice.wake.words`. If nothing is recognised at all, check `--devices` and the OS microphone permission.
- **"Voice unavailable".** The message says what is missing: `pip install -r requirements-voice.txt`,
  PortAudio on Linux, or the Vosk model folder.
- **Web queries return "No web results".** Set `SEARXNG_URL` (or `SEARCH_API_URL` + `SEARCH_API_KEY`) in
  `.env`; SearxNG needs its JSON format enabled. Until then auto-routed web queries fall back to the
  local model.
- **Resuming on different data.** The "best checkpoint" gate compares validation loss, which is only
  meaningful on the same data; autolearn passes `--reset_best` for you.

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
  control/              intents.py (grammar), backends.py (Windows/macOS/Linux), controller.py, timers.py, smarthome.py
  voice/                vad.py, stt.py (Vosk/faster-whisper), tts.py, wake.py, audio.py, loop.py
  daemon.py             supervisor, start/stop/status/logs, autostart for systemd/launchd/Task Scheduler
  runtime.py            builds everything from the config; model hot-reload
  server.py, access.py  HTTP API + web UI; who may control the device over HTTP
  model.py              ZivaGPT, the transformer (KV-cached generation)
  chat.py               chat format + reply stopping, shared by training data and inference
  tokenizer.py          BPE (or dependency-free byte-level fallback)
  device.py             device + precision detection (CUDA/MPS/CPU, fp16 vs bf16). Model size is never inferred from it
  modelconfig.py        ModelConfig + presets + exact parameter counts + memory estimate (no torch needed)
  hf_compat.py          the exact ZivaGPT <-> LlamaForCausalLM weight/config mapping (proven in tests/test_hf_compat.py)
  export_hf.py          export a Hugging Face folder (safetensors, tokenizer, chat template, model card, Ollama Modelfile), --push
  import_hf.py          start from an open Llama-architecture model (e.g. SmolLM2) instead of from scratch
  sft.py, dpo.py        assistant-only fine-tuning; Direct Preference Optimization (+ self-play pair building)
  tools.py              bounded, read-only tool use (calculator, clock, web) + verifiable synthetic tool-use data
  evaluate.py, plan.py  perplexity / bits-per-byte / repetition metrics; compute planner (what can I afford?)
  data/hf_prepare.py    stream Hugging Face datasets into shards / chat JSONL / preference pairs / a trained tokenizer
  checkpoint.py         atomic, crash-safe save/load
  router.py             ZivaWhite: local model + web search/fetch
  train.py              training loop, same script every tier, torchrun-aware
  app.py                the CLI
  security.py           opt-in API key + rate limiting
  webui/index.html      browser UI: chat, mic, hands-free, spoken replies
  connectors/           web_search / web_fetch only
scripts/online_learning/  feedback -> drift/eval/promotion management loop (see below)
tests/                  pytest suite
```

## Running it on Windows

Everything here is plain Python + PyTorch; nothing is Linux/macOS-specific. Use `.venv\Scripts\activate`
(cmd) or `.venv\Scripts\Activate.ps1` (PowerShell), not the Unix `source` form; `start.bat` does all of
that for you. Hardware detection queries Windows directly (`GlobalMemoryStatusEx` through `ctypes`) when
`psutil` isn't available. Autostart uses a Task Scheduler logon task (`daemon install`).


## The model half — ZivaGPT

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

# 1. A general vocabulary (the bundled 12k tokenizer is English books; code, numbers and other languages need more)
pip install -r requirements-publish.txt
python -m zivawhite.app data tokenizer --recipe fineweb --max_docs 300k --vocab_size 32000 --out tokenizers/gen32k

# 2. Pretrain on streamed web text (writes shards directly; no giant text file)
python -m zivawhite.app data text --recipe fineweb --tokenizer_dir tokenizers/gen32k --out_dir data/processed --max_tokens 5b
python -m zivawhite.app train --config configs/general_125m.yaml --data_dir data/processed --tokenizer_dir tokenizers/gen32k --out_dir runs/base

# 3. Teach it to chat (assistant-only loss, system/tool roles, epochs + early stopping)
python -m zivawhite.app data chat --recipe smol-smoltalk --out data/chat/smol.jsonl --max_rows 100k
python -m zivawhite.app tool-data --out data/chat/tool_use.jsonl --n 3000       # verifiable calculator/clock conversations
python -m zivawhite.app sft --data data/chat/*.jsonl --init_from runs/base/checkpoint.pt --tokenizer_dir tokenizers/gen32k --out_dir runs/chat

# 4. Prefer better answers (second stage: needs a chat-tuned model)
python -m zivawhite.app data prefs --recipe ultrafeedback --out data/prefs/uf.jsonl --max_rows 20k
python -m zivawhite.app dpo --pairs data/prefs/uf.jsonl --init_from runs/chat/checkpoint.pt --tokenizer_dir tokenizers/gen32k --sft_alpha 0.1 --out_dir runs/dpo

# 5. Measure, then use it
python -m zivawhite.app eval --checkpoint runs/dpo/checkpoint.pt --tokenizer_dir tokenizers/gen32k --text data/heldout.txt --prompts prompts.txt
python -m zivawhite.app agent "What is 17% of 2340?" --checkpoint runs/dpo/checkpoint.pt
python -m zivawhite.app serve                        # includes an OpenAI-compatible /v1/chat/completions (streaming)

# 6. Publish: Hugging Face, Ollama, vLLM (see docs/PUBLISHING.md)
python -m zivawhite.app export --checkpoint runs/dpo/checkpoint.pt --tokenizer_dir tokenizers/gen32k --out exports/my-model --verify
```

Or start from open weights instead of random numbers (`import-hf`, see `docs/CAPABILITIES.md`: it is the one step
that closes most of the gap to Ollama-class models). What each stage can and cannot do, honestly, is in
[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md); publishing is in [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

**Overfitting guards** (why a fine-tune used to memorise its data): `--max_epochs`, `--patience`, `--dropout`, and a
final-step validation, so a short run is measured too. `checkpoint_best.pt` only moves when validation loss improves,
and that is what `ask`/`serve` load.

**Safety of weights:** checkpoints load with `torch.load(weights_only=True)` first (no pickle code execution);
`export` writes safetensors; `ZIVAWHITE_ALLOW_PICKLE=0` forbids the pickle fallback entirely.

## The router — local model + live web, nothing else

`router.py` decides per query:

- a URL in the query, or freshness language ("today", "latest", "right now",
  a year, "weather", "live score", "current price/president/...", "who is the
  current...") → **web**: search, optionally fetch a page, then have the local
  model (if it's been instruction-tuned) write the answer grounded in what was
  actually retrieved. If the web returns nothing (no search backend configured,
  offline), the local model answers instead of a dead end.
- otherwise, if a local checkpoint is loaded → **local**. The router also checks
  its own confidence (mean top-1 probability over the tokens it generated) and
  falls back to web search if that's below a floor and a search backend is
  configured. That number is a rough signal, not a measure of correctness.
- no local model → **web**, if a search backend is configured.
- nothing configured at all → still returns an honest answer (raw search
  context, or a plain statement that no backend is available) instead of crashing.

Force a route explicitly with `route="local"|"web"` instead of `"auto"` any time.
Chat-tuned checkpoints are prompted in their chat format and stop when the reply
ends; a base checkpoint just continues the text (the response says so).

## What the AGI capability list maps to here

A separate document (RAG explanation + an AGI capability checklist —
transfer learning, abstract reasoning, continuous autonomous learning,
strategic planning, open-ended adaptability, multimodal perception,
embodiment) came up alongside the bug report, in an earlier round.
Same honesty as the table above: nothing here is AGI, and a few items
on that list — a system that adjusts its own understanding "in
real-time without human intervention," or genuinely open-ended
adaptability to scenarios it was never built for — describe a research
problem nobody has actually solved, for any system, at any scale.
Building toward that honestly means bounded, checkable versions of the
pieces that generalize, not a larger claim. Two rows below changed
when the AI-API wrapper was removed (see "Removed: the AI-API wrapper" further
down) — noted inline rather than quietly updated:

| AGI checklist item | What actually exists for this | File |
|---|---|---|
| RAG / "search grounding" | the `web` route: search results become the ONLY context in a strict "answer from this or say you can't" prompt to the local model (once it's instruction-tuned), source URLs returned alongside the answer | `router.py:_run_web` |
| Multimodal perception | **removed this round** along with the AI-API wrapper — it only ever existed because an external API could see images; the local model is a text-only byte/BPE transformer with no vision capability, so there's no honest version of this to keep | — |
| Continuous learning | bounded and gated, not unsupervised real-time self-modification: feedback -> drift score -> eval -> a human-run promotion decision, never automatic. **Also new this round**: `autolearn.py` now watches the datasets folder and trains on whatever you put there; its optional web intake is still gated by the license allow-list below | `scripts/online_learning/`, `autolearn.py` |
| Transfer across tasks | real, measured transfer — train the base model on your text, then fine-tune a copy on your conversations, with optional EWC (`autolearn.ewc`) protecting what the base already learned — not zero-shot, no-retraining transfer, which isn't a real capability of any current system | `autolearn.py` + `continual.py` |
| Strategic planning, abstract reasoning / common sense | **removed this round** — both were previously "route to whichever external API model is strong at this," which no longer exists now that the wrapper is gone; there is no honest bounded version of either using only a from-scratch model at this project's parameter scale, so this is scoped out rather than faked | — |
| Adaptability in open-ended environments, embodiment/robotics | genuinely out of scope for this project — no honest bounded version of either exists to build toward here | — |

## Removed: the AI-API wrapper

This project used to be, honestly, two things wearing one name: a
real local model, plus a wrapper that could delegate answer-writing to
Anthropic, Gemini, any OpenAI-compatible endpoint, or any other REST
API via a generic connector. That wrapper — `connectors/
anthropic_connector.py`, `gemini_connector.py`, `openai_connector.py`,
`generic_connector.py`, `tools.py`'s function-calling loop, and the
`"api"` route in `router.py` — has been removed. Every answer this
project produces now comes from its own trained weights. What's left
in `connectors/` is exactly two things, both read-only and neither an
AI model: `web_search_connector.py` and `web_fetch_connector.py`, used
for grounding a `web`-routed answer and for gathering text `autolearn`
trains on — never for having another company's model write the
answer.

What this costs you, plainly: no more automatic fallback to a strong
external model when the local one is weak or untrained, no more
"any REST API in five lines of `.env`," and no multimodal image input
(that only ever worked because an external API could see images — see
the capability table above). What it buys you: `ask()` never sends
your query to a third party, `.env` no longer needs to be a list of
other companies' API keys, and every answer's quality is a true
reflection of *this* model, not whichever backend happened to answer
that query.

### "Search the web without being limited"

Commercial search APIs cap you (Google CSE: 100 free queries/day,
then billed; Bing similarly) — that's a real limit self-hosting routes
around. **SearxNG**, a free, open-source metasearch engine, fans a
query out to dozens of real engines and hands back JSON. Point
`SEARXNG_URL` at your own instance and there's no per-query quota,
because it's your infrastructure:

```bash
docker run -d --name searxng -p 8888:8888 searxng/searxng
echo "SEARXNG_URL=http://localhost:8888" >> .env
```

`web_search_connector.py` uses it as the primary backend and falls
back to a commercial API (`SEARCH_API_URL`/`SEARCH_API_KEY`) only if
you'd rather not run the container.

### Training itself on public data — the actual guardrail

"Train on online data that's allowed to be used by the public" is
implemented as a literal allow-list, not a vague intention:
`data/sources.py` is a `REGISTRY` of domains with a license note
each (Project Gutenberg — public domain; Wikipedia/Wikisource — CC
BY-SA; arXiv abstracts; specific `.gov` public-domain sources). This
is the one restriction in this project that stayed in place through
every other round of removing limits — it's a copyright/licensing
safeguard, not an arbitrary capability limit, and training on text
without a license that permits it isn't something this project will do
regardless of what else changes. Two independent gates check against
it before anything reaches a training shard:

1. The router's opportunistic loop: when a `web` route fetches a
   page from an allow-listed domain, that text (with URL + license)
   is appended to `data/queue/*.jsonl`. Nothing from an unlisted
   domain is ever queued, even though it was fine to *fetch and read
   once* to answer a question — reading and training are separate
   permissions on purpose.
2. The optional **web intake** in `autolearn.py` (`autolearn.web.enabled`, off by default). It reads
   YOUR topics (`_sources/topics.txt`) and URLs (`_sources/urls.txt`); none ship. It keeps only
   allow-listed result URLs, fetches them, and saves each page as an ordinary file under
   `datasets/text/web/`, with its URL, source and license recorded in
   `datasets/_sources/web_provenance.jsonl`. From there it is just another dataset file: visible,
   deletable, and trained through the same pipeline. No AI model picks the topics or decides what is
   worth fetching (that would quietly bring the wrapper back). The license gate applies at fetch and
   again at export, and no setting turns it off. It does **not** apply to files you place in
   `datasets/` yourself: that data is yours to vouch for.

Add a source by adding an entry to `REGISTRY` with an honest license
note — don't add anything paywalled or ToS-restricted, since that
defeats the point of having the gate at all.

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

## HTTP API

```bash
python -m zivawhite.app serve --port 8000     # API + web UI only; `run` adds the background jobs. Binds 127.0.0.1
```

```
GET  /v1/health                                    runtime, which model file is loaded, web search configured?
GET  /v1/version                                   app version, whether auth is required
POST /v1/ask       {"query": "...", "route": "auto"}            routed answer (identity/local/web); never touches the device
POST /v1/generate  {"prompt": "...", "max_new_tokens": 200, "raw": false}   local model only, new text only
POST /v1/assistant {"text": "...", "session": "...", "route": "auto"}   the full assistant: identity, DEVICE CONTROL, model
GET  /v1/assistant/info                            names, wake words (for the web UI)
GET  /v1/assistant/events?since=N                  background notifications (timers)
POST /v1/voice/stop                                stop speaking (local/authenticated callers)
GET  /v1/datasets                                  folder status + what the learner is doing
GET  /v1/status                                    background jobs, model, learning state
```

`/v1/assistant` is the only endpoint that can act on the machine, and only for loopback callers or ones
presenting the API key whose `Origin` isn't a foreign site (see "Device control"). Everything else about
auth below applies to all `/v1` endpoints except `/v1/health`, `/v1/version` and `/v1/assistant/info`.

No `image_base64`/`image_media_type` fields any more — see "Removed:
the AI-API wrapper" above for why multimodal input didn't survive that
removal, and "Removed: the artificial limits on the model" below for
why `query`/`prompt`/`max_new_tokens` no longer have arbitrary caps.

### Production hardening — opt-in, off by default

Auth and rate limiting default to off/loose, so `launch.py` and
local testing keep working with zero setup. Turn each on via `.env`
(see `.env.example`) before putting this somewhere other than your own
machine can reach it — see `security.py` for the implementation. These
protect the **server** (who can call it, how often) — a different
thing from the model-capacity caps removed below, and unaffected by
that removal:

- **Auth** — set `ZIVAWHITE_SERVER_API_KEY` and the `/v1` endpoints (ask, generate, assistant,
  events, datasets, status) require `Authorization: Bearer <that key>`, 401
  otherwise. `/v1/health` and `/v1/version` stay open (so a monitoring
  probe doesn't need a key), and `/v1/version` reports
  `"auth_required": true/false` so the bundled webui knows to show its
  API-key field automatically.
- **Rate limiting** — `ZIVAWHITE_RATE_LIMIT="<requests>/<seconds>"`
  (default `60/60`), per client IP, 429 + `Retry-After` once exceeded.
  `X-Forwarded-For` is only honored with `ZIVAWHITE_TRUST_PROXY=1` (otherwise any client
  could dodge the limit by faking it). In-memory and
  single-process by design — put a real edge limiter in front if you're
  running multiple worker processes.
- **Binding / CORS** — the server listens on `127.0.0.1` unless you pass
  `--host 0.0.0.0` (set `ZIVAWHITE_SERVER_API_KEY` first; it warns if you don't).
  Browser calls from other sites are refused unless they come from localhost;
  `ZIVAWHITE_CORS_ORIGINS` takes a comma-separated list, or `*`.
- **Structured errors everywhere** — a validation failure, a missing
  route, an expected error (no checkpoint loaded, rate limited,
  unauthorized), and a genuinely unexpected exception all come back as
  `{"error": "..."}` with the right HTTP status, never a raw stack
  trace to the client (the real exception still goes to the server's
  own log).
- **Request logging** — every request logged with
  method/path/status/latency.

### Removed: the artificial limits on the model

`AskBody`/`GenBody` in `app.py` used to cap `query`/`prompt` at 8000
characters, `max_new_tokens` at 2000, and an image at ~11MB decoded.
None of those were real constraints of the model — they were arbitrary
API-layer ceilings. All three are gone: `min_length=1`/`ge=1` remain
(an empty query or a request for zero tokens is a no-op, not a
capacity limit), but there's no longer an upper bound beyond what the
model's own `block_size` and available memory actually allow. Verified
directly: a query of 18,000 characters and a `max_new_tokens=50000`
request are both accepted now where they'd previously have been
rejected at the API layer.

## Experimental modules — real alternatives to the AGI blueprint

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
