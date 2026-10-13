# Your training data goes here

Drop files into this folder. While Ziva White is running (`python -m zivawhite.app run`, the
double-click launcher, or the background daemon), it notices them and trains on them by itself.
There is no sample data and nothing is downloaded for you: **the assistant learns from what you put
here and nothing else.** An empty folder means it has nothing to learn yet, and it says so.

(This folder's location is a setting: `paths.datasets` in `configs/ziva.yaml`, or the
`ZIVAWHITE_DATASETS` environment variable. This README is never used as training data.)

## Two kinds of data

| Put it in | What it teaches | Examples |
|---|---|---|
| `text/` | The language itself: vocabulary, facts, style. | books, articles, notes, documentation, a site you saved, a chat export you want it to *read* |
| `chat/` | How to answer: question in, reply out. | support conversations, Q&A pairs, instruction datasets |

You don't have to sort files into those folders. Structured files (`.jsonl`, `.json`, `.csv`) are
classified by what is *inside* them, wherever they sit. The folders only matter for plain text
files, where they say what the file is.

A model needs to read before it can be taught to answer, so put **text first** (as much as you can),
then chat data. Chat-only data works too, but the model then has to learn the language from your
conversations alone and will be much weaker.

## Formats

**Text:** `.txt` `.md` `.markdown` `.rst` `.text` `.log` `.html` `.htm` `.pdf`
(PDF needs `pip install pypdf`, which `requirements.txt` already includes; scanned PDFs with no text
layer produce nothing.) Rows of `.jsonl`/`.json`/`.csv` with a `text`, `content`, `body`,
`document`, `article` or `passage` field are text too. Project Gutenberg headers and footers are
stripped automatically.

**Conversations**, any of these:

```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]}
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
{"prompt": "What is 2+2?", "response": "4"}
{"instruction": "Translate to French", "input": "cat", "output": "chat"}
{"question": "Who wrote Hamlet?", "answer": "Shakespeare"}
```

(`.jsonl` is one JSON object per line; `.json` is a list of them, or an object holding a list under
`data`, `examples`, `records`, `items`, `train` or `rows`; `.csv`/`.tsv` use the same names as column
headers.) Roles `user`/`human`/`assistant`/`gpt`/`bot`/`model` are understood; a `system` message is
folded into the first user turn.

Plain-text transcripts work in `chat/` (or anywhere, if they contain both user and assistant turns):

```text
User: What's the capital of France?
Assistant: Paris.
---
Q: And of Spain?
A: Madrid.
```

## What gets ignored

* anything whose name starts with `_` or `.`, and everything inside folders that do. Use this for
  `_notes`, `_old`, or `_eval/`, where held-out evaluation files for the online-learning kit live;
* this `README.md`;
* file types not listed above (they're named in `autolearn --status`, so nothing disappears silently);
* exact duplicates, which are dropped.

A file that can't be read (corrupt PDF, invalid JSON, over `autolearn.max_file_mb`) is reported with
the reason and skipped; the rest still trains.

## What happens after you drop a file

1. Within `autolearn.scan_interval_s` (30 s) the watcher sees it. Files modified in the last few
   seconds are left alone, so a file that is still being copied is never read half-written.
2. Everything is cleaned, de-duplicated and rebuilt into a text corpus and a conversation corpus.
3. **Only the corpus that actually changed is trained.** Touching a file, or adding chat data, doesn't
   retrain the text model for nothing.
4. If a corpus is smaller than `autolearn.min_train_tokens` (text) / `min_chat_tokens` (chat), it waits and
   tells you how much more it needs: a few kilobytes would only be memorised.
5. Training runs as a low-priority child process, and only while the machine is idle and plugged in
   (`autolearn.throttle`). A crash is reported and retried next round; nothing is lost.
6. The new model is loaded without restarting anything.

Check progress any time:

```bash
python -m zivawhite.app datasets          # files, what changed, what still needs training
python -m zivawhite.app daemon status     # if it's running in the background
python -m zivawhite.app autolearn --once  # run one round right now, in the foreground
```

## Things to know

* **Removing a file stops it being part of *future* training. It does not make an already-trained model
  forget it.** To start clean, delete `runs/` (the models) and `data/datasets/` (the build state).
* A small model trained on a little data will talk fluently and be wrong a lot. More data and more steps
  help; so does a bigger model (`autolearn.train_args: ["--tier", "small"]`). Use the web route for facts.
* Only use data you have the right to use. The license allow-list that gates the *automatic web
  download* (`autolearn.web`) does not apply to files you put here: that's your call.
* The assistant's name and its creator ("Mahasvin S S (Snow White)") come from `configs/ziva.yaml`,
  not from your data. When you have chat data, a few dozen identity examples generated from that
  config are mixed in so the model learns them too; a text-only dataset is never turned into a chat
  model by them.
* Optional web intake: set `autolearn.web.enabled: true` and put URLs in `_sources/urls.txt`
  (one per line) or search topics in `_sources/topics.txt`. Allow-listed public pages are saved
  into `text/web/` as ordinary files you can read and delete, with their license recorded in
  `_sources/web_provenance.jsonl`. Nothing is bundled in those files.
