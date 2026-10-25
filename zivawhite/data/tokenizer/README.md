# Bundled tokenizer

`vocab.json` + `merges.txt` are a real byte-level BPE tokenizer (vocab_size=12000) for general English,
trained once, offline, with the `tokenizers` library (see `tokenizer.py:BPETokenizer.train`).

## What it is: a starting point for a NEW model, not a fixed setting

* **A model you have not trained yet** starts from this tokenizer, unless you point
  `autolearn.tokenizer_dir` (or `train --tokenizer_dir`) at another one.
* **Every training run saves the tokenizer it used** into `runs/<run>/tokenizer/` and records a fingerprint of it in
  its checkpoints. From then on `ask`, `serve`, the web UI, `generate`, `eval`, `sft`, `dpo` and `export` find
  *that* one next to the checkpoint by themselves. You never repeat `--tokenizer_dir` for a model you trained,
  and a model trained with another vocabulary is never quietly paired with this one: a mismatch stops with a
  message naming what the checkpoint needs and where it looked.
* A model keeps its vocabulary for life. A different tokenizer means a different model: start a new run.

## Why a BPE at all

Without a real tokenizer every entry point fell back to bytes (one token per character): the same passage that
costs 124 byte-level tokens costs about 32 BPE tokens, so a fixed `block_size` covers about 4x more real text.

## Using your own vocabulary

Worth it for a language other than English, or text that is very unlike English books (code, medicine, legal, a
game's lore). It needs enough text to learn from: about 150 characters per vocabulary entry.

```bash
python -m zivawhite.app tokenizer --vocab_size 8000     # learns from your datasets/ folder -> data/tokenizer
# then in configs/ziva.local.yaml, BEFORE the first training run:
#   autolearn:
#     tokenizer_dir: data/tokenizer
```

For a large general-purpose vocabulary trained on streamed web text, see
[docs/MODEL.md](../../../docs/MODEL.md) (`app.py data tokenizer --recipe fineweb ...`).
