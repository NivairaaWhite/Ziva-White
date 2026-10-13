# Bundled tokenizer

`vocab.json` + `merges.txt` are a real byte-level BPE tokenizer (vocab_size=12000) for general English,
trained once, offline, with the `tokenizers` library (see `tokenizer.py:BPETokenizer.train`). The
text it was trained on is no longer shipped with the project: the datasets folder is yours now. See
`tokenizer.py:load_tokenizer` for how this directory is picked up as the default everywhere
(autolearn's text and conversation training, ad-hoc ingest) unless you point at a different,
already-trained tokenizer directory (`autolearn.tokenizer_dir`).

Why it exists: without it every entry point silently fell back to a byte-level tokenizer (one token per
raw byte, vocab_size=260), so the model spent its limited capacity re-learning which byte sequences form a
word. With this one the same English text takes roughly a quarter as many tokens, so a fixed
`block_size` covers ~4x more real text. It is byte-level, so any language or symbol still encodes; it is
just less efficient for text unlike what it was trained on.

To train a tokenizer on YOUR data instead (only worth it for a different language or a very different
domain, and only before you have checkpoints):

```bash
python -c "
from pathlib import Path
from zivawhite.tokenizer import BPETokenizer
files = [str(p) for p in Path('datasets/text').rglob('*.txt')]     # your own text files
BPETokenizer.train(files=files, vocab_size=8000, out_dir='data/my_tokenizer')
"
# then, in configs/ziva.local.yaml:
#   autolearn:
#     tokenizer_dir: data/my_tokenizer
```

Do not retrain over this directory once you have real checkpoints trained against it: a different
vocabulary means token id 5 (say) decodes to a different piece of text, silently corrupting anything
trained under the old one. Train the new vocab somewhere else, as above.
