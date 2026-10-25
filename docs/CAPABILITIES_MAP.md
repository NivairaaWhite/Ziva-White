<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
A short map from an AGI capability checklist to what exists here. See also [CAPABILITIES.md](CAPABILITIES.md).

# What the AGI capability list maps to here

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
