"""
infer.py - raw local-model sampling, no routing, no web.

Chat-tuned checkpoints are prompted in the chat format they were trained on
and stop when the reply ends; base checkpoints continue the prompt text. Pass
--raw to force plain continuation even for a chat-tuned model (useful for
seeing what the model does without the template). Only the newly generated
text is printed, not the prompt.
"""
from __future__ import annotations

import argparse

from .chat import generate_text
from .checkpoint import find_default_checkpoint, load_model
from .device import resolve_runtime
from .tokenizer import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="default: runs/ziva_behavior/ or runs/ziva/ if present")
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--raw", action="store_true", help="plain text continuation, no chat template")
    args = p.parse_args()

    checkpoint = args.checkpoint or find_default_checkpoint()
    if not checkpoint:
        raise SystemExit("no checkpoint given and none found under runs/ -- pass --checkpoint")

    rt = resolve_runtime()
    model, ckpt, device, src = load_model(checkpoint, rt.device)
    tok = load_tokenizer(args.tokenizer_dir)
    if model.cfg.vocab_size != tok.vocab_size:
        raise SystemExit(f"{src}: vocab {model.cfg.vocab_size} != tokenizer vocab {tok.vocab_size} "
                         f"(is `tokenizers` installed? pass --tokenizer_dir if this checkpoint used its own)")

    chat = bool(ckpt.get("extra", {}).get("behavior_tuned", False)) and not args.raw
    print(generate_text(model, tok, args.prompt, chat=chat, device=device, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k))


if __name__ == "__main__":
    main()
