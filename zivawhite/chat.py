"""
chat.py - the one place that defines what a conversation looks like to the
model, used by BOTH the training-data renderer (data/behavior.py) and
inference (router.py, app.py /v1/generate, infer.py).

Why this exists: fine-tuning taught the model to expect

    <|user|>
    hello
    <|assistant|>
    <reply>

but inference used to feed it the bare string "hello", so a chat-tuned
model never saw the format it was trained on and fell back to whatever its
base corpus looked like. Rendering and prompting from the same functions
makes that mismatch impossible.

`<|user|>` / `<|assistant|>` are ordinary text to the tokenizer (they BPE
into several pieces), not reserved ids -- that's kept as-is so existing
checkpoints stay compatible.

Nothing here imports torch at module level, so the formatting and stop
logic can be tested without it.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"   # persona / rules for the conversation (first message)
TOOL_TAG = "<|tool|>"       # the result of a tool call, written by the program, never by the model
DOC_MARKER_PREFIX = "<|doc"  # document boundary lines in corpus files: "<|doc|>" / "<|doc:name|>"

# A chat-mode reply must never run into one of these: it means the model has
# started writing the next turn (or a new document) instead of stopping. <|tool|> is here because a
# model that writes its own tool result is hallucinating one.
STOP_MARKERS = (USER_TAG, ASSISTANT_TAG, SYSTEM_TAG, TOOL_TAG, DOC_MARKER_PREFIX)

_ROLE_TAGS = {"user": USER_TAG, "assistant": ASSISTANT_TAG, "system": SYSTEM_TAG, "tool": TOOL_TAG}

# The same chat format as a Hugging Face `chat_template` (Jinja), written into every export so that
# tokenizer.apply_chat_template(), vLLM and friends build EXACTLY the prompt build_chat_prompt() builds
# (tests/test_chat_template.py renders both and compares them byte for byte). The leading bos_token mirrors
# fit_prompt_ids(), which starts every prompt with <bos>.
HF_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] | trim }}\n"
    "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] | trim }}\n"
    "{% elif m['role'] == 'tool' %}<|tool|>\n{{ m['content'] | trim }}\n"
    "{% elif m['role'] == 'assistant' %}<|assistant|>\n{{ m['content'] | trim }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

Message = Dict[str, str]


def prepare_chat_request(messages: Sequence[Message]) -> "tuple[str, List[Message], str]":
    """Validate an OpenAI-style message list and split it into (the new text, the history before it, its role).
    Raises ValueError with a message fit to show a client. system/user/assistant/tool are all kept as their
    own roles (the model is trained on them); the LAST message must be the user's, or a tool result the model
    is now answering (the tool-use loop)."""
    if not messages:
        raise ValueError("messages must contain at least one message")
    clean: List[Message] = []
    for i, m in enumerate(messages):
        role, content = m.get("role"), m.get("content")
        if role not in _ROLE_TAGS:
            raise ValueError(f"messages[{i}].role must be one of {sorted(_ROLE_TAGS)}, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string")
        clean.append({"role": role, "content": content})
    if clean[-1]["role"] not in ("user", "tool"):
        raise ValueError("the last message must have role 'user' or 'tool' (the model replies to it)")
    if not clean[-1]["content"].strip():
        raise ValueError(f"the last {clean[-1]['role']} message is empty")
    return clean[-1]["content"], clean[:-1], clean[-1]["role"]


def render_turn(role: str, content: str) -> str:
    tag = _ROLE_TAGS.get(role)
    if tag is None:  # used to fall through to the assistant tag, silently mangling e.g. a system prompt
        raise ValueError(f"unknown chat role {role!r}; expected one of {sorted(_ROLE_TAGS)}")
    return f"{tag}\n{content.strip()}"


def render_conversation(messages: Sequence[Message]) -> str:
    """Exactly the text the fine-tuning corpus contains for one example."""
    return "\n".join(render_turn(m["role"], m["content"]) for m in messages)


def build_chat_prompt(user_text: str, history: Optional[Sequence[Message]] = None, role: str = "user") -> str:
    """Render `history` + the new turn (a user message, or a tool result when `role="tool"`), ending on an
    open assistant turn so the model's next tokens are the reply."""
    messages = list(history or []) + [{"role": role, "content": user_text}]
    return render_conversation(messages) + "\n" + ASSISTANT_TAG + "\n"


def clean_reply(text: str, markers: Sequence[str] = STOP_MARKERS) -> str:
    """Cut a generated reply at the first stop marker and trim whitespace."""
    cut = len(text)
    for m in markers:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def make_stop_fn(tokenizer, prompt_len: int, markers: Sequence[str] = (), eos_id: Optional[int] = None,
                 tail_tokens: int = 16) -> Callable:
    """Build the `stop_fn(idx) -> bool` that ZivaGPT.generate calls after every
    new token. Stops when the decoded tail of the *generated* tokens contains
    a stop marker, or on <eos> once real content has been produced.

    <eos> only counts after at least one content token: checkpoints trained
    before per-document EOS (one <eos> after every line) emit <eos> right after
    "<|assistant|>\\n", and stopping there would always give an empty reply."""
    specials = {getattr(tokenizer, a, None) for a in ("pad_id", "bos_id", "eos_id", "unk_id")}
    state = {"content": False}

    def stop_fn(idx) -> bool:
        n_new = idx.size(1) - prompt_len
        if n_new <= 0:
            return False
        last = int(idx[0, -1])
        if last not in specials:
            state["content"] = True
        if eos_id is not None and last == eos_id and state["content"]:
            return True
        if markers:
            tail = idx[0, -min(n_new, tail_tokens):].tolist()
            text = tokenizer.decode(tail)
            return any(m in text for m in markers)
        return False

    return stop_fn


def fit_prompt_ids(tokenizer, text: str, render: Callable[[str], str], limit: int) -> List[int]:
    """Token ids for render(text), guaranteed <= `limit` long. A too-long
    message is shortened from the *left* (the question is usually at the end)
    BEFORE it's wrapped in the chat template, so the template itself (the
    <|user|>/<|assistant|> tags the model depends on) is never cut off."""
    ids = tokenizer.encode(render(text), add_bos=True)
    if len(ids) <= limit:
        return ids
    overhead = len(tokenizer.encode(render(""), add_bos=True))
    budget = max(1, limit - overhead)
    text = tokenizer.decode(tokenizer.encode(text)[-budget:])
    ids = tokenizer.encode(render(text), add_bos=True)
    return ids[-limit:]  # last resort (token boundaries aren't perfectly additive)


def generate_text(model, tokenizer, text: str, *, chat: bool, device: str = "cpu", max_new_tokens: int = 200,
                  temperature: float = 0.8, top_k: Optional[int] = 50, top_p: Optional[float] = None,
                  history: Optional[Sequence[Message]] = None, stats: Optional[dict] = None,
                  min_p: Optional[float] = None, repetition_penalty: float = 1.0,
                  extra_stop: Sequence[str] = (), role: str = "user") -> str:
    """Generate and return ONLY the newly generated text.

    chat=True  : wrap `text` in the chat template, stop when the model starts a
                 new turn or emits <eos>, return the cleaned reply. Use for
                 checkpoints that were fine-tuned on the chat format.
    chat=False : raw continuation of `text` (what a base model can actually do).
    """
    import torch

    block = model.cfg.block_size
    reserve = min(max_new_tokens, max(16, block // 4))  # room left for the reply
    limit = max(8, block - reserve)
    render = (lambda t: build_chat_prompt(t, history)) if chat else (lambda t: t)
    ids = fit_prompt_ids(tokenizer, text, render, limit)

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    markers = (tuple(STOP_MARKERS) + tuple(extra_stop)) if chat else tuple(extra_stop)
    stop_fn = make_stop_fn(tokenizer, idx.size(1), markers=markers,
                           eos_id=getattr(tokenizer, "eos_id", None) if chat else None)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                         top_p=top_p, stop_fn=stop_fn, stats=stats, min_p=min_p,
                         repetition_penalty=repetition_penalty)
    new_text = tokenizer.decode(out[0, idx.size(1):].tolist())
    if chat:
        return clean_reply(new_text, markers)
    return clean_reply(new_text, tuple(extra_stop)) if extra_stop else new_text


def stream_text(model, tokenizer, text: str, *, chat: bool, device: str = "cpu", max_new_tokens: int = 200,
                temperature: float = 0.8, top_k: Optional[int] = 50, top_p: Optional[float] = None,
                history: Optional[Sequence[Message]] = None, min_p: Optional[float] = None,
                repetition_penalty: float = 1.0, info: Optional[dict] = None,
                extra_stop: Sequence[str] = (), role: str = "user") -> Iterator[str]:
    """Like generate_text but yields the reply as it is written (text deltas), for streaming servers.

    The concatenation of everything yielded equals what generate_text would return for the same
    sampled tokens. Two things make a streaming decoder subtle, and both are handled:
      * a stop marker such as "<|user|>" arrives a token at a time, so the last len(marker)-1
        characters are held back until it is clear they are not the start of one;
      * a multi-byte character can be split across tokens (the decode ends in U+FFFD), so an
        incomplete trailing character is held back too.
    Leading and trailing whitespace is never emitted (clean_reply strips it), so a delta never has to
    be retracted. `info`, if given, receives completion_tokens and finish_reason ("stop" | "length")."""
    import torch

    block = model.cfg.block_size
    reserve = min(max_new_tokens, max(16, block // 4))
    limit = max(8, block - reserve)
    render = (lambda t: build_chat_prompt(t, history, role)) if chat else (lambda t: t)
    ids = fit_prompt_ids(tokenizer, text, render, limit)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    markers = (tuple(STOP_MARKERS) + tuple(extra_stop)) if chat else tuple(extra_stop)
    eos_id = getattr(tokenizer, "eos_id", None) if chat else None
    stop_fn = make_stop_fn(tokenizer, idx.size(1), markers=markers, eos_id=eos_id)
    hold = max((len(m) for m in markers), default=0)

    generated: List[int] = []
    emitted = 0          # absolute index into the decoded text of what has been yielded
    stopped = False
    for tok in model.generate_stream(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                                     top_p=top_p, stop_fn=stop_fn, min_p=min_p,
                                     repetition_penalty=repetition_penalty):
        generated.append(tok)
        full = tokenizer.decode(generated)
        cut = _first_marker(full, markers)
        if cut != -1:
            stopped = True
            safe_end = len(full[:cut].rstrip())
        else:
            safe_end = max(0, len(full) - max(hold - 1, 0))
            if full.endswith("\ufffd"):
                safe_end = min(safe_end, len(full) - 1)
            safe_end = len(full[:safe_end].rstrip())
        start = max(emitted, len(full) - len(full.lstrip()))
        if safe_end > start:
            yield full[start:safe_end]
            emitted = safe_end
        if stopped:
            break

    full = tokenizer.decode(generated)
    cut = _first_marker(full, markers)
    final_end = len(full[:cut].rstrip()) if cut != -1 else len(full.rstrip())
    start = max(emitted, len(full) - len(full.lstrip()))
    if final_end > start:
        yield full[start:final_end]
    if info is not None:
        ended_on_eos = bool(generated) and eos_id is not None and generated[-1] == eos_id
        info["completion_tokens"] = len(generated)
        info["finish_reason"] = "length" if (len(generated) >= max_new_tokens and cut == -1 and not ended_on_eos) else "stop"


def _first_marker(text: str, markers: Sequence[str]) -> int:
    """Index of the earliest stop marker in `text`, or -1."""
    cut = -1
    for m in markers:
        i = text.find(m)
        if i != -1 and (cut == -1 or i < cut):
            cut = i
    return cut
