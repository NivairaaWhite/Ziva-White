"""
data/folder.py - the datasets folder.

Drop files into `datasets/` (location: configs/ziva.yaml -> paths.datasets) and the
autolearn job trains on them. There is no bundled sample data and no hardcoded file
list: whatever is in that folder is the training data.

What it understands
-------------------
Text to pretrain on           .txt .md .markdown .rst .text .log .html .htm .pdf*
                              .jsonl/.json/.csv/.tsv rows with a text/content/body column
Conversations to chat-tune    .jsonl/.json/.csv/.tsv in any of these shapes:
                                {"messages": [{"role": "user", "content": "..."}, ...]}
                                {"conversations": [{"from": "human", "value": "..."}, ...]}   (ShareGPT)
                                {"prompt": "...", "response": "..."}
                                {"instruction": "...", "input": "...", "output": "..."}      (Alpaca)
                                {"question": "...", "answer": "..."}  {"user": "..", "assistant": ".."}
                              and .txt/.md under chat/ written as "User: ..." / "Assistant: ..."
                              blocks separated by a line of ---
  (*pdf needs `pip install pypdf`)

Sub-folders are optional. `text/` and `chat/` just tell ambiguous plain-text files what
they are; structured files are classified by what is in them wherever they sit. Anything
whose name starts with "_" or "." is ignored (so `_sources/` and `_eval/` are safe), and
so is the README.md at the top of the folder.

How change detection works
--------------------------
`check()` compares (size, mtime) of every file with the last scan; if something changed,
`rebuild()` regenerates both corpora from the whole folder and records a SHA-256 of each.
Training only happens for a corpus whose hash differs from the one last trained on
(`needs_training`), so touching a file, or adding chat data, never retrains the text model
for nothing. `mark_trained()` is called by the trainer only after training succeeded: a
crash means the next round retries instead of silently forgetting the new data.

Honest limits: removing a file stops it being part of FUTURE training; it does not make
an already-trained model forget it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from ..chat import render_conversation

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".text", ".log", ".html", ".htm", ".pdf"}
TABLE_EXTS = {".jsonl", ".ndjson", ".json", ".csv", ".tsv"}
SUPPORTED_EXTS = TEXT_EXTS | TABLE_EXTS
KINDS = ("text", "chat")

_TEXT_FIELDS = ("text", "content", "body", "document", "article", "passage")
_MESSAGE_FIELDS = ("messages", "conversations", "conversation", "dialog", "dialogue", "chat")
_LIST_WRAPPER_KEYS = ("data", "examples", "records", "items", "train", "rows", "conversations", "messages")
_PAIR_FIELDS = (("prompt", "response"), ("prompt", "completion"), ("instruction", "output"),
                ("question", "answer"), ("user", "assistant"), ("input", "output"),
                ("query", "response"), ("human", "bot"), ("human", "assistant"))
_ROLE_MAP = {"user": "user", "human": "user", "customer": "user", "prompter": "user", "question": "user", "q": "user",
             "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "ai": "assistant", "model": "assistant",
             "answer": "assistant", "a": "assistant", "chatbot": "assistant", "system": "system"}

_TAG_RE = re.compile(r"<\|(?:user|assistant)\|>")
_DOC_LINE_RE = re.compile(r"(?m)^<\|doc")
_BLANKS_RE = re.compile(r"\n{3,}")
_TURN_RE = re.compile(r"^\s*(user|human|q|question|assistant|ai|bot|answer|a)\s*:\s?(.*)$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|={3,})\s*$")
_USER_TURN_RE = re.compile(r"(?im)^\s*(?:user|human|q|question)\s*:")
_ASSISTANT_TURN_RE = re.compile(r"(?im)^\s*(?:assistant|ai|bot|answer|a)\s*:")
_GUTENBERG_START = ("START OF THE PROJECT GUTENBERG", "START OF THIS PROJECT GUTENBERG")
_GUTENBERG_END = ("END OF THE PROJECT GUTENBERG", "END OF THIS PROJECT GUTENBERG")


class UnsupportedFile(Exception):
    """A file we recognise but cannot read right now (missing optional package, too large, ...)."""


# ----------------------------------------------------------------------------- cleaning
def decode_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def clean_text(text: str) -> str:
    """Normalise newlines and make the text safe to embed in a corpus: chat tags in the data
    must not be readable as real turn markers and no line may start with the document-boundary
    marker the shard builder splits on."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _TAG_RE.sub("", text)
    text = _DOC_LINE_RE.sub("< |doc", text)
    return _BLANKS_RE.sub("\n\n", text).strip()


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg's licence header/footer when present (harmless otherwise)."""
    if not any(m in text for m in _GUTENBERG_START):
        return text
    lines = text.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if any(m in line for m in _GUTENBERG_START):
            start = i + 1
            break
    for i in range(len(lines) - 1, start, -1):
        if any(m in lines[i] for m in _GUTENBERG_END):
            end = i
            break
    return "\n".join(lines[start:end])


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        return soup.get_text("\n")
    except ImportError:
        html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
        return re.sub(r"(?s)<[^>]+>", " ", html)


def split_long(text: str, max_chars: int) -> List[str]:
    """Cut a very long document at paragraph boundaries so no single 'document' (which gets one
    end-of-text token) is enormous."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts, buf, size = [], [], 0
    for para in text.split("\n\n"):
        if size + len(para) > max_chars and buf:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


# --------------------------------------------------------------------------- chat shapes
def normalize_messages(raw) -> Optional[List[dict]]:
    """Turn any supported conversation shape into [{role: user|assistant, content}, ...] that
    starts with a user turn and ends on an assistant turn, or None if it is not usable."""
    if not isinstance(raw, list):
        return None
    turns: List[dict] = []
    system = ""
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = _ROLE_MAP.get(str(m.get("role", m.get("from", m.get("speaker", "")))).strip().lower())
        content = m.get("content", m.get("value", m.get("text", m.get("message", ""))))
        if not isinstance(content, str) or not content.strip() or role is None:
            continue
        if role == "system":
            system = content.strip()
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content.strip()
        else:
            turns.append({"role": role, "content": content.strip()})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if system and turns:
        turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"
    if len(turns) < 2:
        return None
    return turns


def classify_record(rec) -> Optional[Tuple[str, object]]:
    """('chat', messages) | ('text', str) | None for one parsed row/object."""
    if isinstance(rec, str):
        return ("text", rec) if rec.strip() else None
    if not isinstance(rec, dict):
        return None
    lowered = {str(k).strip().lower(): v for k, v in rec.items()}
    for key in _MESSAGE_FIELDS:
        if key in lowered:
            msgs = normalize_messages(lowered[key])
            if msgs:
                return ("chat", msgs)
    for q_key, a_key in _PAIR_FIELDS:
        q, a = lowered.get(q_key), lowered.get(a_key)
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            extra = lowered.get("input") if q_key == "instruction" else None
            question = q.strip() + (f"\n\n{extra.strip()}" if isinstance(extra, str) and extra.strip() else "")
            return ("chat", [{"role": "user", "content": question}, {"role": "assistant", "content": a.strip()}])
    for key in _TEXT_FIELDS:
        v = lowered.get(key)
        if isinstance(v, str) and v.strip():
            return ("text", v)
    return None


def parse_transcript(text: str) -> Iterator[List[dict]]:
    """'User: ...' / 'Assistant: ...' blocks; a line of --- or === separates conversations."""
    convo: List[dict] = []

    def flush() -> Optional[List[dict]]:
        msgs = normalize_messages(convo)
        convo.clear()
        return msgs

    for line in text.splitlines():
        if _SEPARATOR_RE.match(line):
            msgs = flush()
            if msgs:
                yield msgs
            continue
        m = _TURN_RE.match(line)
        if m:
            convo.append({"role": m.group(1).lower(), "content": m.group(2)})
        elif convo:
            convo[-1]["content"] += "\n" + line
    msgs = flush()
    if msgs:
        yield msgs


# --------------------------------------------------------------------- file iteration
@dataclass
class DatasetFile:
    rel: str            # path relative to the datasets folder, always "/" separated
    path: Path
    size: int
    mtime_ns: int
    hint: Optional[str]  # "text" | "chat" | None, from the top-level sub-folder name


def _hint_for(rel: str) -> Optional[str]:
    top = rel.split("/", 1)[0].lower() if "/" in rel else ""
    return top if top in KINDS else None


def _read_table(path: Path, ext: str, text: str) -> Iterator[object]:
    if ext in (".jsonl", ".ndjson"):
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield _BadRecord(f"line {lineno} is not valid JSON")
        return
    if ext == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise UnsupportedFile(f"not valid JSON ({e})") from e
        if isinstance(data, dict):
            for key in _LIST_WRAPPER_KEYS:
                inner = data.get(key)
                if isinstance(inner, list) and inner and isinstance(inner[0], (dict, str)):
                    # a bare list of messages is ONE conversation, not many records
                    if key in ("messages", "conversations") and isinstance(inner[0], dict) and \
                            ("role" in inner[0] or "from" in inner[0]):
                        break
                    data = inner
                    break
        yield from (data if isinstance(data, list) else [data])
        return
    # csv / tsv
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:  # pragma: no cover - 32-bit platforms
        csv.field_size_limit(2 ** 31 - 1)
    reader = csv.DictReader(text.splitlines(), delimiter="\t" if ext == ".tsv" else ",")
    for row in reader:
        yield {(k or "").strip().lower(): v for k, v in row.items() if k is not None}


class _BadRecord:
    def __init__(self, why: str):
        self.why = why


def iter_file(df: DatasetFile, max_file_bytes: int = 0, max_doc_chars: int = 0) -> Iterator[Tuple[str, object]]:
    """Yield ('text', str) / ('chat', messages) / ('bad', reason) for one file. Raises
    UnsupportedFile if the whole file cannot be read."""
    ext = df.path.suffix.lower()
    if max_file_bytes and df.size > max_file_bytes:
        raise UnsupportedFile(f"{df.size // (1024 * 1024)} MB is over the {max_file_bytes // (1024 * 1024)} MB "
                              f"limit (autolearn.max_file_mb) -- split the file")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise UnsupportedFile("PDF files need `pip install pypdf`") from e
        reader = PdfReader(str(df.path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        for part in split_long(clean_text(text), max_doc_chars):
            yield ("text", part)
        return

    text = decode_bytes(df.path.read_bytes())

    if ext in TABLE_EXTS:
        for rec in _read_table(df.path, ext, text):
            if isinstance(rec, _BadRecord):
                yield ("bad", rec.why)
                continue
            got = classify_record(rec)
            if got is None:
                yield ("bad", "row has no recognisable text / conversation fields")
            elif got[0] == "text":
                for part in split_long(clean_text(got[1]), max_doc_chars):
                    yield ("text", part)
            else:
                yield got
        return

    # plain text family
    if df.hint == "chat" or (df.hint != "text" and _USER_TURN_RE.search(text) and _ASSISTANT_TURN_RE.search(text)):
        found = False
        for msgs in parse_transcript(text):
            found = True
            yield ("chat", msgs)
        if found:
            return
        if df.hint == "chat":
            yield ("bad", "no 'User:' / 'Assistant:' turns found")
            return
    if ext in (".html", ".htm"):
        text = html_to_text(text)
    elif ext in (".txt", ".text", ".log"):
        text = strip_gutenberg(text)
    for part in split_long(clean_text(text), max_doc_chars):
        yield ("text", part)


# --------------------------------------------------------------------------- reports
@dataclass
class ScanResult:
    files: List[DatasetFile] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)     # unsupported types, listed so nothing vanishes silently
    pending: List[str] = field(default_factory=list)     # modified too recently: probably still being copied


@dataclass
class ChangeReport:
    changed: bool
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)


@dataclass
class BuildReport:
    text_docs: int = 0
    chat_examples: int = 0
    identity_examples: int = 0
    duplicates: int = 0
    bad_records: int = 0
    text_chars: int = 0
    errors: Dict[str, str] = field(default_factory=dict)         # file -> why it could not be read
    bad_examples: Dict[str, str] = field(default_factory=dict)    # file -> first bad-row reason
    text_corpus: Optional[str] = None
    chat_corpus: Optional[str] = None
    text_sha: Optional[str] = None
    chat_sha: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class _HashedWriter:
    """Write to a temp file while hashing, then atomically replace the target."""

    def __init__(self, target: Path):
        self.target = target
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, self._tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".corpus_tmp_")
        self._f = os.fdopen(fd, "wb")
        self._h = hashlib.sha256()
        self.count = 0

    def write(self, s: str) -> None:
        b = s.encode("utf-8")
        self._h.update(b)
        self._f.write(b)

    def finish(self) -> Optional[str]:
        self._f.close()
        if self.count == 0:
            try:
                os.remove(self._tmp)
            except OSError:
                pass
            if self.target.exists():
                self.target.unlink()
            return None
        os.replace(self._tmp, self.target)
        return self._h.hexdigest()

    def abort(self) -> None:
        try:
            self._f.close()
            os.remove(self._tmp)
        except OSError:
            pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _content_key(kind: str, payload) -> str:
    blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1((kind + "\0" + blob).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- folder
class DatasetFolder:
    def __init__(self, root: Path, work_dir: Path, settle_s: float = 5.0, max_file_mb: float = 256,
                 max_doc_chars: int = 1_000_000, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.work_dir = Path(work_dir)
        self.settle_s = settle_s
        self.max_file_bytes = int(max_file_mb * 1024 * 1024) if max_file_mb else 0
        self.max_doc_chars = max_doc_chars
        self._clock = clock
        self._lock = threading.RLock()
        self.state_path = self.work_dir / "state.json"

    @classmethod
    def from_config(cls, cfg) -> "DatasetFolder":
        al = cfg.section("autolearn")
        return cls(cfg.datasets_dir, cfg.data_dir / "datasets", settle_s=float(al.get("settle_s", 5)),
                   max_file_mb=float(al.get("max_file_mb", 256)), max_doc_chars=int(al.get("max_doc_chars", 1_000_000)))

    # -- layout ---------------------------------------------------------------------
    def corpus_path(self, kind: str) -> Path:
        return self.work_dir / kind / "corpus.txt"

    def shards_dir(self, kind: str) -> Path:
        return self.work_dir / kind / "shards"

    def ensure_layout(self) -> None:
        for sub in ("", "text", "chat", "_sources", "_eval"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # -- state ------------------------------------------------------------------------
    def _load_state(self) -> dict:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            if isinstance(st, dict):
                st.setdefault("fingerprints", {})
                st.setdefault("built", {})
                st.setdefault("trained", {})
                return st
        except (OSError, json.JSONDecodeError):
            pass
        return {"fingerprints": {}, "built": {}, "trained": {}}

    def _save_state(self, st: dict) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.work_dir), prefix=".state_tmp_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        os.replace(tmp, self.state_path)

    # -- scanning ----------------------------------------------------------------------
    def scan(self) -> ScanResult:
        result = ScanResult()
        if not self.root.is_dir():
            return result
        now = self._clock()
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")))
            for name in sorted(filenames):
                if name.startswith((".", "_")):
                    continue
                full = Path(dirpath) / name
                rel = full.relative_to(self.root).as_posix()
                if rel.lower() == "readme.md":   # the folder's own instructions are not training data
                    continue
                if full.suffix.lower() not in SUPPORTED_EXTS:
                    result.ignored.append(rel)
                    continue
                try:
                    st = full.stat()
                except OSError:
                    continue
                if self.settle_s > 0 and now - st.st_mtime < self.settle_s:
                    result.pending.append(rel)
                result.files.append(DatasetFile(rel, full, st.st_size, st.st_mtime_ns, _hint_for(rel)))
        return result

    def check(self, scan: Optional[ScanResult] = None) -> ChangeReport:
        """Cheap: compares file sizes/mtimes with the last rebuild. Never reads file contents."""
        scan = scan or self.scan()
        with self._lock:
            prev = self._load_state()["fingerprints"]
        cur = {f.rel: f for f in scan.files}
        added = sorted(r for r in cur if r not in prev)
        removed = sorted(r for r in prev if r not in cur)
        modified = sorted(r for r, f in cur.items()
                          if r in prev and (prev[r].get("size") != f.size or prev[r].get("mtime_ns") != f.mtime_ns))
        return ChangeReport(bool(added or removed or modified), added, modified, removed, list(scan.pending))

    # -- building ------------------------------------------------------------------------
    def rebuild(self, scan: Optional[ScanResult] = None, extra_chat: Optional[List[dict]] = None) -> BuildReport:
        """Regenerate both corpora from the whole folder. `extra_chat` (e.g. identity examples)
        is added only when the folder has real conversations of its own, so a text-only dataset
        never gets turned into a chat model by a handful of built-in examples."""
        scan = scan or self.scan()
        with self._lock:
            report = BuildReport()
            text_w = _HashedWriter(self.corpus_path("text"))
            chat_w = _HashedWriter(self.corpus_path("chat"))
            manifest_path = self.work_dir / "manifest.jsonl"
            seen: set = set()
            manifest: List[dict] = []
            fingerprints: Dict[str, dict] = {}
            try:
                for df in scan.files:
                    entry = {"file": df.rel, "size": df.size, "kind_hint": df.hint,
                             "text_docs": 0, "chat_examples": 0, "bad_records": 0}
                    fingerprints[df.rel] = {"size": df.size, "mtime_ns": df.mtime_ns}
                    try:
                        entry["sha256"] = sha256_file(df.path)
                        fingerprints[df.rel]["sha256"] = entry["sha256"]
                        for kind, payload in iter_file(df, self.max_file_bytes, self.max_doc_chars):
                            if kind == "bad":
                                entry["bad_records"] += 1
                                report.bad_records += 1
                                report.bad_examples.setdefault(df.rel, str(payload))
                                continue
                            key = _content_key(kind, payload)
                            if key in seen:
                                report.duplicates += 1
                                continue
                            seen.add(key)
                            if kind == "text":
                                text_w.write(f"<|doc:{_marker_name(df.rel, entry['text_docs'])}|>\n{payload}\n\n")
                                text_w.count += 1
                                entry["text_docs"] += 1
                                report.text_chars += len(payload)
                            else:
                                chat_w.write("<|doc|>\n" + render_conversation(_clean_messages(payload)) + "\n")
                                chat_w.count += 1
                                entry["chat_examples"] += 1
                    except UnsupportedFile as e:
                        report.errors[df.rel] = str(e)
                        entry["error"] = str(e)
                    except Exception as e:  # noqa: BLE001 -- one malformed file (a corrupt PDF raises
                        # library-specific errors) must never take the whole background job down
                        report.errors[df.rel] = f"{type(e).__name__}: {e}"
                        entry["error"] = report.errors[df.rel]
                    manifest.append(entry)

                if chat_w.count > 0 and extra_chat:
                    for rec in extra_chat:
                        msgs = normalize_messages(rec["messages"])
                        if not msgs:
                            continue
                        key = _content_key("chat", msgs)
                        if key in seen:
                            continue
                        seen.add(key)
                        chat_w.write("<|doc|>\n" + render_conversation(_clean_messages(msgs)) + "\n")
                        chat_w.count += 1
                        report.identity_examples += 1
                report.text_docs, report.chat_examples = text_w.count, chat_w.count - report.identity_examples
                report.text_sha = text_w.finish()
                report.chat_sha = chat_w.finish()
            except BaseException:
                text_w.abort()
                chat_w.abort()
                raise
            if report.text_sha:
                report.text_corpus = str(self.corpus_path("text"))
            if report.chat_sha:
                report.chat_corpus = str(self.corpus_path("chat"))

            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in manifest:
                    f.write(json.dumps({**entry, "built_at": self._clock()}, ensure_ascii=False) + "\n")
            st = self._load_state()
            st["fingerprints"] = fingerprints
            st["built"] = {"text": report.text_sha, "chat": report.chat_sha}
            st["built_at"] = self._clock()
            st["last_report"] = {k: v for k, v in report.to_dict().items() if k not in ("text_sha", "chat_sha")}
            self._save_state(st)
            return report

    # -- what still needs training --------------------------------------------------------
    def needs_training(self, kind: str) -> bool:
        with self._lock:
            st = self._load_state()
        built = st["built"].get(kind)
        return bool(built) and built != st["trained"].get(kind)

    def mark_trained(self, kind: str) -> None:
        with self._lock:
            st = self._load_state()
            st["trained"][kind] = st["built"].get(kind)
            st["trained_at"] = self._clock()
            self._save_state(st)

    def built_sha(self, kind: str) -> Optional[str]:
        with self._lock:
            return self._load_state()["built"].get(kind)

    def get_meta(self, key: str):
        """Small bits of caller-owned bookkeeping kept in the state file (e.g. a signature of the
        built-in identity examples, so changing them in the config triggers a rebuild)."""
        with self._lock:
            return self._load_state().get("meta", {}).get(key)

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            st = self._load_state()
            st.setdefault("meta", {})[key] = value
            self._save_state(st)

    def status(self) -> dict:
        """Everything a status screen needs, without reading file contents."""
        scan = self.scan()
        change = self.check(scan)
        with self._lock:
            st = self._load_state()
        by_hint = {"text": 0, "chat": 0, "unsorted": 0}
        for f in scan.files:
            by_hint[f.hint or "unsorted"] += 1
        return {
            "folder": str(self.root),
            "exists": self.root.is_dir(),
            "files": len(scan.files),
            "files_by_folder": by_hint,
            "total_mb": round(sum(f.size for f in scan.files) / (1024 * 1024), 3),
            "ignored": scan.ignored[:50],
            "pending_files": scan.pending,
            "changed_since_last_build": change.changed,
            "added": change.added, "modified": change.modified, "removed": change.removed,
            "text_needs_training": bool(st["built"].get("text")) and st["built"].get("text") != st["trained"].get("text"),
            "chat_needs_training": bool(st["built"].get("chat")) and st["built"].get("chat") != st["trained"].get("chat"),
            "last_build": st.get("last_report"),
            "built_at": st.get("built_at"),
            "trained_at": st.get("trained_at"),
        }


def _marker_name(rel: str, index: int) -> str:
    safe = rel.replace("|", "_").replace(">", "_").replace("\n", " ")
    return safe if index == 0 else f"{safe}#{index}"


def _clean_messages(messages: List[dict]) -> List[dict]:
    return [{"role": m["role"], "content": clean_text(m["content"])} for m in messages]
