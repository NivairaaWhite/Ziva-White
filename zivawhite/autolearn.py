"""
autolearn.py - the model learns from whatever you put in the datasets folder.

    datasets/            <- you drop files here (location: configs/ziva.yaml -> paths.datasets)
        text/            prose, articles, notes, books ......... teaches the language model
        chat/            conversations ........................ teaches it to answer
        anything.jsonl   structured files are sorted automatically (see datasets/README.md)

Every `autolearn.scan_interval_s` seconds one round runs:

    scan the folder  ->  nothing changed? done.
                     ->  files still being copied? wait.
                     ->  rebuild corpora (dedupe, clean, manifest)
                     ->  corpus changed since last training?  tokenize into shards
                     ->  train (a child process, low priority, only while the machine is idle)
                     ->  record what was trained, so a crash retries instead of forgetting

Nothing here is hardcoded: not the folder, not the topics, not the URLs, not the step
counts. They all come from configs/ziva.yaml. There is no bundled sample data and no
smoke-test corpus -- an empty folder means "nothing to learn yet", and the job says so.

Run it:
    python -m zivawhite.app autolearn              # keep watching the folder (what the daemon does)
    python -m zivawhite.app autolearn --once       # one round, then exit
    python -m zivawhite.app autolearn --status     # what is in the folder and what still needs training

Optional web intake (autolearn.web.enabled): allow-listed public pages are downloaded INTO the
datasets folder (datasets/text/web/), so they are visible, deletable and go through exactly the
same pipeline as your own files. The license allow-list in data/sources.py still gates every
page; that is a copyright safeguard, not a capability limit, and config cannot bypass it.
Topics and URLs come from your own files (autolearn.web.topics_file / urls_file); none ship.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from .config import Config, get_config
from .data.folder import DatasetFolder


# =============================================================================== throttle
@dataclass
class Throttle:
    """Decides whether it is polite to start a training run right now. Machine state is read
    through injectable probes so this can be tested without psutil."""

    only_when_idle: bool = True
    max_cpu_percent: float = 70.0
    pause_on_battery: bool = True
    cpu_probe: Optional[Callable[[], float]] = None
    battery_probe: Optional[Callable[[], Optional[bool]]] = None   # True = plugged in, False = on battery, None = no battery

    @classmethod
    def from_config(cls, cfg: Config) -> "Throttle":
        t = cfg.section("autolearn").get("throttle", {}) or {}
        return cls(bool(t.get("only_when_idle", True)), float(t.get("max_cpu_percent", 70)),
                   bool(t.get("pause_on_battery", True)))

    def _cpu(self) -> float:
        if self.cpu_probe:
            return self.cpu_probe()
        import psutil

        return psutil.cpu_percent(interval=0.5)

    def _plugged_in(self) -> Optional[bool]:
        if self.battery_probe:
            return self.battery_probe()
        try:
            import psutil

            b = psutil.sensors_battery()
        except (ImportError, AttributeError, OSError):
            return None
        return None if b is None else bool(b.power_plugged)

    def ok(self) -> Tuple[bool, str]:
        if self.pause_on_battery and self._plugged_in() is False:
            return False, "on battery power"
        if self.only_when_idle:
            try:
                cpu = self._cpu()
            except Exception:  # noqa: BLE001 - a failing probe must never block learning forever
                return True, ""
            if cpu > self.max_cpu_percent:
                return False, f"machine busy (CPU {cpu:.0f}% > {self.max_cpu_percent:.0f}%)"
        return True, ""


# ============================================================================== subprocess
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def run_command(argv: Sequence[str], cwd: str, low_priority: bool = True,
                stop: Optional[threading.Event] = None, poll_s: float = 0.5,
                on_output: Optional[Callable[[str], None]] = None) -> int:
    """Run a training command as a child process. A child, not in-process: a crash or an
    out-of-memory kill takes down only the run (and is reported), it can be stopped on demand,
    and it can run at below-normal OS priority so your own work stays responsive. With
    `on_output`, the child's output is forwarded line by line (so it reaches the rotated daemon log)."""
    kwargs: dict = {"cwd": cwd, "stdin": subprocess.DEVNULL}
    if on_output is not None:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    if low_priority:
        if os.name == "nt":
            kwargs["creationflags"] = BELOW_NORMAL_PRIORITY_CLASS
        else:
            kwargs["preexec_fn"] = lambda: os.nice(10)
    proc = subprocess.Popen(list(argv), **kwargs)
    reader = None
    if on_output is not None:
        def pump() -> None:
            for line in proc.stdout:           # ends when the child closes its output
                on_output(line.rstrip())
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
    try:
        while True:
            try:
                return proc.wait(timeout=poll_s)
            except subprocess.TimeoutExpired:
                if stop is not None and stop.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    return -15
    finally:
        if proc.poll() is None:
            proc.kill()
        if reader is not None:
            reader.join(timeout=5)


# ================================================================================ learner
@dataclass
class LearnReport:
    actions: List[str] = field(default_factory=list)
    trained: List[str] = field(default_factory=list)
    waiting: Optional[str] = None
    errors: List[str] = field(default_factory=list)         # unreadable files, failed rounds
    train_failed: bool = False

    @property
    def idle(self) -> bool:
        return not (self.actions or self.trained or self.waiting or self.errors)


class DatasetLearner:
    def __init__(self, cfg: Optional[Config] = None, folder: Optional[DatasetFolder] = None, identity=None,
                 runner: Optional[Callable[..., int]] = None, throttle: Optional[Throttle] = None,
                 log: Callable[[str], None] = print, stop: Optional[threading.Event] = None,
                 python: Optional[str] = None, shard_builder: Optional[Callable[..., Tuple[int, int]]] = None,
                 on_trained: Optional[Callable[[], None]] = None):
        self.cfg = cfg or get_config()
        self.al = self.cfg.section("autolearn")
        self.folder = folder or DatasetFolder.from_config(self.cfg)
        if identity is None:
            from .identity import Identity

            identity = Identity.from_config(self.cfg)
        self.identity = identity
        self.runner = runner or self._run_logged
        self.on_trained = on_trained            # e.g. Runtime.reload_model: use the new weights without a restart
        self.throttle = throttle or Throttle.from_config(self.cfg)
        self.log = log
        self.stop = stop or threading.Event()
        self.python = python or sys.executable
        self._build_shards = shard_builder
        self._too_small: dict = {}          # kind -> corpus hash already judged too small (not re-tokenized every round)
        self._last_waiting: Optional[str] = None   # so a standing "waiting for more data" is logged once, not every round
        self._lock = threading.Lock()
        self.report = LearnReport()
        self.state = {"state": "starting", "rounds": 0, "last_round_at": None, "last_trained_at": None,
                      "last_error": None, "waiting": None, "training": None}

    def _run_logged(self, cmd, cwd, low_priority, stop) -> int:
        return run_command(cmd, cwd, low_priority, stop, on_output=lambda line: self.log(f"    {line}"))

    # -- paths from config ---------------------------------------------------------------
    @property
    def base_run(self) -> Path:
        return self.cfg.resolve(self.al.get("base_run", "runs/ziva"))

    @property
    def chat_run(self) -> Path:
        return self.cfg.resolve(self.al.get("chat_run", "runs/ziva_behavior"))

    @staticmethod
    def _best_checkpoint(run_dir: Path) -> Optional[Path]:
        # same rule as checkpoint.resolve_serving_checkpoint: the best snapshot wins if present
        for name in ("checkpoint_best.pt", "checkpoint.pt"):
            if (run_dir / name).exists():
                return run_dir / name
        return None

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    # -- shards --------------------------------------------------------------------------
    def _shards_for(self, kind: str, corpus: Path) -> Tuple[int, int]:
        builder = self._build_shards
        if builder is None:
            from .data.ingest import build_shards as builder
        tokenizer_dir = self.al.get("tokenizer_dir")
        out = str(self.folder.shards_dir(kind))
        if kind == "chat":
            # conversation sets are small: keep the validation floor proportionate
            return builder(str(corpus), tokenizer_dir, out, doc_mode="marker", min_val_tokens=600, max_block_size=512)
        return builder(str(corpus), tokenizer_dir, out, doc_mode="marker")

    # -- command lines ---------------------------------------------------------------------
    def _common_train_args(self) -> List[str]:
        args = ["--eval_interval", str(self.al.get("eval_interval", 100)),
                "--ckpt_interval", str(self.al.get("ckpt_interval", 100))]
        if self.al.get("tokenizer_dir"):
            args += ["--tokenizer_dir", str(self.al["tokenizer_dir"])]
        if self.al.get("ewc"):
            args += ["--ewc", str(self.cfg.resolve(self.al["ewc"]))]
        args += [str(a) for a in (self.al.get("train_args") or [])]
        return args

    def _guard_args(self, prefix: str, default_epochs: float, default_patience: int) -> List[str]:
        """Overfitting guards for train.py: cap the passes over the data and stop when validation loss
        stops improving. `<prefix>_max_epochs` / `<prefix>_patience` in the autolearn config; 0 turns one off.
        (A fixed step count used to mean a small dataset was read dozens of times: memorisation.)"""
        args: List[str] = []
        epochs = float(self.al.get(f"{prefix}_max_epochs", default_epochs) or 0)
        patience = int(self.al.get(f"{prefix}_patience", default_patience) or 0)
        if epochs > 0:
            args += ["--max_epochs", f"{epochs:g}"]
        if patience > 0:
            args += ["--patience", str(patience)]
        return args

    def base_command(self, shards: Path, steps: int) -> List[str]:
        return [self.python, "-m", "zivawhite.train", "--data_dir", str(shards), "--out_dir", str(self.base_run),
                "--extra_steps", str(steps), "--resume", "--schedule", "constant", "--reset_best",
                *self._guard_args("text", 16, 5), *self._common_train_args()]

    def chat_command(self, shards: Path, base_ckpt: Path, steps: int) -> List[str]:
        lr = float(self.al.get("chat_lr", 1e-4))
        return [self.python, "-m", "zivawhite.train", "--data_dir", str(shards), "--out_dir", str(self.chat_run),
                "--init_from", str(base_ckpt), "--max_steps", str(steps), "--behavior_tuned",
                "--max_lr", str(lr), "--min_lr", str(lr / 10), "--warmup_steps", "10",
                *self._guard_args("chat", 3, 3),
                *(["--dropout", f"{float(self.al.get('chat_dropout', 0.1)):g}"] if float(self.al.get("chat_dropout", 0.1) or 0) > 0 else []),
                *self._common_train_args()]

    # -- one training step ------------------------------------------------------------------
    def _train(self, label: str, cmd: List[str]) -> bool:
        ok, why = self.throttle.ok()
        if not ok:
            self.report.waiting = f"not training {label} yet: {why}"
            self._set(state="throttled", waiting=self.report.waiting)
            return False
        self._set(state=f"training {label}", training=label, waiting=None)
        self.log(f"[autolearn] training {label}")
        rc = self.runner(cmd, str(self.cfg.root), bool((self.al.get("throttle") or {}).get("low_priority", True)),
                         self.stop)
        self._set(training=None)
        if rc != 0:
            msg = ("stopped before it finished" if self.stop.is_set()
                   else f"training {label} failed (exit code {rc}); see the log above. It will be retried.")
            self.report.errors.append(msg)
            self.report.train_failed = True
            self._set(last_error=msg)
            self.log(f"[autolearn] {msg}")
            return False
        return True

    def _prepare(self, kind: str) -> Optional[Path]:
        """Tokenize `kind`'s corpus into shards. None = nothing to train on (yet), with the reason
        in self.report.waiting."""
        built = self.folder.built_sha(kind)
        corpus = self.folder.corpus_path(kind)
        if not built or not corpus.exists():
            return None
        if self._too_small.get(kind) == built:
            return None
        n_train, n_val = self._shards_for(kind, corpus)
        need = int(self.al.get("min_chat_tokens", 2000) if kind == "chat" else self.al.get("min_train_tokens", 8192))
        if n_train < need:
            self._too_small[kind] = built
            self.report.waiting = (f"{kind} data is only {n_train:,} tokens; the minimum is {need:,} "
                                   f"(autolearn.min_{'chat' if kind == 'chat' else 'train'}_tokens). "
                                   f"Add more {kind} data and it will train automatically.")
            return None
        self._too_small.pop(kind, None)
        self.log(f"[autolearn] {kind}: {n_train:,} train / {n_val:,} val tokens")
        return self.folder.shards_dir(kind)

    # -- one round ---------------------------------------------------------------------------
    def run_once(self) -> LearnReport:
        self.report = report = LearnReport()
        folder = self.folder
        try:
            folder.ensure_layout()
            scan = folder.scan()
            if scan.pending:
                report.waiting = f"waiting for {len(scan.pending)} file(s) to finish copying: {', '.join(scan.pending[:3])}"
                self._set(state="waiting", waiting=report.waiting)
                return report

            extra = self.identity.training_records() if self.identity else []
            extra_sig = hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()
            if folder.check(scan).changed or (scan.files and folder.get_meta("identity_sig") != extra_sig):
                rep = folder.rebuild(scan, extra_chat=extra)
                folder.set_meta("identity_sig", extra_sig)
                report.actions.append(
                    f"rebuilt corpora from {len(scan.files)} file(s): {rep.text_docs} text document(s), "
                    f"{rep.chat_examples} conversation(s)" + (f" (+{rep.identity_examples} identity)" if rep.identity_examples else ""))
                if rep.bad_records:
                    report.actions.append(f"skipped {rep.bad_records} unrecognised row(s); first per file: {rep.bad_examples}")
                if rep.duplicates:
                    report.actions.append(f"dropped {rep.duplicates} exact duplicate(s)")
                for rel, why in rep.errors.items():
                    report.errors.append(f"{rel}: {why}")
                for line in report.actions + report.errors:
                    self.log(f"[autolearn] {line}")

            text_todo, chat_todo = folder.needs_training("text"), folder.needs_training("chat")
            if not (text_todo or chat_todo):
                self._set(state="idle", waiting=None)
                return report

            has_text, has_chat = bool(folder.built_sha("text")), bool(folder.built_sha("chat"))
            base_ckpt = self._best_checkpoint(self.base_run)
            trained_base = False

            # ---- base language model ------------------------------------------------------------
            if text_todo or (chat_todo and base_ckpt is None):
                # no text at all but conversations exist: learn the language from the conversations
                kind = "text" if has_text else "chat"
                shards = self._prepare(kind)
                if shards is not None:
                    steps = int(self.al.get("text_train_steps", 200) if base_ckpt else self.al.get("initial_text_steps", 1000))
                    if self._train(f"base model ({kind} data)", self.base_command(shards, steps)):
                        trained_base = True
                        report.trained.append("text" if kind == "text" else "language (from conversations)")
                        if kind == "text":
                            folder.mark_trained("text")
                        base_ckpt = self._best_checkpoint(self.base_run)
                        self._set(last_trained_at=time.time())

            # ---- conversation-tuned model: always warm-started from the current base --------------
            if has_chat and base_ckpt is not None and (chat_todo or trained_base) and not report.train_failed:
                shards = self._prepare("chat")
                if shards is not None:
                    steps = int(self.al.get("chat_train_steps", 100))
                    if self._train("conversation model", self.chat_command(shards, base_ckpt, steps)):
                        folder.mark_trained("chat")
                        report.trained.append("chat")
                        self._set(last_trained_at=time.time())
            elif chat_todo and base_ckpt is None and not report.waiting and not report.train_failed:
                report.waiting = "conversation data is ready; waiting for a base model to fine-tune"

            if report.trained:
                self.log(f"[autolearn] trained: {', '.join(report.trained)}")
                if self.on_trained is not None:
                    try:
                        self.on_trained()
                    except Exception as e:  # noqa: BLE001 - a failed reload must not undo a successful training
                        self.log(f"[autolearn] trained, but loading the new model failed: {type(e).__name__}: {e}")
                else:
                    self.log("[autolearn] restart the server to use the new model")
            self._set(state="waiting" if report.waiting else "idle", waiting=report.waiting)
        except Exception as e:  # noqa: BLE001 - a bad round must never kill the background job
            msg = f"{type(e).__name__}: {e}"
            report.errors.append(msg)
            self._set(state="error", last_error=msg)
            self.log(f"[autolearn] round failed: {msg}")
        finally:
            if report.waiting != self._last_waiting:
                self._last_waiting = report.waiting
                if report.waiting:
                    self.log(f"[autolearn] {report.waiting}")
            self._set(rounds=self.state["rounds"] + 1, last_round_at=time.time())
        return report

    def run_forever(self, stop: Optional[threading.Event] = None) -> None:
        if stop is not None:
            self.stop = stop
        interval = max(1.0, float(self.al.get("scan_interval_s", 30)))
        web = WebIntake.from_config(self.cfg, self.log)
        self.log(f"[autolearn] watching {self.folder.root} every {interval:.0f}s")
        while not self.stop.is_set():
            if web is not None:
                try:
                    web.maybe_run()
                except Exception as e:  # noqa: BLE001
                    self.log(f"[autolearn] web intake failed: {type(e).__name__}: {e}")
            self.run_once()
            self.stop.wait(interval)
        self._set(state="stopped")


# ================================================================================= web intake
def _web_connectors():
    from .connectors import WebFetchConnector, WebSearchConnector

    return WebSearchConnector(), WebFetchConnector()


def _read_lines(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _topic_stream(topics_file: Optional[str]) -> Iterator[str]:
    """Topics come from YOUR file, one per line. There is no built-in topic list."""
    topics = _read_lines(topics_file)
    return itertools.cycle(topics) if topics else iter(())


def _queue_document(queue_dir: str, url: str, text: str) -> None:
    os.makedirs(queue_dir, exist_ok=True)
    path = os.path.join(queue_dir, "autolearn.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "text": text, "queued_at": time.time()}, ensure_ascii=False) + "\n")


def run_search_cycle(search, fetch, topics: Iterator[str], queue_dir: str, n_topics: int, max_chars: int,
                     pages_per_topic: int = 2) -> int:
    """One cycle of search -> allow-listed filter -> fetch -> queue. Returns documents queued."""
    from .data.sources import is_allow_listed

    queued, seen = 0, set()
    for _ in range(n_topics):
        topic = next(topics, None)
        if topic is None or topic in seen:     # fewer topics than slots: don't refetch the same one
            break
        seen.add(topic)
        result = search.call(topic, max_results=5)
        if not result.ok:
            print(f"[autolearn] search failed for {topic!r}: {result.error}")
            continue
        urls = [u for u in result.meta.get("urls", []) if u and is_allow_listed(u)]
        if not urls:
            print(f"[autolearn] {topic!r}: no allow-listed results (see data/sources.py:REGISTRY)")
            continue
        for url in urls[:pages_per_topic]:
            page = fetch.call(url, max_chars=max_chars)
            if not page.ok or not page.text:
                continue
            _queue_document(queue_dir, url, page.text)
            queued += 1
            print(f"[autolearn] queued {url}")
    return queued


def run_bulk_cycle(fetch_urls: List[str], queue_dir: str, max_chars: int, fetch=None) -> int:
    """Fetch a fixed URL list straight into the queue (works with no search backend). `fetch`
    is injectable so tests need no network."""
    from .data.sources import is_allow_listed

    if fetch is None:
        fetch = _web_connectors()[1]
    queued = 0
    for url in fetch_urls:
        if not is_allow_listed(url):
            print(f"[autolearn] skipping {url} -- not in data/sources.py:REGISTRY")
            continue
        page = fetch.call(url, max_chars=max_chars)
        if not page.ok or not page.text:
            continue
        _queue_document(queue_dir, url, page.text)
        queued += 1
        print(f"[autolearn] queued {url}")
    return queued


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def export_queue_to_dataset(queue_dir: str, datasets_dir: Path) -> int:
    """Move queued web pages (the router's opportunistic queue and the web intake's) into
    datasets/text/web/ as ordinary files -- visible, auditable, deletable -- with a provenance
    line (url, license) in datasets/_sources/web_provenance.jsonl. Returns files written."""
    import glob
    from urllib.parse import urlparse

    from .data.sources import is_allow_listed, source_for

    out_dir = Path(datasets_dir) / "text" / "web"
    prov = Path(datasets_dir) / "_sources" / "web_provenance.jsonl"
    n = 0
    for qf in glob.glob(os.path.join(queue_dir, "*.jsonl")):
        with open(qf, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url, text = rec.get("url", ""), rec.get("text", "")
            if not text or not is_allow_listed(url):      # the license gate applies again on export
                continue
            host = _SLUG_RE.sub("-", (urlparse(url).hostname or "page").lower()).strip("-")
            name = f"{host}-{hashlib.sha1(url.encode()).hexdigest()[:10]}.txt"
            target = out_dir / name
            if target.exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            src = source_for(url)
            prov.parent.mkdir(parents=True, exist_ok=True)
            with open(prov, "a", encoding="utf-8") as pf:
                pf.write(json.dumps({"file": f"text/web/{name}", "url": url, "source": src.key if src else "unknown",
                                     "license": src.license if src else "unknown", "fetched_at": rec.get("queued_at")}) + "\n")
            n += 1
        os.remove(qf)
    return n


class WebIntake:
    """Optional. Pulls allow-listed public pages into datasets/text/web/ on a timer."""

    def __init__(self, cfg: Config, log: Callable[[str], None] = print, search=None, fetch=None):
        self.cfg = cfg
        self.web = cfg.section("autolearn").get("web", {}) or {}
        self.log = log
        self._search, self._fetch = search, fetch
        self._last: Optional[float] = None
        self._topics: Optional[Iterator[str]] = None

    @classmethod
    def from_config(cls, cfg: Config, log: Callable[[str], None] = print) -> Optional["WebIntake"]:
        web = cfg.section("autolearn").get("web", {}) or {}
        return cls(cfg, log) if web.get("enabled") else None

    def drain_queue(self) -> int:
        """Pages the assistant fetched while answering you (data/queue) become dataset files too."""
        exported = export_queue_to_dataset(str(self.cfg.data_dir / "queue"), self.cfg.datasets_dir)
        if exported:
            self.log(f"[autolearn] added {exported} page(s) to {self.cfg.datasets_dir / 'text' / 'web'}")
        return exported

    def maybe_run(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        if self._last is not None and now - self._last < float(self.web.get("interval_s", 3600)):
            return self.drain_queue()
        self._last = now
        return self.run()

    def run(self) -> int:
        search, fetch = (self._search, self._fetch) if self._fetch else _web_connectors()
        queue_dir = str(self.cfg.data_dir / "queue")
        max_chars = int(self.web.get("max_chars", 4000))
        topics_file = str(self.cfg.resolve(self.web["topics_file"])) if self.web.get("topics_file") else None
        urls_file = str(self.cfg.resolve(self.web["urls_file"])) if self.web.get("urls_file") else None
        if self._topics is None:
            self._topics = _topic_stream(topics_file)
        has_search = search is not None and search.available()
        urls = _read_lines(urls_file)
        if has_search:
            run_search_cycle(search, fetch, self._topics, queue_dir, int(self.web.get("topics_per_cycle", 5)), max_chars)
        if urls:
            run_bulk_cycle(urls, queue_dir, max_chars, fetch=fetch)
        if not urls and not (has_search and _read_lines(topics_file)):
            self.log(f"[autolearn] web intake is on but has nothing to fetch: put URLs in {urls_file}, or topics in "
                     f"{topics_file} with a search backend (SEARXNG_URL) in .env")
        exported = export_queue_to_dataset(queue_dir, self.cfg.datasets_dir)
        if exported:
            self.log(f"[autolearn] web intake added {exported} page(s) to {self.cfg.datasets_dir / 'text' / 'web'}")
        return exported


# ============================================================================== legacy helper
def retrain(checkpoint_dir: str, shards_out: str, tokenizer_dir: Optional[str], extra_steps: int,
            ewc: Optional[str] = None, ewc_lambda: float = 1000.0) -> None:
    """In-process continue-training (kept for the queue -> `ingest` workflow and its tests).
    The datasets job uses DatasetLearner, which runs training as a managed child process."""
    from .checkpoint import load_checkpoint

    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, map_location="cpu")["step"]
    argv = ["train.py", "--data_dir", shards_out, "--out_dir", checkpoint_dir,
            "--max_steps", str(start_step + extra_steps), "--resume", "--schedule", "constant", "--reset_best"]
    if tokenizer_dir:
        argv += ["--tokenizer_dir", tokenizer_dir]
    if ewc:
        argv += ["--ewc", ewc, "--ewc_lambda", str(ewc_lambda)]
    sys.argv = argv
    from . import train as train_mod

    train_mod.main()


# ==================================================================================== CLI
def print_status(cfg: Config) -> None:
    st = DatasetFolder.from_config(cfg).status()
    print(f"datasets folder : {st['folder']}" + ("" if st["exists"] else "   (does not exist yet; it is created on first run)"))
    fb = st["files_by_folder"]
    print(f"files           : {st['files']}  ({fb['text']} in text/, {fb['chat']} in chat/, {fb['unsorted']} sorted automatically)  {st['total_mb']} MB")
    if st["ignored"]:
        print(f"ignored         : {', '.join(st['ignored'][:8])}  (unsupported type; see datasets/README.md)")
    if st["pending_files"]:
        print(f"still copying   : {', '.join(st['pending_files'][:5])}")
    print(f"changed         : {st['changed_since_last_build']}  (added {len(st['added'])}, modified {len(st['modified'])}, removed {len(st['removed'])})")
    print(f"needs training  : text={st['text_needs_training']}  chat={st['chat_needs_training']}")
    lb = st.get("last_build")
    if lb:
        print(f"last build      : {lb.get('text_docs', 0)} text doc(s), {lb.get('chat_examples', 0)} conversation(s), "
              f"{lb.get('duplicates', 0)} duplicate(s) dropped, {len(lb.get('errors', {}))} unreadable file(s)")
        for rel, why in (lb.get("errors") or {}).items():
            print(f"   ! {rel}: {why}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="run one scan/train round and exit")
    p.add_argument("--status", action="store_true", help="show what is in the datasets folder and what still needs training")
    p.add_argument("--no-throttle", action="store_true", help="train even if the machine is busy or on battery")
    args = p.parse_args(argv)

    cfg = get_config()
    if args.status:
        print_status(cfg)
        return
    learner = DatasetLearner(cfg, throttle=Throttle(only_when_idle=False, pause_on_battery=False) if args.no_throttle else None)
    if args.once:
        learner.folder.ensure_layout()
        report = learner.run_once()
        if report.idle:
            print("[autolearn] nothing to do: no new or changed data in the datasets folder." if learner.folder.scan().files
                  else f"[autolearn] the datasets folder is empty. Put training files in {learner.folder.root} (see datasets/README.md).")
        sys.exit(1 if (report.errors or report.train_failed) else 0)
    try:
        learner.run_forever()
    except KeyboardInterrupt:
        print("\n[autolearn] stopped")


if __name__ == "__main__":
    main()
