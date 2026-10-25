"""The datasets-folder learner: what it trains, when, in what order, and what it never does.
A fake runner stands in for train.py and a fake shard builder for tokenization, so this is
pure Python: no torch, no network."""
import json
import os
from pathlib import Path

from helpers import make_cfg

from zivawhite.autolearn import (DatasetLearner, Throttle, WebIntake, _topic_stream, export_queue_to_dataset,
                                 run_command)
from zivawhite.connectors.base import Connector, ConnectorResult


class FakeRunner:
    """Records every training command; on success creates the checkpoint train.py would."""

    def __init__(self, results=None):
        self.calls, self.results = [], list(results or [])

    def __call__(self, cmd, cwd, low_priority, stop):
        self.calls.append(list(cmd))
        rc = self.results.pop(0) if self.results else 0
        if rc == 0:
            out = Path(cmd[cmd.index("--out_dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoint.pt").write_text("fake")
        return rc

    def flag(self, i, name):
        c = self.calls[i]
        return c[c.index(name) + 1]


def fake_shards(n_train=50_000, n_val=2_000):
    return lambda *a, **k: (n_train, n_val)


def learner(tmp_path, runner=None, shards=None, throttle=None, cfg=None, **cfg_over):
    cfg = cfg or make_cfg(tmp_path, **cfg_over)
    return DatasetLearner(cfg, runner=runner or FakeRunner(), shard_builder=shards or fake_shards(),
                          throttle=throttle, log=lambda *_: None)


def put(cfg, rel, text):
    p = cfg.datasets_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


CHAT_ROW = json.dumps({"prompt": "hi there", "response": "hello, how can I help?"})


# ------------------------------------------------------------------------------- basics
def test_empty_folder_does_nothing_and_does_not_invent_data(tmp_path):
    lr = learner(tmp_path)
    rep = lr.run_once()
    assert rep.idle and lr.runner.calls == []
    assert (lr.cfg.datasets_dir / "text").is_dir() and (lr.cfg.datasets_dir / "chat").is_dir()   # layout created for you


def test_text_data_trains_the_base_model_once(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "text/story.txt", "a long story " * 50)
    rep = lr.run_once()
    assert rep.trained == ["text"] and len(lr.runner.calls) == 1
    run = lr.runner
    assert run.flag(0, "--extra_steps") == "1000"                # no model yet -> the initial run
    assert "--resume" in run.calls[0] and "--reset_best" in run.calls[0] and "--behavior_tuned" not in run.calls[0]
    assert run.flag(0, "--out_dir").endswith(os.path.join("runs", "ziva"))
    assert run.flag(0, "--data_dir") == str(lr.folder.shards_dir("text"))

    assert lr.run_once().idle and len(run.calls) == 1            # nothing changed -> nothing retrained


def test_later_text_changes_use_the_smaller_continue_step_count(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "a.txt", "first document " * 40)
    lr.run_once()
    put(lr.cfg, "b.txt", "second document " * 40)
    lr.run_once()
    assert lr.runner.flag(1, "--extra_steps") == "200"


def test_step_counts_and_paths_come_from_config_not_code(tmp_path):
    lr = learner(tmp_path, autolearn={"initial_text_steps": 7, "base_run": "my_runs/base",
                                       "train_args": ["--tier", "small"], "tokenizer_dir": "tok"},
                 paths={"datasets": "my_data"})
    put(lr.cfg, "x.txt", "words " * 100)
    lr.run_once()
    assert lr.runner.flag(0, "--extra_steps") == "7"
    assert lr.runner.flag(0, "--out_dir").endswith(os.path.join("my_runs", "base"))
    assert lr.runner.flag(0, "--tier") == "small" and lr.runner.flag(0, "--tokenizer_dir") == "tok"
    assert (tmp_path / "my_data" / "text").is_dir()


# ------------------------------------------------------------------------------ chat flow
def test_chat_data_fine_tunes_from_the_base_and_never_overwrites_it(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "text/book.txt", "prose " * 200)
    put(lr.cfg, "chat/talk.jsonl", CHAT_ROW)
    rep = lr.run_once()
    assert rep.trained == ["text", "chat"]
    base, chat = lr.runner.calls
    assert "--behavior_tuned" in chat and "--init_from" in chat
    assert lr.runner.flag(1, "--init_from").endswith(os.path.join("runs", "ziva", "checkpoint.pt"))
    assert lr.runner.flag(1, "--out_dir").endswith(os.path.join("runs", "ziva_behavior"))
    assert lr.runner.flag(1, "--max_steps") == "100" and lr.runner.flag(1, "--max_lr") == "0.0001"
    assert lr.runner.flag(0, "--out_dir") != lr.runner.flag(1, "--out_dir")


def test_chat_only_data_learns_language_from_the_conversations_then_tunes(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "chat/talk.jsonl", CHAT_ROW)
    rep = lr.run_once()
    assert len(lr.runner.calls) == 2 and "--behavior_tuned" not in lr.runner.calls[0]
    assert lr.runner.flag(0, "--data_dir") == str(lr.folder.shards_dir("chat"))
    assert "chat" in rep.trained
    assert not lr.folder.needs_training("chat")


def test_only_the_changed_kind_retrains(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "text/book.txt", "prose " * 200)
    put(lr.cfg, "chat/talk.jsonl", CHAT_ROW)
    lr.run_once()
    lr.runner.calls.clear()

    put(lr.cfg, "chat/more.jsonl", json.dumps({"prompt": "bye", "response": "see you soon"}))
    lr.run_once()
    assert len(lr.runner.calls) == 1 and "--behavior_tuned" in lr.runner.calls[0]    # text untouched

    lr.runner.calls.clear()
    put(lr.cfg, "text/new.txt", "fresh prose " * 100)
    lr.run_once()
    assert len(lr.runner.calls) == 2                                                  # base moved -> chat re-tuned on top
    assert "--behavior_tuned" not in lr.runner.calls[0] and "--behavior_tuned" in lr.runner.calls[1]


def test_identity_examples_are_trained_only_alongside_real_chat_data(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "text/book.txt", "prose " * 200)
    lr.run_once()
    assert lr.folder.built_sha("chat") is None                      # a text-only dataset is not turned into a chat model

    put(lr.cfg, "chat/talk.jsonl", CHAT_ROW)
    lr.run_once()
    corpus = lr.folder.corpus_path("chat").read_text(encoding="utf-8")
    assert "Mahasvin S S" in corpus and "Snow White" in corpus      # from configs/ziva.yaml, not from code


def test_changing_the_creator_in_the_config_retrains_the_chat_model(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "chat/talk.jsonl", CHAT_ROW)
    lr.run_once()
    assert "Mahasvin S S" in lr.folder.corpus_path("chat").read_text(encoding="utf-8")

    cfg2 = make_cfg(tmp_path, identity={"creator": {"name": "Someone Else", "alias": "Another"}})
    lr2 = DatasetLearner(cfg2, runner=FakeRunner(), shard_builder=fake_shards(), log=lambda *_: None)
    lr2.run_once()
    assert "Someone Else" in lr2.folder.corpus_path("chat").read_text(encoding="utf-8")
    assert len(lr2.runner.calls) >= 1                                # the new facts were actually trained


# ----------------------------------------------------------------- waiting / failing / throttle
def test_too_little_data_waits_and_says_how_much(tmp_path):
    builds = []
    lr = learner(tmp_path, shards=lambda *a, **k: builds.append(1) or (300, 20))
    put(lr.cfg, "text/tiny.txt", "just a few words")
    rep = lr.run_once()
    assert lr.runner.calls == [] and "8,192" in rep.waiting and "300" in rep.waiting
    lr.run_once()
    lr.run_once()
    assert len(builds) == 1                                            # not re-tokenized every round while nothing changes
    assert lr.folder.needs_training("text")                            # still pending

    put(lr.cfg, "text/more.txt", "enough additional words to change the corpus " * 10)
    lr._build_shards = lambda *a, **k: builds.append(1) or (50_000, 2_000)
    assert lr.run_once().trained == ["text"] and len(builds) == 2      # more data -> trains automatically


def test_files_still_being_copied_are_waited_for(tmp_path):
    lr = learner(tmp_path, autolearn={"settle_s": 3600})
    put(lr.cfg, "text/copying.txt", "partial")
    rep = lr.run_once()
    assert "finish copying" in rep.waiting and lr.runner.calls == []


def test_a_failed_training_run_is_retried_on_the_next_round(tmp_path):
    runner = FakeRunner(results=[1, 0])
    lr = learner(tmp_path, runner=runner)
    put(lr.cfg, "text/book.txt", "prose " * 200)
    rep = lr.run_once()
    assert rep.train_failed and rep.errors and lr.folder.needs_training("text")
    rep = lr.run_once()
    assert rep.trained == ["text"] and not lr.folder.needs_training("text") and len(runner.calls) == 2


def test_an_unreadable_file_does_not_block_training_of_the_rest(tmp_path):
    lr = learner(tmp_path)
    put(lr.cfg, "text/book.txt", "prose " * 200)
    put(lr.cfg, "broken.json", "{nope")
    rep = lr.run_once()
    assert rep.trained == ["text"] and any("broken.json" in e for e in rep.errors)


def test_busy_machine_or_battery_defers_training_without_losing_it(tmp_path):
    state = {"cpu": 95.0, "plugged": False}
    throttle = Throttle(True, 70.0, True, cpu_probe=lambda: state["cpu"], battery_probe=lambda: state["plugged"])
    lr = learner(tmp_path, throttle=throttle)
    put(lr.cfg, "text/book.txt", "prose " * 200)

    rep = lr.run_once()
    assert "battery" in rep.waiting and lr.runner.calls == []
    state["plugged"] = True
    rep = lr.run_once()
    assert "busy" in rep.waiting and lr.runner.calls == []
    state["cpu"] = 10.0
    assert lr.run_once().trained == ["text"]


def test_throttle_decisions():
    assert Throttle(False, 50, False).ok() == (True, "")
    assert Throttle(True, 50, False, cpu_probe=lambda: 10).ok()[0]
    assert not Throttle(True, 50, False, cpu_probe=lambda: 90).ok()[0]
    assert Throttle(True, 50, True, cpu_probe=lambda: 10, battery_probe=lambda: None).ok()[0]    # desktop: no battery
    def boom(): raise RuntimeError("probe broke")
    assert Throttle(True, 50, False, cpu_probe=boom).ok()[0]                                      # never blocks forever


def test_stopping_mid_round_is_reported_not_treated_as_a_crash_loop(tmp_path):
    lr = learner(tmp_path, runner=FakeRunner(results=[-15]))
    put(lr.cfg, "text/book.txt", "prose " * 200)
    lr.stop.set()
    rep = lr.run_once()
    assert rep.train_failed and "stopped" in rep.errors[0]


def test_run_command_runs_and_reports_the_exit_code(tmp_path):
    import sys
    assert run_command([sys.executable, "-c", "raise SystemExit(0)"], str(tmp_path), low_priority=False) == 0
    assert run_command([sys.executable, "-c", "raise SystemExit(3)"], str(tmp_path), low_priority=True) == 3


# ----------------------------------------------------------------------------- web intake
class FakeSearch(Connector):
    name = "web_search"
    def available(self): return True
    def call(self, prompt, **kw):
        return ConnectorResult(text="s", source="web_search",
                               meta={"urls": ["https://en.wikipedia.org/wiki/Moon", "https://evil.example/x"]})


class FakeFetch(Connector):
    name = "web_fetch"
    def __init__(self): self.calls = []
    def available(self): return True
    def call(self, url, **kw):
        self.calls.append(url)
        return ConnectorResult(text=f"page text for {url}", source="web_fetch")


def test_no_topics_ship_with_the_project():
    assert list(_topic_stream(None)) == [] and list(_topic_stream("/does/not/exist.txt")) == []


def test_web_intake_is_off_by_default(tmp_path):
    assert WebIntake.from_config(make_cfg(tmp_path)) is None


def test_web_intake_writes_allow_listed_pages_into_the_datasets_folder(tmp_path):
    cfg = make_cfg(tmp_path, autolearn={"web": {"enabled": True}})
    (cfg.datasets_dir / "_sources").mkdir(parents=True)
    (cfg.datasets_dir / "_sources" / "topics.txt").write_text("# my topics\nthe moon\n")
    (cfg.datasets_dir / "_sources" / "urls.txt").write_text("https://en.wikipedia.org/wiki/Sun\nhttps://evil.example/y\n")
    fetch = FakeFetch()
    intake = WebIntake(cfg, log=lambda *_: None, search=FakeSearch(), fetch=fetch)
    n = intake.run()
    assert n == 2                                                       # moon (search) + sun (url list)
    assert all("evil.example" not in u for u in fetch.calls)           # the license gate is not configurable
    files = sorted((cfg.datasets_dir / "text" / "web").glob("*.txt"))
    assert len(files) == 2 and files[0].read_text().startswith("page text for")
    prov = [json.loads(l) for l in (cfg.datasets_dir / "_sources" / "web_provenance.jsonl").read_text().splitlines()]
    assert {p["source"] for p in prov} == {"wikipedia"} and "CC BY-SA" in prov[0]["license"]


def test_web_intake_respects_its_interval(tmp_path):
    cfg = make_cfg(tmp_path, autolearn={"web": {"enabled": True, "interval_s": 100}})
    intake = WebIntake(cfg, log=lambda *_: None, search=FakeSearch(), fetch=FakeFetch())
    runs = []
    intake.run = lambda: runs.append(1) or 0
    intake.maybe_run(now=1000.0)
    intake.maybe_run(now=1050.0)
    intake.maybe_run(now=1101.0)
    assert len(runs) == 2


def test_queue_export_refuses_non_allow_listed_urls_and_clears_the_queue(tmp_path):
    q, ds = tmp_path / "q", tmp_path / "ds"
    q.mkdir()
    (q / "web_ingest.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"url": "https://www.gutenberg.org/files/1/1-0.txt", "text": "good", "queued_at": 1},
        {"url": "https://not-listed.example/p", "text": "bad", "queued_at": 2},
    ]) + "\n")
    assert export_queue_to_dataset(str(q), ds) == 1
    assert not list(q.glob("*.jsonl")) and len(list((ds / "text" / "web").glob("*.txt"))) == 1


def test_a_standing_wait_is_logged_once_not_every_round(tmp_path):
    lines = []
    lr = DatasetLearner(make_cfg(tmp_path), runner=FakeRunner(), shard_builder=fake_shards(n_train=300), log=lines.append)
    put(lr.cfg, "text/tiny.txt", "just a few words")
    for _ in range(4):
        lr.run_once()
    assert sum("only 300 tokens" in l for l in lines) == 1


def test_training_output_is_forwarded_line_by_line(tmp_path):
    import sys
    got = []
    rc = run_command([sys.executable, "-c", "print('step 10 loss 3.2'); print('step 20 loss 3.0')"], str(tmp_path),
                     low_priority=False, on_output=got.append)
    assert rc == 0 and got == ["step 10 loss 3.2", "step 20 loss 3.0"]


def test_a_running_training_child_is_stopped_on_request(tmp_path):
    import sys
    import threading
    import time
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    t0 = time.time()
    rc = run_command([sys.executable, "-c", "import time; time.sleep(60)"], str(tmp_path), low_priority=False, stop=stop, poll_s=0.1)
    assert rc != 0 and time.time() - t0 < 10


# ------------------------------------------------------------------ priority is lowered by the parent, not in a fork hook
import os as _os  # noqa: E402
import pytest  # noqa: E402
import subprocess as _sp  # noqa: E402
import sys as _sys  # noqa: E402


def test_the_child_is_started_without_a_preexec_fn(tmp_path, monkeypatch):
    """preexec_fn runs Python between fork and exec in a multithreaded process, which can deadlock (review item)."""
    from zivawhite import autolearn

    seen = {}
    real = _sp.Popen

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)
    monkeypatch.setattr(autolearn.subprocess, "Popen", spy)
    assert autolearn.run_command([_sys.executable, "-c", "pass"], cwd=str(tmp_path), low_priority=True) == 0
    assert "preexec_fn" not in seen


@pytest.mark.skipif(_os.name == "nt", reason="POSIX niceness")
def test_low_priority_really_lowers_the_childs_niceness(tmp_path):
    from zivawhite import autolearn

    lines = []
    code = "import os, time; time.sleep(0.5); print('nice', os.nice(0))"       # read it after the parent has had time to act
    assert autolearn.run_command([_sys.executable, "-c", code], cwd=str(tmp_path), low_priority=True, on_output=lines.append) == 0
    here = _os.nice(0)
    assert lines and int(lines[-1].split()[1]) >= here + 10 or int(lines[-1].split()[1]) == 19
    lines.clear()
    assert autolearn.run_command([_sys.executable, "-c", code], cwd=str(tmp_path), low_priority=False, on_output=lines.append) == 0
    assert int(lines[-1].split()[1]) == here


# ------------------------------------------------------------------ the model keeps its vocabulary across training rounds
def _save_tokenizer_with_run(run_dir, tag="x", n=700):
    """What train.py leaves behind: <run>/tokenizer/{vocab.json,merges.txt}, next to the checkpoint."""
    import json as _j
    tdir = Path(run_dir) / "tokenizer"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "vocab.json").write_text(_j.dumps({f"{tag}{i}": i for i in range(n)}))
    (tdir / "merges.txt").write_text("#version: 0.2\n")
    return tdir


def test_first_round_uses_the_bundled_tokenizer_and_passes_no_flag(tmp_path):
    lr = learner(tmp_path)
    assert lr.tokenizer_dir() is None
    put(lr.cfg, "text/story.txt", "a long story " * 50)
    lr.run_once()
    assert "--tokenizer_dir" not in lr.runner.calls[0]


def test_later_rounds_keep_the_vocabulary_the_base_model_was_trained_with(tmp_path):
    """The reported bug: shards and training silently went back to the bundled 12k tokenizer. Now, once the base run has
    a saved tokenizer, shards are built with it and train.py is pointed at it."""
    seen = {}

    def shards(corpus, tokenizer_dir, out, **kw):
        seen.setdefault("dirs", []).append(tokenizer_dir)
        return 50_000, 2_000
    lr = learner(tmp_path, shards=shards)
    put(lr.cfg, "text/story.txt", "a long story " * 50)
    lr.run_once()                                                   # round 1: bundled, nothing saved yet
    assert seen["dirs"] == [None]
    saved = _save_tokenizer_with_run(lr.base_run)                   # ... train.py saved its tokenizer with the run
    assert lr.tokenizer_dir() == str(saved)
    put(lr.cfg, "text/more.txt", "another long story " * 50)
    lr.run_once()                                                   # round 2
    assert seen["dirs"][-1] == str(saved)                          # shards use the model's own vocabulary
    assert lr.runner.flag(1, "--tokenizer_dir") == str(saved)      # and so does training


def test_an_explicit_tokenizer_dir_always_wins_over_the_saved_one(tmp_path):
    lr = learner(tmp_path, autolearn={"tokenizer_dir": "tok"})
    _save_tokenizer_with_run(lr.base_run)
    assert lr.tokenizer_dir() == "tok"


def test_a_relative_tokenizer_dir_is_resolved_against_the_project_folder_for_shards(tmp_path):
    seen = {}
    lr = learner(tmp_path, shards=lambda corpus, tdir, out, **kw: seen.setdefault("t", tdir) and (50_000, 2_000),
                 autolearn={"tokenizer_dir": "my_tok"})
    put(lr.cfg, "text/story.txt", "a long story " * 50)
    lr.run_once()
    assert seen["t"] == str(lr.cfg.resolve("my_tok"))
    assert lr.runner.flag(0, "--tokenizer_dir") == "my_tok"        # the subprocess runs in the project folder: unchanged


# ------------------------------------------------------------------ learning a vocabulary from your own data (optional)
def test_the_tokenizer_command_learns_from_the_datasets_folder_with_a_stand_in_trainer(tmp_path):
    from types import SimpleNamespace
    from zivawhite.data.folder import DatasetFolder
    from zivawhite.data.tokenizer_train import NotEnoughText, iter_dataset_texts, train_from_datasets

    cfg = make_cfg(tmp_path)
    put(cfg, "text/a.txt", "the quick brown fox " * 400)
    put(cfg, "chat/c.jsonl", CHAT_ROW + "\n")
    folder = DatasetFolder.from_config(cfg)
    texts = list(iter_dataset_texts(folder))
    assert any("quick brown fox" in t for t in texts) and any("how can I help" in t for t in texts)   # chat turns count too

    got = {}

    def fake_trainer(it, vocab_size, out_dir):
        got["corpus"], got["vocab"], got["out"] = "".join(it), vocab_size, out_dir
        return SimpleNamespace(vocab_size=vocab_size)
    res = train_from_datasets(folder, 50, str(tmp_path / "tok"), trainer=fake_trainer)
    assert res["vocab_size"] == 50 and got["vocab"] == 50 and "quick brown fox" in got["corpus"]

    try:
        train_from_datasets(folder, 8000, str(tmp_path / "tok2"), trainer=fake_trainer)    # 8000 x 150 chars: far more than we have
        raise AssertionError("should have refused")
    except NotEnoughText as e:
        assert "8,000" in str(e) and "--force" in str(e)
    assert train_from_datasets(folder, 8000, str(tmp_path / "tok3"), force=True, trainer=fake_trainer)["vocab_size"] == 8000
