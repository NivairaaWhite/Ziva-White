import json
import os

from zivawhite.connectors.base import Connector, ConnectorResult


class FakeSearch(Connector):
    """Always returns a fixed set of result URLs, mixing allow-listed
    (gutenberg.org) and not-allow-listed (example.com) domains, so tests
    can confirm the not-allow-listed one never reaches the queue."""

    name = "web_search"

    def available(self):
        return True

    def call(self, prompt, **kw):
        return ConnectorResult(
            text="search snippet",
            source="web_search",
            meta={"urls": ["https://www.gutenberg.org/files/1/1-0.txt", "https://example.com/not-allowed"]},
        )


class FakeFetch(Connector):
    name = "web_fetch"

    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def call(self, url, **kw):
        self.calls.append(url)
        return ConnectorResult(text=f"fetched body for {url}", source="web_fetch")


def test_run_search_cycle_only_queues_allow_listed_urls(tmp_path):
    from zivawhite.autolearn import run_search_cycle

    queue_dir = str(tmp_path / "queue")
    fetch = FakeFetch()
    queued = run_search_cycle(FakeSearch(), fetch, iter(["some topic"]), queue_dir, n_topics=1, max_chars=1000)

    assert queued == 1  # only the gutenberg.org URL, example.com is filtered out
    assert fetch.calls == ["https://www.gutenberg.org/files/1/1-0.txt"]

    queue_file = os.path.join(queue_dir, "autolearn.jsonl")
    with open(queue_file) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 1
    assert records[0]["url"] == "https://www.gutenberg.org/files/1/1-0.txt"


def test_run_bulk_cycle_skips_non_allow_listed_urls(tmp_path):
    from zivawhite.autolearn import run_bulk_cycle

    queue_dir = str(tmp_path / "queue")
    fetch = FakeFetch()
    urls = ["https://www.gutenberg.org/files/2/2-0.txt", "https://not-a-real-registry-domain.example/page"]
    queued = run_bulk_cycle(urls, queue_dir, max_chars=1000, fetch=fetch)

    assert queued == 1
    assert fetch.calls == ["https://www.gutenberg.org/files/2/2-0.txt"]


def test_fold_and_retrain_end_to_end_on_a_tiny_queue(tmp_path):
    """The actual point of autolearn: queue -> fold -> shards -> a real
    (tiny) train.py run that produces a checkpoint. No network involved --
    the queue is populated directly, same shape run_bulk_cycle/
    run_search_cycle would leave it in."""
    from zivawhite.autolearn import retrain
    from zivawhite.data.ingest import build_shards, fold_queue

    queue_dir = str(tmp_path / "queue")
    os.makedirs(queue_dir, exist_ok=True)
    with open(os.path.join(queue_dir, "autolearn.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "url": "https://www.gutenberg.org/files/3/3-0.txt",
            "text": "The quick brown fox jumps over the lazy dog. " * 500,
            "queued_at": 0,
        }) + "\n")

    corpus_out = str(tmp_path / "corpus.txt")
    manifest_out = str(tmp_path / "manifest.jsonl")
    shards_out = str(tmp_path / "shards")
    out_dir = str(tmp_path / "run")

    n_folded = fold_queue(queue_dir, corpus_out, manifest_out)
    assert n_folded == 1
    assert not os.listdir(queue_dir)  # queue is cleared after folding

    n_train, n_val = build_shards(corpus_out, None, shards_out)
    assert n_train > 0

    retrain(out_dir, shards_out, None, extra_steps=2)
    assert os.path.exists(os.path.join(out_dir, "checkpoint.pt"))
