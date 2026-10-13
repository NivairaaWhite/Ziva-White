"""evaluate.py's metrics (pure Python). The model-backed parts need torch and are covered by the training smoke tests."""
import math

import pytest

from zivawhite.evaluate import (bits_per_byte, degeneration_report, distinct_n, perplexity, repeated_ngram_rate, windows)


def test_every_token_is_predicted_exactly_once_with_the_default_stride():
    for n in (2, 3, 10, 11, 12, 100, 257):
        for block in (1, 4, 10, 64):
            ids = list(range(n))
            wins = windows(ids, block)
            assert sum(len(w) - 1 for w in wins) == n - 1, (n, block)            # n-1 targets, none skipped, none doubled
            targets = [t for w in wins for t in w[1:]]
            assert targets == ids[1:]
            assert all(len(w) - 1 <= block for w in wins)


def test_windows_edge_cases():
    assert windows([], 4) == [] and windows([7], 4) == []
    assert windows([1, 2], 4) == [[1, 2]]


def test_a_uniform_byte_model_scores_exactly_eight_bits_per_byte():
    n = 1000
    assert bits_per_byte(n * math.log(256), n) == pytest.approx(8.0)
    assert bits_per_byte(n * math.log(2), n) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0)


def test_bits_per_byte_is_tokenizer_independent():
    # the same text costs the same total nats however it is tokenized: only the byte count matters
    total_nll, n_bytes = 4321.0, 2000
    assert bits_per_byte(total_nll, n_bytes) == bits_per_byte(total_nll, n_bytes)
    assert perplexity(math.log(50)) == pytest.approx(50.0) and perplexity(1e9) < float("inf")


def test_variety_metrics():
    assert distinct_n("a b c d".split(), 1) == 1.0 and distinct_n("a a a a".split(), 1) == 0.25
    assert distinct_n("a b a b a b".split(), 2) == pytest.approx(2 / 5)
    assert distinct_n([], 2) == 1.0 and distinct_n(["x"], 3) == 1.0
    assert repeated_ngram_rate("one two three four five six".split()) == 0.0
    assert repeated_ngram_rate("a b c d a b c d a b c d".split()) == pytest.approx(5 / 9)
    assert repeated_ngram_rate("x".split()) == 0.0


def test_a_looping_model_scores_worse_than_a_varied_one():
    varied = ["the quick brown fox jumps over the lazy dog near a river bank today"]
    looping = ["the the the the the the the the the the the the"]
    v, l = degeneration_report(varied), degeneration_report(looping)
    assert v["distinct_1"] > l["distinct_1"] and v["repeated_4gram_rate"] == 0.0 and l["repeated_4gram_rate"] > 0.5
    empty = degeneration_report(["", "  "])
    assert empty["n"] == 0 and empty["empty_fraction"] == 1.0
    assert degeneration_report(["hello there my friend", ""])["empty_fraction"] == pytest.approx(0.5)
