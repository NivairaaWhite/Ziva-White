#!/usr/bin/env python3
"""
online_learning_loop.py

Ties the four data pieces together into the actual management loop for
online learning:

  feedback_log.jsonl      -> incoming signal, batched
  replay_buffer_sample.jsonl -> rehearsal mix, prevents forgetting
  eval_benchmark.jsonl     -> fixed yardstick for gating updates
  (writes) drift_monitor_log.jsonl, version_registry.jsonl

WHAT'S REAL vs WHAT'S A STUB, honestly:
- Batch quality stats (accept/edit/reject rates) are computed for real from
  feedback_log.jsonl.
- Drift score is computed for real, from actual category-distribution and
  length statistics of each batch vs. the seed baseline.
- The model-quality eval (`run_eval_suite()` below) is now REAL, not a
  stub, IF you pass --checkpoint: it loads that checkpoint into an actual
  ZivaWhite instance (zivawhite/router.py) and generates a real
  response for every eval_benchmark.jsonl record. Scoring is a free,
  dependency-free lexical-overlap heuristic by default (token-F1 against
  reference_output + an acceptance_criteria keyword check). It is a crude
  signal -- good for "did this checkpoint get worse", not for ranking
  answers. (An LLM-as-judge option used to exist here; it needed the
  Anthropic/OpenAI connectors, which this project no longer has, so it was
  removed rather than left as a branch that could never run.) Run
  with no --checkpoint at all and this still degrades to the original
  honest "no real eval available" None, exactly like before -- the
  heuristic promotion fallback below is unchanged for that case.
- Promotion/rollback decisions fall back to a conservative heuristic when
  no real eval result is available, and every version_registry entry says
  so explicitly via `decision_basis`.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")   # outputs live here: drift log, registry, replay buffer
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _input_dirs():
    """Where YOUR files are looked for: <datasets folder>/_eval/ first (configs/ziva.yaml: paths.datasets),
    then this kit's data/ folder. No sample data ships in either."""
    dirs = []
    try:
        from zivawhite.config import get_config
        dirs.append(str(get_config().datasets_dir / "_eval"))
    except Exception:  # noqa: BLE001 - config unavailable: fall back to the kit's own folder
        pass
    return dirs + [DATA_DIR]

BATCH_SIZE = 8
REHEARSAL_RATIO = 0.5          # fraction of each training mix drawn from replay buffer
DRIFT_ALERT_THRESHOLD = 0.35   # above this, flag for human review regardless of quality
QUALITY_PROMOTE_THRESHOLD = 0.6  # heuristic fallback only -- see run_eval_suite()

_WORD_RE = re.compile(r"[a-z0-9']+")


def load_jsonl(name):
    for d in _input_dirs():
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


def batchify(records, size):
    for i in range(0, len(records), size):
        yield records[i:i + size]


def compute_batch_stats(batch):
    counts = Counter(r["feedback_type"] for r in batch)
    total = len(batch)
    lengths = [len(r["input"]) for r in batch]
    category_counts = Counter(r["category"] for r in batch)
    return {
        "n": total,
        "accept_rate": counts.get("accept", 0) / total,
        "edit_rate": counts.get("edit", 0) / total,
        "reject_rate": counts.get("reject", 0) / total,
        "implicit_positive_rate": counts.get("implicit_positive", 0) / total,
        "implicit_negative_rate": counts.get("implicit_negative", 0) / total,
        "avg_input_length": statistics.mean(lengths) if lengths else 0,
        "category_distribution": {k: v / total for k, v in category_counts.items()},
    }


def compute_baseline_stats(replay_buffer):
    if not replay_buffer:
        return None
    lengths = [len(" ".join(m["content"] for m in r["messages"] if m["role"] == "user")) for r in replay_buffer]
    category_counts = Counter(r["category"] for r in replay_buffer)
    total = len(replay_buffer)
    return {
        "avg_input_length": statistics.mean(lengths) if lengths else 0,
        "category_distribution": {k: v / total for k, v in category_counts.items()},
    }


def compute_drift_score(batch_stats, baseline_stats):
    """L1 distance between category distributions + normalized length delta.
    Real, computed metric -- not a placeholder. Simple by design so it's easy
    to audit; swap in a proper statistical test (e.g. population stability
    index, KL divergence) once you have enough volume for it to be stable.
    """
    if baseline_stats is None:
        return None

    all_cats = set(batch_stats["category_distribution"]) | set(baseline_stats["category_distribution"])
    l1 = sum(
        abs(batch_stats["category_distribution"].get(c, 0) - baseline_stats["category_distribution"].get(c, 0))
        for c in all_cats
    )

    base_len = baseline_stats["avg_input_length"] or 1
    length_delta = abs(batch_stats["avg_input_length"] - base_len) / base_len

    # weighted blend; category shift weighted higher since it's the stronger
    # signal for "this batch looks like a different kind of traffic"
    return round(0.7 * (l1 / 2) + 0.3 * min(length_delta, 1.0), 4)


def build_training_mix(usable_batch_examples, replay_buffer, rehearsal_ratio):
    """Combine new usable examples with a rehearsal sample. Returns counts
    only (this kit doesn't own an actual trainer loop -- autolearn / train.py do that)."""
    n_new = len(usable_batch_examples)
    if n_new == 0 or not replay_buffer:
        n_replay = 0
    else:
        # solve for n_replay such that n_replay / (n_new + n_replay) ~= ratio
        n_replay = round((rehearsal_ratio * n_new) / max(1 - rehearsal_ratio, 1e-6))
        n_replay = min(n_replay, len(replay_buffer))
    return {"n_new": n_new, "n_replay": n_replay, "n_total": n_new + n_replay}


# -- real eval implementation -------------------------------------------------

def _tokens(text):
    return _WORD_RE.findall((text or "").lower())


def _lexical_f1(candidate, reference):
    """Token-level F1 overlap, the same family of metric SQuAD/most
    generation benchmarks use for a free, no-dependency baseline. Not a
    substitute for semantic scoring -- good enough to gate "did the update
    make responses look more or less like the reference" without needing
    an API key or an embeddings model."""
    cand_tokens, ref_tokens = _tokens(candidate), _tokens(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_counts, ref_counts = Counter(cand_tokens), Counter(ref_tokens)
    overlap = sum((cand_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _keyword_hit_rate(candidate, acceptance_criteria):
    if not acceptance_criteria:
        return 1.0
    cand_lower = (candidate or "").lower()
    hits = sum(1 for phrase in acceptance_criteria if phrase.lower() in cand_lower)
    return hits / len(acceptance_criteria)


def _score_heuristic(candidate, record):
    f1 = _lexical_f1(candidate, record.get("reference_output", ""))
    kw = _keyword_hit_rate(candidate, record.get("acceptance_criteria", []))
    return round(0.6 * f1 + 0.4 * kw, 4)


def run_eval_suite(model_version=None, mw=None, route="local"):
    """
    Real when `mw` is provided (a zivawhite.router.ZivaWhite
    instance, already pointed at whatever checkpoint you're evaluating):
    generates a response for every eval_benchmark.jsonl record and scores
    it with a free lexical-overlap heuristic -- works fully offline.

    Returns None (same as the original stub) when `mw` is None or there's
    no eval set to run -- "no real eval available" stays an honest signal
    callers can check for, never a faked number.
    """
    if mw is None:
        return None
    eval_set = load_jsonl("eval_benchmark.jsonl")
    if not eval_set:
        return None

    scores = []
    for record in eval_set:
        answer = mw.ask(record["input"], route=route)
        scores.append(_score_heuristic(answer.text, record))
    return round(sum(scores) / len(scores), 4) if scores else None


def usable_examples(batch):
    """Which feedback records actually carry a usable training target."""
    usable = []
    for r in batch:
        if r["feedback_type"] == "accept":
            usable.append({"input": r["input"], "target": r["model_output"], "source": "accept"})
        elif r["feedback_type"] == "edit" and r.get("corrected_output"):
            usable.append({"input": r["input"], "target": r["corrected_output"], "source": "edit"})
        elif r["feedback_type"] == "implicit_positive":
            usable.append({"input": r["input"], "target": r["model_output"], "source": "implicit_positive"})
        # reject / implicit_negative: excluded from positive targets by design
    return usable


def _load_zivawhite(checkpoint_path):
    """Import lazily -- this script (and validate_dataset.py etc.) should
    still run with zero zivawhite/torch dependency when --checkpoint
    isn't given, matching the kit's original standalone-analysis design."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from zivawhite.router import ZivaWhite

    return ZivaWhite(checkpoint_path=checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                         help="path to a checkpoint.pt to evaluate for real. Omit to keep the original "
                              "heuristic-only behavior (eval_score stays null).")
    parser.add_argument("--route", default="local", choices=["local", "auto"],
                         help="which ZivaWhite route to evaluate (default: local, i.e. the checkpoint itself)")
    args = parser.parse_args()

    feedback = load_jsonl("feedback_log.jsonl")
    replay_buffer = load_jsonl("replay_buffer_sample.jsonl")

    if not feedback:
        print("No feedback_log.jsonl found. Put yours in <datasets folder>/_eval/ (or this kit's data/ folder); "
              "see scripts/online_learning/README.md for the format.")
        return

    mw = _load_zivawhite(args.checkpoint) if args.checkpoint else None
    if args.checkpoint and not mw.has_local_model():
        print(f"Warning: --checkpoint {args.checkpoint} didn't load a model -- "
              f"falling back to heuristic-only eval_score for this run.")
        mw = None

    baseline_stats = compute_baseline_stats(replay_buffer)
    if baseline_stats is None:
        print("Warning: no replay buffer loaded -- drift scores will be null "
              "and rehearsal mixing will be skipped. Run build_replay_buffer.py first.")

    drift_log = []
    version_log = []
    version_counter = 1
    prev_quality = None

    for batch_idx, batch in enumerate(batchify(feedback, BATCH_SIZE), start=1):
        batch_id = f"batch_{batch_idx:04d}"
        stats = compute_batch_stats(batch)
        drift_score = compute_drift_score(stats, baseline_stats) if baseline_stats else None

        drift_log.append({
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_examples": stats["n"],
            "avg_input_length": round(stats["avg_input_length"], 1),
            "category_distribution": stats["category_distribution"],
            "drift_score": drift_score,
            "baseline_reference": "zivawhite_dataset_v1" if baseline_stats else None,
            "flag_for_review": bool(drift_score is not None and drift_score > DRIFT_ALERT_THRESHOLD),
        })

        usable = usable_examples(batch)
        mix = build_training_mix(usable, replay_buffer, REHEARSAL_RATIO)

        # batch_quality is a REAL computed proxy (acceptance signal), not a
        # model-performance score -- see run_eval_suite() docstring.
        batch_quality = round(
            stats["accept_rate"] + 0.6 * stats["edit_rate"] + 0.3 * stats["implicit_positive_rate"],
            4,
        )

        eval_result = run_eval_suite(model_version=f"v1.{version_counter}", mw=mw, route=args.route)

        if eval_result is not None:
            promoted = eval_result >= (prev_quality or 0)
            decision_basis = "real_eval (lexical heuristic)"
        else:
            drift_ok = drift_score is None or drift_score <= DRIFT_ALERT_THRESHOLD
            promoted = drift_ok and batch_quality >= QUALITY_PROMOTE_THRESHOLD
            decision_basis = "heuristic_pending_real_eval"

        version_log.append({
            "version_id": f"v1.{version_counter}",
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_mix": mix,
            "batch_quality_proxy": batch_quality,
            "drift_score": drift_score,
            "eval_score": eval_result,
            "promoted": promoted,
            "decision_basis": decision_basis,
            "notes": "eval_score is null -- pass --checkpoint to wire up a real model" if eval_result is None else "",
        })

        version_counter += 1
        prev_quality = eval_result if eval_result is not None else batch_quality

    with open(os.path.join(DATA_DIR, "drift_monitor_log.jsonl"), "w", encoding="utf-8") as f:
        for entry in drift_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(os.path.join(DATA_DIR, "version_registry.jsonl"), "w", encoding="utf-8") as f:
        for entry in version_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {len(feedback)} feedback records into {len(drift_log)} batches.")
    print(f"Wrote drift_monitor_log.jsonl ({len(drift_log)} entries)")
    print(f"Wrote version_registry.jsonl ({len(version_log)} entries)")
    n_promoted = sum(1 for v in version_log if v["promoted"])
    print(f"{n_promoted}/{len(version_log)} batches would be promoted under the current {'real eval' if mw else 'heuristic'}.")
    if mw is None:
        print("\nReminder: promotion decisions above are heuristic-only. Pass --checkpoint "
              "runs/ziva/checkpoint.pt to score against your actual model instead.")


if __name__ == "__main__":
    main()
