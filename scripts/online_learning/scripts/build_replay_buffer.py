#!/usr/bin/env python3
"""
Builds data/replay_buffer_sample.jsonl from YOUR conversations in the datasets folder.

Online/continual learning has a well-known failure mode: a model that keeps learning only from new
incoming data gradually forgets what it knew before ("catastrophic forgetting"). The standard
mitigation is a replay/rehearsal buffer -- a sampled slice of earlier training data mixed back in
with every new update, so old behaviour keeps getting reinforced alongside the new.

This script reads every conversation under <datasets folder>/ (location: configs/ziva.yaml ->
paths.datasets, any format datasets/README.md lists) and takes a stratified sample, roughly equal per
"category". A conversation's category is the name of the file it came from (support.jsonl ->
"support"), so organising your chat data into files by topic gives you topic-balanced rehearsal for
free. Nothing is bundled: with an empty datasets folder it says so and writes nothing.

    python scripts/build_replay_buffer.py [--per-category 4] [--seed 7]
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR.parent / "data"
sys.path.insert(0, str(THIS_DIR.parents[2]))


def load_conversations(cfg):
    """{category: [(origin, messages), ...]} from every chat-shaped record in the datasets folder."""
    from zivawhite.data.folder import DatasetFolder, iter_file

    folder = DatasetFolder.from_config(cfg)
    by_category = {}
    for df in folder.scan().files:
        n = 0
        try:
            for kind, payload in iter_file(df, folder.max_file_bytes, folder.max_doc_chars):
                if kind == "chat":
                    by_category.setdefault(Path(df.rel).stem, []).append((f"{df.rel}#{n}", payload))
                    n += 1
        except Exception as e:  # noqa: BLE001 - an unreadable file is reported, not fatal
            print(f"skipping {df.rel}: {e}")
    return by_category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from zivawhite.config import get_config

    cfg = get_config()
    by_category = load_conversations(cfg)
    if not by_category:
        print(f"No conversations found in {cfg.datasets_dir}. Put chat data there first (see datasets/README.md).")
        return

    rng = random.Random(args.seed)                 # deterministic sampling
    buffer = []
    for category, records in sorted(by_category.items()):
        for origin, messages in rng.sample(records, min(args.per_category, len(records))):
            buffer.append({
                "buffer_id": f"buf_{len(buffer) + 1:04d}",
                "source": "datasets_folder",
                "original_id": origin,
                "category": category,
                "tags": [],
                "messages": messages,
                "weight": 1.0,
                "last_used": None,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "replay_buffer_sample.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in buffer:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Sampled {len(buffer)} replay entries across {len(by_category)} categories -> {out_path}")


if __name__ == "__main__":
    main()
