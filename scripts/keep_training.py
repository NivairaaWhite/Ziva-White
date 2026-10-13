#!/usr/bin/env python3
"""
keep_training.py - "it should only be better and better": run training
in chunks, indefinitely or up to --total_steps, automatically resuming
after each chunk. Meant to be left running in the background (or
restarted any time) on a CPU box that has spare cycles.

This script is *just* the loop that keeps feeding the model more
training time -- the actual "never regresses" guarantee comes from
train.py + checkpoint.py: every eval, if val_loss improves,
checkpoint_best.pt is (re)written; ask/serve always prefer that file
over the rolling checkpoint.pt (see checkpoint.py:resolve_serving_checkpoint).
So even though checkpoint.pt keeps moving forward step by step (and can
get temporarily worse -- a bad batch, an LR bump), what's actually
served can only get better run over run.

Safe to interrupt with Ctrl+C at any point: the current chunk's own
--ckpt_interval checkpoints are unaffected (crash-safe, atomic writes --
see checkpoint.py), you just lose progress since the last one, exactly
like interrupting train.py directly. Rerun this script any time to
keep going -- it reads the existing checkpoint's step count and
continues from there.

Usage:
    # run forever (until you Ctrl+C), 500 steps at a time
    python scripts/keep_training.py --data_dir data/datasets/text/shards --out_dir runs/ziva

    # stop automatically once the checkpoint reaches step 50000
    python scripts/keep_training.py --data_dir data/datasets/text/shards --out_dir runs/ziva --total_steps 50000

    # bigger chunks, no pause between them, fewer/cheaper evals
    python scripts/keep_training.py --data_dir data/datasets/text/shards --out_dir runs/ziva \\
        --chunk_steps 2000 --sleep_seconds 0 --eval_interval 500 --ckpt_interval 500
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _checkpoint_step(ckpt_path: str) -> int:
    if not os.path.exists(ckpt_path):
        return 0
    sys.path.insert(0, str(ROOT))
    from zivawhite.checkpoint import load_checkpoint

    return load_checkpoint(ckpt_path, map_location="cpu")["step"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/datasets/text/shards")
    p.add_argument("--out_dir", default="runs/ziva")
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--tier", default=None, choices=[None, "tiny", "small", "medium", "large"])
    p.add_argument("--chunk_steps", type=int, default=500, help="how many additional steps to train per chunk")
    p.add_argument("--total_steps", type=int, default=None,
                    help="stop once the checkpoint reaches this absolute step count; omit to run until interrupted")
    p.add_argument("--sleep_seconds", type=float, default=5.0,
                    help="pause between chunks, e.g. to let a laptop cool down or to watch progress")
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--ckpt_interval", type=int, default=100)
    args = p.parse_args()

    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    step = _checkpoint_step(ckpt_path)
    if step:
        print(f"[keep_training] found existing checkpoint at step {step:,} -- continuing")
    else:
        print("[keep_training] no existing checkpoint -- starting fresh")

    chunk_n = 0
    try:
        while args.total_steps is None or step < args.total_steps:
            target = step + args.chunk_steps
            if args.total_steps is not None:
                target = min(target, args.total_steps)
            chunk_n += 1
            print(f"\n[keep_training] chunk {chunk_n}: {step:,} -> {target:,} steps")

            cmd = [
                sys.executable, "-m", "zivawhite.train",
                "--data_dir", args.data_dir, "--out_dir", args.out_dir,
                "--max_steps", str(target),
                "--eval_interval", str(args.eval_interval), "--ckpt_interval", str(args.ckpt_interval),
            ]
            if step > 0:
                cmd.append("--resume")
            # One LR curve across all chunks. Without this each chunk decayed LR over its own
            # --max_steps and jumped back to the peak at the next chunk (a sawtooth).
            if args.total_steps is not None:
                cmd += ["--lr_horizon", str(args.total_steps)]
            else:
                cmd += ["--schedule", "constant"]
            if args.tokenizer_dir:
                cmd += ["--tokenizer_dir", args.tokenizer_dir]
            if args.tier:
                cmd += ["--tier", args.tier]

            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"[keep_training] chunk {chunk_n} exited with code {result.returncode} -- stopping "
                      f"(checkpoint.pt/checkpoint_best.pt from earlier chunks are untouched; fix the error "
                      f"and rerun this script to pick back up).")
                return

            new_step = _checkpoint_step(ckpt_path)
            if new_step <= step:
                print(f"[keep_training] step didn't advance ({step:,} -> {new_step:,}) -- stopping to avoid "
                      f"spinning forever (check --data_dir has enough tokens for this many steps).")
                return
            step = new_step

            if args.sleep_seconds > 0 and (args.total_steps is None or step < args.total_steps):
                time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        print(f"\n[keep_training] stopped at step {step:,}. checkpoint.pt and checkpoint_best.pt are "
              f"safely saved -- rerun this exact command any time to keep going.")


if __name__ == "__main__":
    main()
