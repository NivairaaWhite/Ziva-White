"""The fine-tuning / resume flow that used to leave the untuned model being served.
Each case runs the real train.py for a handful of steps on a tiny corpus (CPU, seconds)."""
import hashlib
import os
import sys

import pytest

from zivawhite.checkpoint import load_checkpoint
from zivawhite.data.ingest import build_shards
from zivawhite.train import lr_at


def _shards(tmp_path, name="shards", seed_text="the quick brown fox jumps over the lazy dog "):
    corpus = tmp_path / f"{name}.txt"
    corpus.write_text("".join(f"{seed_text}{i} " * 8 + "\n" for i in range(400)), encoding="utf-8")
    out = str(tmp_path / name)
    build_shards(str(corpus), None, out, doc_mode="line", min_val_tokens=600, max_block_size=256)
    return out


def _train(*args):
    from zivawhite import train as train_mod

    sys.argv = ["train.py", "--batch_size", "2", "--grad_accum", "1", "--warmup_steps", "1", *map(str, args)]
    train_mod.main()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_lr_schedule_options():
    kw = dict(warmup_steps=10, max_lr=1.0, min_lr=0.1)
    assert lr_at(0, max_steps=100, **kw) == pytest.approx(0.1)            # warmup starts low
    assert lr_at(50, max_steps=100, schedule="constant", **kw) == 1.0
    assert lr_at(500, max_steps=100, schedule="constant", **kw) == 1.0
    # chunked training: chunk 2 (max_steps=200) at step 150 must follow the SAME curve as chunk 1 would
    # with the real horizon of 1000, instead of decaying over its own 200 steps
    assert lr_at(150, max_steps=200, horizon=1000, **kw) > lr_at(150, max_steps=200, **kw)
    assert lr_at(2000, max_steps=200, horizon=1000, **kw) == 0.1           # past the horizon -> min_lr


def test_finetune_never_touches_the_base_checkpoint_and_becomes_the_best(tmp_path):
    base_data, tune_data = _shards(tmp_path, "base"), _shards(tmp_path, "tune", "a different corpus ")
    base, tuned = str(tmp_path / "base_run"), str(tmp_path / "tuned_run")
    _train("--data_dir", base_data, "--out_dir", base, "--max_steps", 4, "--eval_interval", 2, "--ckpt_interval", 4)
    base_ck = os.path.join(base, "checkpoint.pt")
    before = _sha(base_ck)

    _train("--data_dir", tune_data, "--out_dir", tuned, "--init_from", base_ck, "--max_steps", 3,
           "--eval_interval", 2, "--ckpt_interval", 3, "--behavior_tuned")

    assert _sha(base_ck) == before                                   # the base model is only read
    tuned_ck = load_checkpoint(os.path.join(tuned, "checkpoint.pt"))
    assert tuned_ck["step"] == 3                                     # starts at 0, not at the base's step
    assert tuned_ck["extra"]["behavior_tuned"] is True
    assert tuned_ck["config"] == load_checkpoint(base_ck)["config"]  # same shape as the base
    best = load_checkpoint(os.path.join(tuned, "checkpoint_best.pt"))
    assert best["extra"]["behavior_tuned"] is True                   # what serving loads is the tuned model


def test_a_run_that_never_evals_still_replaces_a_stale_best(tmp_path):
    data = _shards(tmp_path)
    run = str(tmp_path / "run")
    _train("--data_dir", data, "--out_dir", run, "--max_steps", 4, "--eval_interval", 2, "--ckpt_interval", 4)
    stale_step = load_checkpoint(os.path.join(run, "checkpoint_best.pt"))["step"]
    # a fresh run into the same folder, too short to reach an eval: the old best must not survive
    _train("--data_dir", data, "--out_dir", run, "--max_steps", 1, "--eval_interval", 1000, "--ckpt_interval", 1000)
    assert load_checkpoint(os.path.join(run, "checkpoint_best.pt"))["step"] == 1 != stale_step


def test_resume_rebuilds_the_model_from_the_checkpoint_not_the_hardware_tier(tmp_path):
    data = _shards(tmp_path)
    run = str(tmp_path / "run")
    _train("--data_dir", data, "--out_dir", run, "--max_steps", 2, "--eval_interval", 1000, "--ckpt_interval", 1000,
           "--tier", "tiny")
    n_embd = load_checkpoint(os.path.join(run, "checkpoint.pt"))["config"]["n_embd"]
    # asking for a different tier while resuming used to rebuild a differently-shaped model and fail to load
    _train("--data_dir", data, "--out_dir", run, "--max_steps", 4, "--eval_interval", 1000, "--ckpt_interval", 1000,
           "--tier", "small", "--resume")
    after = load_checkpoint(os.path.join(run, "checkpoint.pt"))
    assert after["step"] == 4 and after["config"]["n_embd"] == n_embd


def test_config_yaml_values_become_defaults_and_flags_override_them(tmp_path):
    data = _shards(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data_dir: {data}\nmax_steps: 2\neval_interval: 1000\nckpt_interval: 1000\n", encoding="utf-8")
    run = str(tmp_path / "run")
    _train("--config", str(cfg), "--out_dir", run)
    assert load_checkpoint(os.path.join(run, "checkpoint.pt"))["step"] == 2   # max_steps came from the file
    run2 = str(tmp_path / "run2")
    _train("--config", str(cfg), "--out_dir", run2, "--max_steps", 3)
    assert load_checkpoint(os.path.join(run2, "checkpoint.pt"))["step"] == 3  # the command line wins
