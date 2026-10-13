"""Shared test helpers (plain module, imported by tests in this folder)."""
from pathlib import Path

from zivawhite.config import DEFAULT_CONFIG_PATH, Config, deep_merge, load_config


def make_cfg(tmp_path: Path, **overrides) -> Config:
    """The shipped defaults, re-rooted at tmp_path, with `overrides` deep-merged on top
    (e.g. make_cfg(tmp, autolearn={"scan_interval_s": 1})). Hermetic: a developer's own
    configs/ziva.local.yaml or $ZIVAWHITE_* variables never leak into a test."""
    base = load_config(path=DEFAULT_CONFIG_PATH, root=Path(tmp_path), env={})
    data = deep_merge(base.data, {"autolearn": {"settle_s": 0, "throttle": {"only_when_idle": False,
                                                                           "pause_on_battery": False}}})
    return Config(deep_merge(data, overrides), root=Path(tmp_path))
