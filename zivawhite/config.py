"""
config.py - one place that decides every setting.

Layers, lowest to highest priority:

  1. configs/ziva.yaml           shipped defaults (documented line by line)
  2. configs/ziva.local.yaml     your overrides -- only the keys you want to change
     (or the file named by $ZIVAWHITE_CONFIG instead of this one)
  3. environment                 ZIVAWHITE_DATASETS (folder of training data)

Nothing else in the project hardcodes a dataset path, wake word, creator name,
app list or port: it asks this module. Deliberately torch-free so the CLI, the
voice loop and the device-control layer all start instantly.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent          # the project folder (where configs/ lives)
DEFAULT_CONFIG_PATH = ROOT / "configs" / "ziva.yaml"
LOCAL_CONFIG_PATH = ROOT / "configs" / "ziva.local.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    pass


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`. Dicts merge key by key;
    every other type (lists included) is replaced wholesale, so overriding
    `voice.wake.words` means exactly the list you wrote."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


class Config:
    """Read-only view over the merged settings with dotted access and path helpers."""

    def __init__(self, data: Dict[str, Any], root: Path = ROOT):
        self.data = data
        self.root = Path(root)

    # -- access ---------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"missing setting {dotted!r} -- restore it from configs/ziva.yaml")
        return value

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    # -- paths ------------------------------------------------------------------
    def resolve(self, value: str | os.PathLike) -> Path:
        """Expand ~ and make relative paths relative to the project root."""
        p = Path(os.path.expanduser(str(value)))
        return p if p.is_absolute() else (self.root / p)

    def path(self, dotted: str) -> Path:
        return self.resolve(self.require(dotted))

    @property
    def datasets_dir(self) -> Path:
        return self.path("paths.datasets")

    @property
    def data_dir(self) -> Path:
        return self.path("paths.data")

    @property
    def runs_dir(self) -> Path:
        return self.path("paths.runs")

    @property
    def models_dir(self) -> Path:
        return self.path("paths.models")

    # -- validation ---------------------------------------------------------------
    def problems(self) -> List[str]:
        """Human-readable problems with the merged settings (empty list = fine).
        Checked at startup so a typo in your override file fails loudly and early,
        not as a confusing error an hour into a background run."""
        out: List[str] = []

        def need(dotted: str, types: tuple, what: str) -> None:
            v = self.get(dotted, _MISSING)
            if v is _MISSING:
                out.append(f"{dotted}: missing")
            elif isinstance(v, bool) and bool not in types:
                out.append(f"{dotted}: expected {what}, got a boolean")
            elif not isinstance(v, types):
                out.append(f"{dotted}: expected {what}, got {type(v).__name__}")

        need("identity.assistant_name", (str,), "text")
        need("identity.creator.name", (str,), "text")
        need("paths.datasets", (str,), "a folder path")
        need("autolearn.scan_interval_s", (int, float), "a number")
        need("autolearn.settle_s", (int, float), "a number")
        need("voice.wake.words", (list,), "a list of phrases")
        need("voice.wake.sensitivity", (int, float), "a number between 0 and 1")
        need("background.port", (int,), "a port number")
        need("background.jobs", (list,), "a list of job names")
        need("device_control.apps", (dict,), "a mapping of app name -> launch targets")

        sens = self.get("voice.wake.sensitivity")
        if isinstance(sens, (int, float)) and not isinstance(sens, bool) and not 0.0 < float(sens) <= 1.0:
            out.append("voice.wake.sensitivity: must be greater than 0 and at most 1")
        words = self.get("voice.wake.words")
        if isinstance(words, list) and not [w for w in words if isinstance(w, str) and w.strip()]:
            out.append("voice.wake.words: needs at least one non-empty phrase")
        port = self.get("background.port")
        if isinstance(port, int) and not isinstance(port, bool) and not 1 <= port <= 65535:
            out.append("background.port: must be between 1 and 65535")
        known_jobs = {"autolearn", "voice", "timers"}
        jobs = self.get("background.jobs")
        if isinstance(jobs, list):
            for j in jobs:
                if j not in known_jobs:
                    out.append(f"background.jobs: unknown job {j!r} (known: {sorted(known_jobs)})")
        stt_engine = self.get("voice.stt.engine")
        if stt_engine not in ("vosk", "faster_whisper"):
            out.append(f"voice.stt.engine: {stt_engine!r} is not one of vosk, faster_whisper")
        tts_engine = self.get("voice.tts.engine")
        if tts_engine not in ("auto", "system", "pyttsx3", "custom"):
            out.append(f"voice.tts.engine: {tts_engine!r} is not one of auto, system, pyttsx3, custom")
        return out


_cached: Optional[Config] = None


def load_config(path: Optional[os.PathLike | str] = None, root: Path = ROOT, env: Optional[dict] = None) -> Config:
    """Build the merged Config. `path` replaces the shipped defaults file (used by tests);
    normally call get_config()."""
    env = os.environ if env is None else env
    base_path = Path(path) if path else root / "configs" / "ziva.yaml"
    if not base_path.exists():
        raise ConfigError(f"{base_path} not found -- run ZivaWhite from its project folder")
    data = _read_yaml(base_path)

    override_env = env.get("ZIVAWHITE_CONFIG", "").strip()
    override_path = Path(override_env) if override_env else root / "configs" / "ziva.local.yaml"
    if override_path.exists() and override_path.resolve() != base_path.resolve():
        data = deep_merge(data, _read_yaml(override_path))
    elif override_env and not override_path.exists():
        raise ConfigError(f"$ZIVAWHITE_CONFIG points at {override_path}, which does not exist")

    datasets_env = env.get("ZIVAWHITE_DATASETS", "").strip()
    if datasets_env:
        data = deep_merge(data, {"paths": {"datasets": datasets_env}})
    return Config(data, root=root)


def get_config(reload: bool = False) -> Config:
    global _cached
    if _cached is None or reload:
        _cached = load_config()
    return _cached
