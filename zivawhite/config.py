"""
config.py - one place that decides every setting.

Layers, lowest to highest priority:

  1. configs/ziva.yaml           shipped defaults (documented line by line). In an installed package
                                 (pip install .) the same file ships inside it as zivawhite/defaults/ziva.yaml.
  2. configs/ziva.local.yaml     your overrides -- only the keys you want to change
     (or the file named by $ZIVAWHITE_CONFIG instead of this one)
  3. environment                 ZIVAWHITE_DATASETS (folder of training data)

Where does "the project folder" (ROOT) come from? It is where datasets/, data/, runs/ and models/ live and where
configs/ziva.local.yaml is looked for:
  * $ZIVAWHITE_HOME if you set it;
  * else the source checkout this file sits in (also what `pip install -e .` gives you);
  * else, for a normal `pip install`, the folder you run the command from.

Nothing else in the project hardcodes a dataset path, wake word, creator name,
app list or port: it asks this module. Deliberately torch-free so the CLI, the
voice loop and the device-control layer all start instantly.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent
_CHECKOUT_DIR = _PACKAGE_DIR.parent


def _find_root() -> Path:
    env = os.environ.get("ZIVAWHITE_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if (_CHECKOUT_DIR / "configs" / "ziva.yaml").is_file():
        return _CHECKOUT_DIR                              # a source checkout (or `pip install -e .` of one)
    return Path.cwd()                                     # an installed package: work where you run it


def shipped_defaults_path() -> Path:
    """The shipped defaults file: configs/ziva.yaml in a checkout, else the copy inside the installed package."""
    checkout = _CHECKOUT_DIR / "configs" / "ziva.yaml"
    return checkout if checkout.is_file() else _PACKAGE_DIR / "defaults" / "ziva.yaml"


ROOT = _find_root()                                       # the project folder (datasets/, data/, runs/, local overrides)
DEFAULT_CONFIG_PATH = shipped_defaults_path()
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
            raise ConfigError(f"missing setting {dotted!r} -- restore it from the shipped defaults (configs/ziva.yaml)")
        return value

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    # -- paths ------------------------------------------------------------------
    def resolve(self, value: str | os.PathLike) -> Path:
        """Expand ~ and make relative paths relative to the project root."""
        s = str(value)
        p = Path(os.path.expanduser(s))
        # A leading '/' is absolute on every OS, even though Windows Path
        # does not treat it as such (it lacks a drive letter).
        if p.is_absolute() or s.startswith("/"):
            return p
        return self.root / p

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
        out.extend(self._problems_router_limits_tokenizer())
        return out

    def _problems_router_limits_tokenizer(self) -> List[str]:
        out: List[str] = []

        def number(dotted: str, lo: float, hi: Optional[float] = None, *, integer: bool = False, lo_open: bool = False,
                   nullable: bool = False) -> None:
            v = self.get(dotted)
            if v is None:
                if not nullable and self.get(dotted, _MISSING) is not _MISSING:
                    out.append(f"{dotted}: must not be empty")
                return
            if isinstance(v, bool) or not isinstance(v, int if integer else (int, float)):
                out.append(f"{dotted}: expected {'a whole number' if integer else 'a number'}, got {type(v).__name__}")
            elif (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
                out.append(f"{dotted}: {v} is out of range ({'>' if lo_open else '>='} {lo}{'' if hi is None else f' and <= {hi}'})")

        number("router.confidence_floor", 0.0, 1.0)
        number("router.loop_threshold", 0.0, 1.0, lo_open=True)
        number("limits.max_new_tokens", 1, integer=True, nullable=True)
        number("limits.lock_timeout_s", 0.0, lo_open=True)
        number("limits.max_body_bytes", 0, integer=True)
        number("voice.tts.timeout_s", 0.0, lo_open=True)

        years = self.get("router.freshness_years")
        if years is not None and not (isinstance(years, list) and len(years) == 2 and all(
                isinstance(y, int) and not isinstance(y, bool) for y in years) and years[0] <= years[1]):
            out.append("router.freshness_years: expected [first_year, last_year] (two whole numbers, first <= last) or null")
        patterns = self.get("router.freshness_patterns")
        if patterns is not None:
            if not isinstance(patterns, list) or not all(isinstance(x, str) for x in patterns):
                out.append("router.freshness_patterns: expected a list of regular expressions")
            else:
                import re

                for x in patterns:
                    try:
                        re.compile(x)
                    except re.error as e:
                        out.append(f"router.freshness_patterns: {x!r} is not a valid regular expression ({e})")

        tok = self.get("autolearn.tokenizer_dir")
        if tok:
            from .tokenizer import has_tokenizer

            if not isinstance(tok, str):
                out.append("autolearn.tokenizer_dir: expected a folder path or null")
            elif not has_tokenizer(str(self.resolve(tok))):
                out.append(f"autolearn.tokenizer_dir: {self.resolve(tok)} holds no tokenizer (expected vocab.json + merges.txt). "
                           f"Create one with `python -m zivawhite.app tokenizer`, or set it back to null for the bundled one.")
        return out


_cached: Optional[Config] = None


def load_config(path: Optional[os.PathLike | str] = None, root: Path = ROOT, env: Optional[dict] = None) -> Config:
    """Build the merged Config. `path` replaces the shipped defaults file (used by tests);
    normally call get_config()."""
    env = os.environ if env is None else env
    if path:
        base_path = Path(path)
    else:
        own_copy = root / "configs" / "ziva.yaml"          # a full copy in your project folder wins; else the shipped one
        base_path = own_copy if own_copy.exists() else DEFAULT_CONFIG_PATH
    if not base_path.exists():
        raise ConfigError(f"{base_path} not found -- the shipped defaults are missing; reinstall, or restore configs/ziva.yaml")
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
