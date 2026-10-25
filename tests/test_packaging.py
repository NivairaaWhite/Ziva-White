"""The package must be installable as a real package, not only from a checkout (review item: the built wheel lacked
the web UI and the bundled tokenizer, and declared no runtime dependencies)."""
import os
import re
from pathlib import Path

import pytest

try:
    import tomllib                                   # Python 3.11+
except ModuleNotFoundError:                          # 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parent.parent


def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _names(lines):
    out = {}
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        out[m.group(1).lower().replace("_", "-")] = m.group(2).replace(" ", "")
    return out


def test_runtime_dependencies_are_declared_and_agree_with_requirements_txt():
    declared = _names(pyproject()["project"]["dependencies"])
    required = _names((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert declared, "the package declares no runtime dependencies"
    assert declared == required, "pyproject.toml and requirements.txt disagree"


def test_every_runtime_dependency_has_an_upper_bound():
    for name, spec in _names(pyproject()["project"]["dependencies"]).items():
        assert "<" in spec, f"{name} has no upper bound ({spec!r})"


def test_every_subpackage_is_listed_so_the_wheel_is_complete():
    listed = set(pyproject()["tool"]["setuptools"]["packages"])
    found = {str(p.parent.relative_to(ROOT)).replace(os.sep, ".") for p in (ROOT / "zivawhite").rglob("__init__.py")}
    assert found <= listed, f"packages missing from pyproject.toml: {sorted(found - listed)}"
    assert "zivawhite.defaults" in listed


def test_the_files_the_app_reads_at_runtime_ship_in_the_package():
    data = pyproject()["tool"]["setuptools"]["package-data"]
    assert "webui/*.html" in data["zivawhite"]
    assert any(p.startswith("tokenizer/") for p in data["zivawhite.data"])
    assert "*.yaml" in data["zivawhite.defaults"]
    assert (ROOT / "zivawhite" / "webui" / "index.html").is_file()
    assert (ROOT / "zivawhite" / "data" / "tokenizer" / "vocab.json").is_file() and (ROOT / "zivawhite" / "data" / "tokenizer" / "merges.txt").is_file()


def test_defaults_ship_from_configs_and_the_private_override_never_does():
    p = pyproject()["tool"]["setuptools"]
    assert p["package-dir"]["zivawhite.defaults"] == "configs"
    assert "ziva.local.yaml" in p["exclude-package-data"]["zivawhite.defaults"]
    assert (ROOT / "configs" / "ziva.yaml").is_file() and (ROOT / "configs" / "__init__.py").is_file()


def test_the_config_loader_has_a_fallback_for_an_installed_package():
    from zivawhite import config
    assert config.shipped_defaults_path().is_file()
    assert (config._PACKAGE_DIR / "defaults").name == "defaults"      # where the wheel puts configs/
