"""Which .env file gets loaded. Deliberately dumb: one explicit place, never a search up the directory tree.

python-dotenv's find_dotenv() walks UP from the working directory through every parent folder until it finds a
.env. Run ZivaWhite from ~/projects/ziva and a forgotten ~/.env (or one in a shared parent) would silently apply:
settings, and API keys, from a file you never meant. So instead, in order:

  1. $ZIVAWHITE_ENV_FILE        an explicit path
  2. ./.env                     in the folder you run it from
  3. <project folder>/.env      where the source checkout lives (only if this really is the checkout)

Nothing else. A variable already set in the real environment always wins over the file (python-dotenv's default).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def dotenv_path(cwd: Optional[Path] = None, env: Optional[dict] = None) -> Optional[str]:
    env = os.environ if env is None else env
    explicit = (env.get("ZIVAWHITE_ENV_FILE") or "").strip()
    if explicit:
        return explicit if Path(explicit).expanduser().is_file() else None
    cwd = Path.cwd() if cwd is None else Path(cwd)
    checkout = Path(__file__).resolve().parent.parent
    candidates = [cwd / ".env"]
    if (checkout / "configs" / "ziva.yaml").is_file():          # a source checkout, not site-packages
        candidates.append(checkout / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def load_env() -> Optional[str]:
    """Load the .env chosen by dotenv_path() into os.environ; returns the path used (or None)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None          # zero-dependency degrade: `export SEARXNG_URL=...` still works
    path = dotenv_path()
    if path:
        load_dotenv(path)
    return path
