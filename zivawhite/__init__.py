"""
ZivaWhite
============

One system, two faces:

  1. A real, from-scratch, trainable transformer language model
     (zivawhite.model.ZivaGPT) that runs and trains on anything
     from an 8GB CPU laptop up to multi-GPU boxes.

  2. A router (zivawhite.router.ZivaWhite) that answers every query through
     the SAME interface, deciding per query whether the local model answers
     directly or a live web search / page fetch grounds the answer first (the
     local model then writes it). There is no external AI-model API: every
     answer comes from this project's own weights.

There is one entrypoint (`zivawhite.router.ZivaWhite`); the local model is
the only thing that writes answers, and the web is a source of context for it.
"""

# Load .env (SEARXNG_URL, SEARCH_API_URL, ZIVAWHITE_SERVER_API_KEY, ...) into the
# process environment before anything reads os.getenv() for it. Every
# connector reads os.getenv() directly, so without this a freshly-created .env
# would be silently ignored however the app was started (CLI, launch.py, or
# tests). python-dotenv never overwrites a variable already set in the real
# environment, so an explicit `export`/`set` still wins over .env.
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    # zero-dependency degrade: .env just won't be picked up automatically;
    # `export SEARXNG_URL=...` (or setting it before launch) still works.
    pass

__all__ = ["ZivaWhite", "ZivaGPT", "ModelConfig", "resolve_runtime"]
__version__ = "0.4.0"

# Lazy exports (PEP 562). `import zivawhite.config`, `zivawhite.data.folder`, the voice and
# device-control packages and the CLI's dataset commands never touch torch; the heavy modules
# below are only imported the first time someone actually asks for one of these names, so
# `from zivawhite import ZivaWhite` still works exactly as before.
_LAZY = {
    "ZivaWhite": ".router",
    "ZivaGPT": ".model",
    "ModelConfig": ".modelconfig",   # torch-free: sizing, planning and export work without PyTorch
    "resolve_runtime": ".device",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
