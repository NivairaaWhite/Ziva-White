"""Test helper: a stand-in for the few torch bits the chat/streaming layer touches, used ONLY when torch is not
installed (so these tests still run in a bare sandbox or CI image). Imported once by name, so whether real torch is
present is decided exactly once and every test module agrees. Nothing is installed at import time: see install()."""
import sys

try:
    import torch as _real  # noqa: F401
    HAVE_TORCH = hasattr(_real, "nn")
except ImportError:
    HAVE_TORCH = False


class Row:
    def __init__(self, v): self.v = list(v)
    def tolist(self): return list(self.v)


class Tensor:
    def __init__(self, rows): self.rows = [list(r) for r in rows]
    def size(self, d): return len(self.rows) if d == 0 else len(self.rows[0])
    def __getitem__(self, key):
        r, c = key
        return self.rows[r][c] if isinstance(c, int) else Row(self.rows[r][c])


class FakeTorch:
    long = "long"
    @staticmethod
    def tensor(rows, dtype=None, device=None): return Tensor(rows)


_saved = {}


def install():
    """Put the stand-in in sys.modules (a no-op when real torch is present). Call from setup_function and pair it
    with uninstall() in teardown_function, so the stand-in never leaks into tests that expect torch to be absent
    (e.g. pytest.importorskip("torch") must still skip)."""
    if not HAVE_TORCH:
        _saved["torch"] = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch()


def uninstall():
    if not HAVE_TORCH and "torch" in _saved:
        if _saved["torch"] is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _saved["torch"]
        _saved.clear()


def make_tensor(rows):
    """A (1, n) token tensor: real torch if available, else the stand-in."""
    if HAVE_TORCH:
        import torch
        return torch.tensor(rows)
    return Tensor(rows)


def row_of(idx):
    return idx[0].tolist() if HAVE_TORCH else idx.rows[0][:]


class Cfg:
    block_size = 4096


class ScriptedModel:
    """generate_stream replays `script` (token ids), honouring stop_fn exactly like the real model, and records
    what it was asked (prompt ids + sampling kwargs)."""
    cfg = Cfg()

    def __init__(self, script):
        self.script = list(script)
        self.last_prompt_ids, self.last_kw = None, None

    def generate_stream(self, idx, max_new_tokens, stop_fn=None, **kw):
        seq = row_of(idx)
        self.last_prompt_ids, self.last_kw = list(seq), dict(kw, max_new_tokens=max_new_tokens)
        for tok in self.script[:max_new_tokens]:
            seq.append(tok)
            yield tok
            if stop_fn is not None and stop_fn(make_tensor([seq])):
                return
