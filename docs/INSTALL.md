# Installing

Back to the [README](../README.md).

## From a checkout (the normal way)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
```

Or just run `start.bat` / `start.command` / `./start.sh`: they do that for you.

In a checkout the "project folder" (where `datasets/`, `data/`, `runs/` and `configs/ziva.local.yaml` live) is the
checkout itself.

## As a package

```bash
pip install .            # or: python -m build --wheel, then pip install dist/zivawhite-*.whl
mkdir ~/my-ziva && cd ~/my-ziva
ziva datasets --init     # same as: python -m zivawhite.app datasets --init
ziva run
```

The wheel carries everything the program reads at run time: the web UI, the bundled tokenizer, and the default
settings (`zivawhite/defaults/ziva.yaml`, the same file as `configs/ziva.yaml` in the repository) plus the model
presets. The dependencies are declared, so `pip install` pulls them in.

An installed copy has no checkout to live in, so the **project folder is the folder you run it from** (or
`ZIVAWHITE_HOME`, if you set it). Your data goes in `datasets/` there, models in `runs/`, and your overrides in
`configs/ziva.local.yaml` there. `.env` is read from the same folder (or the file named by `ZIVAWHITE_ENV_FILE`);
parent folders are never searched.

| | checkout | installed package |
|---|---|---|
| project folder | the checkout | `ZIVAWHITE_HOME`, else the folder you run from |
| default settings | `configs/ziva.yaml` | `zivawhite/defaults/ziva.yaml` |
| your overrides | `configs/ziva.local.yaml` | `<project folder>/configs/ziva.local.yaml` |
| the launchers `start.*`, `launch.py` | yes | not included: use `ziva run` |

## Reproducible installs

`requirements.txt` and `pyproject.toml` give every runtime dependency a lower bound and an upper bound at the next
major version (a tested range, not an exact version). To pin the exact versions of an environment you have checked
works:

```bash
pip freeze > requirements.lock          # git-ignored; keep it wherever you keep such things
pip install -r requirements.lock        # on another machine
```

(No lockfile is committed: the right one depends on your platform and on which PyTorch build — CPU, CUDA, ROCm — you
install.)

## Tests and CI

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`.github/workflows/ci.yml` runs the suite on Ubuntu and Windows (Python 3.10 and 3.12) and builds the wheel,
failing if the web UI, tokenizer or default settings are missing from it, or a private `ziva.local.yaml` is in it.
