#!/usr/bin/env python3
"""
launch.py - the "single click" behaviour, in one place so start.bat (Windows), start.sh (Linux) and
start.command (macOS) can all just call `python launch.py` and behave identically.

What it does, in order:
  1. makes sure the datasets folder exists (location: configs/ziva.yaml -> paths.datasets) and tells
     you what is in it. Nothing is downloaded and no sample data is created: the assistant learns
     from the files YOU put there.
  2. picks a free port, starting at background.port (so a second click doesn't crash on
     "address in use");
  3. starts `zivawhite.app run`: the web UI and API on 127.0.0.1 (not 0.0.0.0: a double-clicked
     local app should not default to being reachable from your network) PLUS the background jobs
     from the config -- the datasets watcher that trains on new files, timers, and the voice loop
     if you enabled it;
  4. waits until the server really answers /v1/health, THEN opens your browser (no "site can't be
     reached" flash).

With no model trained yet the UI still opens (identity answers, device commands and timers work);
questions are answered once autolearn has trained a first model from your data.

    python launch.py              # normal
    python launch.py --no-browser # headless / autostart
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from zivawhite.config import ConfigError, get_config  # noqa: E402  (torch-free)


def find_free_port(start: int = 8000, tries: int = 10) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port found in range {start}-{start + tries}")


def wait_for_server(port: int, timeout_s: float = 60.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1.0)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def describe_datasets(cfg) -> None:
    from zivawhite.data.folder import DatasetFolder

    folder = DatasetFolder.from_config(cfg)
    folder.ensure_layout()
    st = folder.status()
    print(f"[launch] datasets folder: {st['folder']}")
    if st["files"] == 0:
        print("[launch]   it is empty. Drop text files into text/ and conversations into chat/ and the assistant")
        print("[launch]   trains on them automatically (see datasets/README.md). Until then there is no local model.")
    else:
        print(f"[launch]   {st['files']} file(s), {st['total_mb']} MB; training starts in the background when something is new.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    try:
        cfg = get_config()
    except ConfigError as e:
        print(f"[launch] configuration problem: {e}")
        sys.exit(1)
    for problem in cfg.problems():
        print(f"[launch] config warning: {problem}")
    describe_datasets(cfg)

    port = find_free_port(int(cfg.get("background.port", 8000)))
    cmd = [sys.executable, "-m", "zivawhite.app", "run", "--host", "127.0.0.1", "--port", str(port)]
    print(f"[launch] starting Ziva White on http://127.0.0.1:{port} ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))

    try:
        if wait_for_server(port):
            if not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            print("[launch] ready. Ctrl+C here stops everything (use `daemon start` to run without this window).")
        else:
            print(f"[launch] server didn't respond in time -- open http://127.0.0.1:{port}/ manually,")
            print("[launch] or check the output above for an error.")
        proc.wait()
    except KeyboardInterrupt:
        print("\n[launch] stopping...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
