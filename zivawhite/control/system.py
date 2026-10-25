"""
control/system.py - the only place that touches the operating system.

Every backend goes through a `System` object, so tests substitute a recording fake and no test
ever locks a screen or changes a volume. The real one never uses a shell: commands are argument
lists, so nothing a person says can be interpreted as shell syntax.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from typing import Optional, Sequence, Tuple


class System:
    name = sys.platform

    @property
    def os(self) -> str:
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        return "linux"

    def run(self, argv: Sequence[str], timeout: float = 15.0) -> Tuple[int, str]:
        """Run a command to completion. Returns (exit code, stdout); (127, reason) if it cannot start."""
        try:
            cp = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, shell=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return cp.returncode, (cp.stdout or "") + (cp.stderr if cp.returncode else "")
        except FileNotFoundError:
            return 127, f"{argv[0]}: not found"
        except subprocess.TimeoutExpired:
            return 124, f"{argv[0]}: timed out"
        except OSError as e:
            return 126, str(e)

    def which(self, name: str) -> Optional[str]:
        return shutil.which(name)

    def spawn(self, argv: Sequence[str]) -> None:
        """Start a program and leave it running independently of us."""
        kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        subprocess.Popen(list(argv), **kw)

    def startfile(self, target: str) -> None:               # Windows only
        os.startfile(target)                                # type: ignore[attr-defined]  # noqa: S606

    def open_url(self, url: str) -> bool:
        return bool(webbrowser.open(url))

    def press_key(self, vk: int) -> None:                    # Windows only: a real media/volume key press
        import ctypes

        user32 = ctypes.windll.user32                        # type: ignore[attr-defined]
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, 2, 0)                      # KEYEVENTF_KEYUP
