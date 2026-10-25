"""
daemon.py - keeping the assistant running in the background.

Three layers:

  Supervisor      runs each background job (datasets watcher, voice loop, timers) in its own
                  thread. A job that crashes is restarted with exponential backoff; one that keeps
                  crashing is marked failed instead of looping forever; one that cannot work on this
                  machine (no microphone, no speech model) is marked unavailable with the reason and
                  is NOT retried, because retrying cannot install a package. A heartbeat thread
                  writes runs/daemon_status.json every few seconds so anything can see what is alive.

  start/stop/...  `daemon start` launches `python -m zivawhite.app run --daemon` fully detached
                  (no terminal needed), `daemon stop` asks it to shut down cleanly through a stop
                  file -- the one mechanism that works the same on Windows, macOS and Linux -- and
                  only force-kills the process tree if it does not comply.

  autostart       `daemon install` registers it to start at login: a systemd user service on Linux,
                  a LaunchAgent on macOS, a Task Scheduler logon task on Windows.

Everything (paths, intervals, restart limits, log rotation) comes from `background:` in the config.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import plistlib
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config
from .voice.errors import VoiceUnavailable

PERMANENT_ERRORS: Tuple[type, ...] = (VoiceUnavailable,)
SERVICE_NAME = "ziva-white"
LAUNCHD_LABEL = "com.zivawhite.assistant"
TASK_NAME = "ZivaWhite"


# ================================================================================= supervisor
@dataclass
class JobStatus:
    name: str
    state: str = "starting"        # starting | running | restarting | stopped | failed | unavailable
    restarts: int = 0
    last_error: Optional[str] = None
    started_at: Optional[float] = None
    detail: Optional[str] = None


class Supervisor:
    def __init__(self, jobs: Dict[str, Callable[[threading.Event], None]], max_restarts: int = 5,
                 heartbeat_s: float = 5.0, status_path: Optional[Path] = None, stop_file: Optional[Path] = None,
                 extra_status: Optional[Callable[[], dict]] = None, on_stop_request: Optional[Callable[[], None]] = None,
                 log: Callable[[str], None] = print, backoff: Callable[[int], float] = lambda n: min(60.0, 2.0 ** n),
                 healthy_after_s: float = 300.0, clock: Callable[[], float] = time.time):
        self.jobs = dict(jobs)
        self.max_restarts = max_restarts
        self.heartbeat_s = heartbeat_s
        self.status_path = Path(status_path) if status_path else None
        self.stop_file = Path(stop_file) if stop_file else None
        self.extra_status = extra_status
        self.on_stop_request = on_stop_request
        self.log = log
        self.backoff = backoff
        self.healthy_after_s = healthy_after_s
        self._clock = clock
        self.stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self.status: Dict[str, JobStatus] = {n: JobStatus(n) for n in self.jobs}
        self.started_at = clock()

    # -- one job's life -----------------------------------------------------------------------------
    def _supervise(self, name: str, fn: Callable[[threading.Event], None]) -> None:
        st = self.status[name]
        while not self.stop_event.is_set():
            began = self._clock()
            with self._lock:
                st.state, st.started_at = "running", began
            try:
                fn(self.stop_event)
                with self._lock:
                    st.state = "stopped"
                return
            except PERMANENT_ERRORS as e:
                with self._lock:
                    st.state, st.last_error, st.detail = "unavailable", str(e), "not retried: fix the cause and restart"
                self.log(f"[daemon] {name} unavailable: {e}")
                return
            except Exception as e:  # noqa: BLE001 - a job crashing is exactly what this exists to contain
                if self.stop_event.is_set():
                    break
                with self._lock:
                    if self._clock() - began >= self.healthy_after_s:
                        st.restarts = 0                      # it ran healthily for a long time: not a crash loop
                    st.restarts += 1
                    st.last_error = f"{type(e).__name__}: {e}"
                    over = st.restarts > self.max_restarts
                    st.state = "failed" if over else "restarting"
                self.log(f"[daemon] {name} crashed ({st.last_error}); "
                         + ("giving up" if over else f"restart {st.restarts}/{self.max_restarts}"))
                if over:
                    return
                self.stop_event.wait(self.backoff(st.restarts))
        with self._lock:
            st.state = "stopped"

    # -- lifecycle ----------------------------------------------------------------------------------------
    def start(self) -> None:
        for name, fn in self.jobs.items():
            t = threading.Thread(target=self._supervise, args=(name, fn), name=f"job-{name}", daemon=True)
            t.start()
            self._threads.append(t)
        hb = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        hb.start()
        self._threads.append(hb)

    def request_stop(self, why: str = "") -> None:
        if not self.stop_event.is_set():
            self.log(f"[daemon] stopping{': ' + why if why else ''}")
            self.stop_event.set()
            if self.on_stop_request:
                try:
                    self.on_stop_request()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self, timeout: float = 30.0) -> None:
        self.request_stop()
        deadline = self._clock() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - self._clock()))
        self._write_status(final=True)

    # -- status -------------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            jobs = {n: asdict(s) for n, s in self.status.items()}
        out = {"pid": os.getpid(), "started_at": self.started_at, "heartbeat": self._clock(), "jobs": jobs}
        if self.extra_status:
            try:
                out["runtime"] = self.extra_status()
            except Exception as e:  # noqa: BLE001
                out["runtime"] = {"error": str(e)}
        return out

    def _write_status(self, final: bool = False) -> None:
        if not self.status_path:
            return
        snap = self.snapshot()
        if final:
            snap["stopped"] = True
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.status_path.parent), prefix=".status_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2, default=str)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    def _heartbeat(self) -> None:
        while not self.stop_event.is_set():
            self._write_status()
            if self.stop_file and self.stop_file.exists():
                try:
                    self.stop_file.unlink()
                except OSError:
                    pass
                self.request_stop("stop requested")
                return
            self.stop_event.wait(self.heartbeat_s)


# ===================================================================================== paths
def daemon_paths(cfg: Config) -> Dict[str, Path]:
    bg = cfg.section("background")
    return {"status": cfg.resolve(bg.get("status_file", "runs/daemon_status.json")),
            "pid": cfg.resolve(bg.get("pid_file", "runs/daemon.pid")),
            "log": cfg.resolve(bg.get("log_file", "runs/daemon.log")),
            "stop": cfg.resolve(bg.get("pid_file", "runs/daemon.pid")).with_suffix(".stop")}


# ====================================================================================== logging
class _StreamToLogger:
    """File-like object so print() and library stderr output land in the rotated log."""
    encoding = "utf-8"

    def __init__(self, logger: logging.Logger, level: int):
        self.logger, self.level, self._buf = logger, level, ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.log(self.level, self._buf.rstrip())
        self._buf = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("not a real file")


def setup_daemon_logging(cfg: Config) -> Path:
    """Send everything (logging, print, uncaught stderr) to a size-rotated log file."""
    bg = cfg.section("background")
    path = daemon_paths(cfg)["log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(str(path), maxBytes=int(float(bg.get("log_max_mb", 5)) * 1024 * 1024),
                                                   backupCount=int(bg.get("log_backups", 3)), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    sys.stdout = _StreamToLogger(logging.getLogger("ziva"), logging.INFO)       # type: ignore[assignment]
    sys.stderr = _StreamToLogger(logging.getLogger("ziva.stderr"), logging.WARNING)   # type: ignore[assignment]
    return path


def tail_log(cfg: Config, lines: int = 50) -> str:
    path = daemon_paths(cfg)["log"]
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


# =============================================================================== processes
def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def _looks_like_ziva(pid: int) -> bool:
    """Guard against a stale pid file whose number was reused by an unrelated process."""
    try:
        import psutil

        return "zivawhite" in " ".join(psutil.Process(pid).cmdline()).lower()
    except Exception:  # noqa: BLE001
        return False


def read_pid(cfg: Config) -> Optional[int]:
    try:
        return int(daemon_paths(cfg)["pid"].read_text().strip())
    except (OSError, ValueError):
        return None


def write_pid(cfg: Config) -> None:
    p = daemon_paths(cfg)["pid"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))


def clear_pid(cfg: Config) -> None:
    p = daemon_paths(cfg)["pid"]
    try:
        if p.exists() and p.read_text().strip() == str(os.getpid()):
            p.unlink()
    except OSError:
        pass


def daemon_status(cfg: Config, verify: Callable[[int], bool] = _looks_like_ziva, now: Callable[[], float] = time.time) -> dict:
    paths = daemon_paths(cfg)
    pid = read_pid(cfg)
    alive = pid is not None and _pid_alive(pid) and verify(pid)
    status = None
    try:
        status = json.loads(paths["status"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if not alive:
        return {"state": "stopped" if pid is None else "stale", "pid": pid, "log": str(paths["log"]),
                "note": "a pid file was left behind by a process that is gone" if pid is not None else None}
    hb_s = float(cfg.get("background.heartbeat_s", 5))
    age = now() - float(status.get("heartbeat", 0)) if status else None
    state = "running" if status and age is not None and age < max(30.0, hb_s * 6) else "unresponsive"
    return {"state": state, "pid": pid, "uptime_s": round(now() - status["started_at"]) if status else None,
            "heartbeat_age_s": round(age, 1) if age is not None else None,
            "jobs": status.get("jobs") if status else None, "runtime": status.get("runtime") if status else None,
            "log": str(paths["log"])}


def rotate_log(path: Path, max_bytes: int, backups: int) -> None:
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    for i in range(backups - 1, 0, -1):
        src, dst = path.with_name(f"{path.name}.{i}"), path.with_name(f"{path.name}.{i + 1}")
        if src.exists():
            os.replace(src, dst)
    os.replace(path, path.with_name(f"{path.name}.1"))


def start_daemon(cfg: Config, python: Optional[str] = None, popen=subprocess.Popen, wait_s: float = 12.0,
                 verify: Callable[[int], bool] = _looks_like_ziva) -> Tuple[bool, str]:
    st = daemon_status(cfg, verify=verify)
    if st["state"] in ("running", "unresponsive"):
        return False, f"already running (pid {st['pid']})"
    paths = daemon_paths(cfg)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    for stale in (paths["stop"], paths["status"]):
        try:
            stale.unlink()
        except OSError:
            pass
    bg = cfg.section("background")
    rotate_log(paths["log"], int(float(bg.get("log_max_mb", 5)) * 1024 * 1024), int(bg.get("log_backups", 3)))
    argv = [python or sys.executable, "-m", "zivawhite.app", "run", "--daemon"]
    kw: dict = {"cwd": str(cfg.root), "stdin": subprocess.DEVNULL, "close_fds": True}
    log_f = open(paths["log"], "ab")
    kw.update(stdout=log_f, stderr=subprocess.STDOUT)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000     # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    try:
        proc = popen(argv, **kw)
    finally:
        log_f.close()
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"exited immediately (code {proc.returncode}). Last log lines:\n{tail_log(cfg, 15)}"
        if paths["status"].exists():
            return True, f"started (pid {proc.pid}); log: {paths['log']}"
        time.sleep(0.2)
    return True, f"started (pid {proc.pid}) but has not reported status yet; check: {paths['log']}"


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def stop_daemon(cfg: Config, timeout: float = 30.0, verify: Callable[[int], bool] = _looks_like_ziva) -> Tuple[bool, str]:
    pid = read_pid(cfg)
    paths = daemon_paths(cfg)
    if pid is None or not _pid_alive(pid) or not verify(pid):
        try:
            paths["pid"].unlink()
        except OSError:
            pass
        return False, "not running"
    paths["stop"].parent.mkdir(parents=True, exist_ok=True)
    paths["stop"].write_text(str(time.time()))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            for p in (paths["pid"], paths["stop"]):
                try:
                    p.unlink()
                except OSError:
                    pass
            return True, f"stopped (pid {pid})"
        time.sleep(0.25)
    _kill_tree(pid)                                  # did not shut down cleanly: end it and its training child
    for p in (paths["pid"], paths["stop"]):
        try:
            p.unlink()
        except OSError:
            pass
    return True, f"did not stop within {timeout:.0f}s; force-killed pid {pid} and its children"


# =============================================================================== autostart
def render_systemd_unit(root: Path, python: str) -> str:
    return (f"[Unit]\nDescription=Ziva White assistant\nAfter=network-online.target sound.target\n\n"
            f"[Service]\nType=simple\nWorkingDirectory={root.as_posix()}\nExecStart=\"{python}\" -m zivawhite.app run --daemon\n"
            f"Restart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n")


def render_launchd_plist(root: Path, python: str, log_path: Path) -> bytes:
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [python, "-m", "zivawhite.app", "run", "--daemon"],
        "WorkingDirectory": root.as_posix(),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": log_path.as_posix(),
        "StandardErrorPath": log_path.as_posix(),
    })


def windows_task_command(root: Path, python: str) -> List[str]:
    pyw = Path(python).with_name("pythonw.exe")
    exe = str(pyw if pyw.exists() else python)
    tr = f'cmd /c start "" /D "{root}" /B "{exe}" -m zivawhite.app run --daemon'
    return ["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "ONLOGON", "/TR", tr, "/F"]


def autostart_files(os_name: str, home: Path) -> Path:
    if os_name == "linux":
        return home / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
    if os_name == "macos":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    raise ValueError(os_name)


def _os_name() -> str:
    return "windows" if sys.platform.startswith("win") else "macos" if sys.platform == "darwin" else "linux"


def install_autostart(cfg: Config, os_name: Optional[str] = None, home: Optional[Path] = None, python: Optional[str] = None,
                      run: Callable = subprocess.run) -> Tuple[bool, str]:
    os_name, home, python = os_name or _os_name(), home or Path.home(), python or sys.executable
    root = cfg.root
    if os_name == "linux":
        unit = autostart_files("linux", home)
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(render_systemd_unit(root, python), encoding="utf-8")
        cmds = [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME}.service"]]
        where = f"{unit}  (starts at login; stop with: systemctl --user stop {SERVICE_NAME})"
    elif os_name == "macos":
        plist = autostart_files("macos", home)
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_bytes(render_launchd_plist(root, python, daemon_paths(cfg)["log"]))
        cmds = [["launchctl", "unload", str(plist)], ["launchctl", "load", "-w", str(plist)]]
        where = str(plist)
    elif os_name == "windows":
        cmds = [windows_task_command(root, python)]
        where = f"Task Scheduler task '{TASK_NAME}' (runs at logon)"
    else:
        return False, f"unsupported OS {os_name!r}"
    for cmd in cmds:
        try:
            cp = run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"could not run {cmd[0]}: {e}"
        if cp.returncode != 0 and cmd[1:2] != ["unload"]:
            return False, f"`{' '.join(cmd[:3])}` failed: {(cp.stderr or cp.stdout).strip()[:200]}"
    return True, f"installed: {where}"


def uninstall_autostart(cfg: Config, os_name: Optional[str] = None, home: Optional[Path] = None,
                        run: Callable = subprocess.run) -> Tuple[bool, str]:
    os_name, home = os_name or _os_name(), home or Path.home()
    if os_name == "linux":
        unit = autostart_files("linux", home)
        run(["systemctl", "--user", "disable", "--now", f"{SERVICE_NAME}.service"], capture_output=True, text=True)
        existed = unit.exists()
        unit.unlink(missing_ok=True)
        run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True)
        return True, "removed" if existed else "was not installed"
    if os_name == "macos":
        plist = autostart_files("macos", home)
        run(["launchctl", "unload", str(plist)], capture_output=True, text=True)
        existed = plist.exists()
        plist.unlink(missing_ok=True)
        return True, "removed" if existed else "was not installed"
    if os_name == "windows":
        cp = run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True, text=True)
        return True, "removed" if cp.returncode == 0 else "was not installed"
    return False, f"unsupported OS {os_name!r}"
