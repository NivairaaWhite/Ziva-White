"""Background running: the supervisor (restart, backoff, give-up, unavailable), heartbeat/status,
real cross-process start/stop, log rotation and the autostart files for each OS."""
import json
import os
import plistlib
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from helpers import make_cfg

from zivawhite import daemon
from zivawhite.daemon import (Supervisor, daemon_paths, daemon_status, install_autostart, render_launchd_plist,
                              render_systemd_unit, rotate_log, stop_daemon, uninstall_autostart, windows_task_command)
from zivawhite.voice.errors import VoiceUnavailable


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def sup(jobs, **kw):
    kw.setdefault("backoff", lambda n: 0.0)
    kw.setdefault("log", lambda *_: None)
    kw.setdefault("heartbeat_s", 0.05)
    return Supervisor(jobs, **kw)


# ----------------------------------------------------------------------------- supervisor
def test_a_job_runs_until_asked_to_stop():
    started = threading.Event()
    def job(stop): started.set(); stop.wait()
    s = sup({"a": job})
    s.start()
    assert started.wait(2) and s.status["a"].state == "running"
    s.stop(timeout=3)
    assert s.status["a"].state == "stopped"


def test_a_crashing_job_is_restarted_and_then_runs():
    calls = []
    def flaky(stop):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        stop.wait()
    s = sup({"a": flaky})
    s.start()
    assert wait_for(lambda: len(calls) == 3 and s.status["a"].state == "running")
    assert s.status["a"].restarts == 2 and "boom" in s.status["a"].last_error
    s.stop(timeout=3)


def test_a_job_that_keeps_crashing_is_given_up_on_not_looped_forever():
    calls = []
    def always(stop): calls.append(1); raise RuntimeError("nope")
    s = sup({"a": always}, max_restarts=3)
    s.start()
    assert wait_for(lambda: s.status["a"].state == "failed")
    assert len(calls) == 4                                      # first try + 3 restarts
    s.stop(timeout=3)


def test_long_healthy_runs_reset_the_crash_counter():
    t = {"now": 1000.0}
    n = []
    def job(stop):
        n.append(1)
        t["now"] += 1000                                         # it ran "for a long time" before dying
        if len(n) < 6:
            raise RuntimeError("rare crash")
        stop.wait()
    s = sup({"a": job}, max_restarts=2, healthy_after_s=300, clock=lambda: t["now"])
    s.start()
    assert wait_for(lambda: len(n) == 6 and s.status["a"].state == "running")     # 5 crashes > max_restarts, but never consecutive
    s.stop(timeout=3)


def test_unavailable_jobs_are_reported_and_not_retried():
    calls = []
    def voice(stop): calls.append(1); raise VoiceUnavailable("No Vosk model at models/vosk/model")
    s = sup({"voice": voice})
    s.start()
    assert wait_for(lambda: s.status["voice"].state == "unavailable")
    time.sleep(0.2)
    assert len(calls) == 1 and "Vosk" in s.status["voice"].last_error and "not retried" in s.status["voice"].detail
    s.stop(timeout=3)


def test_one_failing_job_does_not_affect_the_others():
    ok = threading.Event()
    def bad(stop): raise RuntimeError("x")
    def good(stop): ok.set(); stop.wait()
    s = sup({"bad": bad, "good": good}, max_restarts=0)
    s.start()
    assert ok.wait(2) and wait_for(lambda: s.status["bad"].state == "failed")
    assert s.status["good"].state == "running"
    s.stop(timeout=3)


def test_stop_during_backoff_exits_promptly():
    def always(stop): raise RuntimeError("x")
    s = sup({"a": always}, backoff=lambda n: 60.0)
    s.start()
    assert wait_for(lambda: s.status["a"].state == "restarting")
    t0 = time.time()
    s.stop(timeout=5)
    assert time.time() - t0 < 2 and s.status["a"].state == "stopped"


def test_heartbeat_writes_status_atomically(tmp_path):
    path = tmp_path / "status.json"
    s = sup({"a": lambda stop: stop.wait()}, status_path=path, extra_status=lambda: {"model": "m"})
    s.start()
    assert wait_for(path.exists)
    data = json.loads(path.read_text())
    assert data["pid"] == os.getpid() and data["jobs"]["a"]["state"] in ("starting", "running") and data["runtime"] == {"model": "m"}
    s.stop(timeout=3)
    assert json.loads(path.read_text())["stopped"] is True
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".status_")]      # no temp files left behind


def test_stop_file_requests_a_clean_shutdown(tmp_path):
    stop_file = tmp_path / "daemon.stop"
    asked = []
    s = sup({"a": lambda stop: stop.wait()}, stop_file=stop_file, on_stop_request=lambda: asked.append(1))
    s.start()
    stop_file.write_text("x")
    assert wait_for(s.stop_event.is_set) and asked == [1] and not stop_file.exists()
    s.stop(timeout=3)


def test_a_broken_status_provider_does_not_kill_the_heartbeat(tmp_path):
    def broken(): raise RuntimeError("no")
    s = sup({"a": lambda stop: stop.wait()}, status_path=tmp_path / "s.json", extra_status=broken)
    s.start()
    assert wait_for(lambda: (tmp_path / "s.json").exists())
    assert "error" in json.loads((tmp_path / "s.json").read_text())["runtime"]
    s.stop(timeout=3)


# ------------------------------------------------------------------- processes (for real)
FAKE_DAEMON = r'''
import json, os, sys, time, pathlib
pid_file, status_file, stop_file = map(pathlib.Path, sys.argv[2:5])   # argv[1] == "zivawhite" so the identity check passes
pid_file.write_text(str(os.getpid()))
while not stop_file.exists():
    status_file.write_text(json.dumps({"pid": os.getpid(), "started_at": time.time() - 5, "heartbeat": time.time(), "jobs": {}}))
    time.sleep(0.05)
'''


def spawn_fake(cfg):
    p = daemon_paths(cfg)
    for x in p.values():
        x.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", FAKE_DAEMON, "zivawhite", str(p["pid"]), str(p["status"]), str(p["stop"])])
    assert wait_for(p["status"].exists)
    return proc


def test_status_stop_roundtrip_with_a_real_process(tmp_path):
    cfg = make_cfg(tmp_path)
    assert daemon_status(cfg)["state"] == "stopped"
    proc = spawn_fake(cfg)
    try:
        st = daemon_status(cfg)
        assert st["state"] == "running" and st["pid"] == proc.pid and st["uptime_s"] >= 5
        ok, msg = stop_daemon(cfg, timeout=10)
        assert ok and "stopped" in msg and "force" not in msg           # a clean stop-file shutdown, not a kill
        assert proc.wait(timeout=5) == 0
        assert daemon_status(cfg)["state"] == "stopped"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_an_unresponsive_process_is_force_killed(tmp_path):
    cfg = make_cfg(tmp_path)
    p = daemon_paths(cfg)
    p["pid"].parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(1)", "zivawhite"])
    p["pid"].write_text(str(proc.pid))
    try:
        ok, msg = stop_daemon(cfg, timeout=1.0)
        assert ok and "force-killed" in msg
        assert proc.wait(timeout=5) is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_stale_pid_file_is_not_mistaken_for_a_running_daemon(tmp_path):
    cfg = make_cfg(tmp_path)
    p = daemon_paths(cfg)
    p["pid"].parent.mkdir(parents=True, exist_ok=True)
    p["pid"].write_text("999999")
    assert daemon_status(cfg)["state"] == "stale"
    assert stop_daemon(cfg)[1] == "not running" and not p["pid"].exists()


def test_a_reused_pid_belonging_to_another_program_is_left_alone(tmp_path):
    cfg = make_cfg(tmp_path)
    p = daemon_paths(cfg)
    p["pid"].parent.mkdir(parents=True, exist_ok=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])      # no "zivawhite" in its command line
    try:
        p["pid"].write_text(str(other.pid))
        assert daemon_status(cfg)["state"] == "stale"
        assert stop_daemon(cfg)[1] == "not running"
        assert other.poll() is None                                                       # we did not touch it
    finally:
        other.kill()


def test_start_refuses_a_second_copy(tmp_path):
    cfg = make_cfg(tmp_path)
    proc = spawn_fake(cfg)
    try:
        ok, msg = daemon.start_daemon(cfg)
        assert not ok and "already running" in msg
    finally:
        stop_daemon(cfg, timeout=10)
        proc.wait(timeout=5)


def test_start_launches_detached_and_reports_an_early_crash(tmp_path):
    cfg = make_cfg(tmp_path)
    seen = {}
    class Dead:
        returncode, pid = 3, 4242
        def poll(self): return 3
    def fake_popen(argv, **kw):
        seen.update(argv=argv, kw=kw)
        daemon_paths(cfg)["log"].write_bytes(b"Traceback: something broke\n")
        return Dead()
    ok, msg = daemon.start_daemon(cfg, python="PY", popen=fake_popen)
    assert not ok and "exited immediately" in msg and "something broke" in msg
    assert seen["argv"] == ["PY", "-m", "zivawhite.app", "run", "--daemon"]
    assert seen["kw"]["stdin"] == subprocess.DEVNULL and seen["kw"]["cwd"] == str(cfg.root)
    assert seen["kw"].get("start_new_session") or seen["kw"].get("creationflags")        # detached from the terminal


# ------------------------------------------------------------------------------ logging
def test_log_rotation(tmp_path):
    log = tmp_path / "d.log"
    log.write_bytes(b"x" * 100)
    rotate_log(log, max_bytes=1000, backups=3)
    assert log.exists() and not (tmp_path / "d.log.1").exists()                         # small: left alone
    for gen in range(4):
        log.write_bytes(f"gen{gen}".encode() + b"x" * 2000)
        rotate_log(log, max_bytes=1000, backups=3)
    assert not log.exists()
    assert (tmp_path / "d.log.1").read_bytes().startswith(b"gen3") and (tmp_path / "d.log.3").read_bytes().startswith(b"gen1")
    assert not (tmp_path / "d.log.4").exists()                                           # only `backups` generations kept


def test_tail_log(tmp_path):
    cfg = make_cfg(tmp_path)
    assert daemon.tail_log(cfg) == ""
    p = daemon_paths(cfg)["log"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    assert daemon.tail_log(cfg, 3).splitlines() == ["line 97", "line 98", "line 99"]


def test_stream_to_logger_captures_print_output():
    import logging
    records = []
    class H(logging.Handler):
        def emit(self, r): records.append(r.getMessage())
    lg = logging.getLogger("t.stream"); lg.addHandler(H()); lg.setLevel(logging.INFO)
    s = daemon._StreamToLogger(lg, logging.INFO)
    s.write("hello "); s.write("world\nsecond\n\n"); s.write("partial"); s.flush()
    assert records == ["hello world", "second", "partial"] and s.isatty() is False


# ----------------------------------------------------------------------------- autostart
def test_systemd_unit_contents():
    unit = render_systemd_unit(Path("/opt/ziva white"), "/usr/bin/python3")
    assert 'ExecStart="/usr/bin/python3" -m zivawhite.app run --daemon' in unit
    assert "WorkingDirectory=/opt/ziva white" in unit and "Restart=on-failure" in unit and "WantedBy=default.target" in unit


def test_launchd_plist_is_valid_and_complete():
    plist = plistlib.loads(render_launchd_plist(Path("/Users/me/ziva"), "/usr/bin/python3", Path("/tmp/d.log")))
    assert plist["Label"] == "com.zivawhite.assistant" and plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == ["/usr/bin/python3", "-m", "zivawhite.app", "run", "--daemon"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["StandardOutPath"] == "/tmp/d.log"


def test_windows_task_command():
    cmd = windows_task_command(Path("C:/Ziva White"), "C:/Python/python.exe")
    assert cmd[:6] == ["schtasks", "/Create", "/TN", "ZivaWhite", "/SC", "ONLOGON"] and cmd[-1] == "/F"
    tr = cmd[cmd.index("/TR") + 1]
    assert "zivawhite.app run --daemon" in tr and "Ziva White" in tr


class FakeRun:
    def __init__(self, rc=0): self.calls, self.rc = [], rc
    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.rc, "", "denied" if self.rc else "")


def test_install_and_uninstall_linux(tmp_path):
    cfg = make_cfg(tmp_path)
    run = FakeRun()
    ok, msg = install_autostart(cfg, "linux", tmp_path, "/py", run)
    unit = tmp_path / ".config/systemd/user/ziva-white.service"
    assert ok and unit.exists() and ["systemctl", "--user", "enable", "--now", "ziva-white.service"] in run.calls
    ok, msg = uninstall_autostart(cfg, "linux", tmp_path, run)
    assert ok and not unit.exists()


def test_install_macos_and_windows(tmp_path):
    cfg = make_cfg(tmp_path)
    run = FakeRun()
    ok, _ = install_autostart(cfg, "macos", tmp_path, "/py", run)
    assert ok and (tmp_path / "Library/LaunchAgents/com.zivawhite.assistant.plist").exists()
    assert any(c[:2] == ["launchctl", "load"] for c in run.calls)
    run = FakeRun()
    ok, msg = install_autostart(cfg, "windows", tmp_path, "C:/py/python.exe", run)
    assert ok and run.calls[0][0] == "schtasks" and "ZivaWhite" in msg


def test_install_failure_is_reported_not_swallowed(tmp_path):
    cfg = make_cfg(tmp_path)
    ok, msg = install_autostart(cfg, "linux", tmp_path, "/py", FakeRun(rc=1))
    assert not ok and "failed" in msg and "denied" in msg
    assert install_autostart(cfg, "plan9", tmp_path, "/py", FakeRun())[0] is False
