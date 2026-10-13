"""The HTTP layer. Needs FastAPI + httpx (pip install -r requirements-dev.txt); skipped otherwise.
No test here can touch your machine: the device backend is replaced by a recording fake."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from helpers import make_cfg
from test_device_control import FakeSystem

from zivawhite.control.backends import get_backend
from zivawhite.runtime import Runtime
from zivawhite.server import create_app

KEY = "k3y"
LOCAL = ("127.0.0.1", 50000)


def make(tmp_path, monkeypatch, client=LOCAL, key=None):
    if key:
        monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", key)
    else:
        monkeypatch.delenv("ZIVAWHITE_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("ZIVAWHITE_RATE_LIMIT", "0/60")
    rt = Runtime(make_cfg(tmp_path), load_model=False, log=lambda *_: None)
    system = FakeSystem("linux", ["pactl", "loginctl"])
    rt.controller.backend = get_backend(system)
    return TestClient(create_app(rt), client=client), rt, system


def test_health_and_info_work_with_no_model(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    h = c.get("/v1/health").json()
    assert h["status"] == "ok" and h["local_model_loaded"] is False
    info = c.get("/v1/assistant/info").json()
    assert info["creator"] == {"name": "Mahasvin S S", "alias": "Snow White"} and "hey ziva" in info["wake_words"]
    assert c.get("/").status_code == 200 and "Ziva White" in c.get("/").text


def test_ask_answers_the_creator_question_but_never_controls_the_device(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch)
    r = c.post("/v1/ask", json={"query": "who created you?"}).json()
    assert r["route"] == "identity" and "Mahasvin S S" in r["text"]
    c.post("/v1/ask", json={"query": "volume up"})
    assert system.ran == []                                       # /v1/ask has no control path at all


def test_assistant_endpoint_controls_the_device_from_localhost(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch)
    r = c.post("/v1/assistant", json={"text": "set the volume to 30"}).json()
    assert r["kind"] == "control" and r["ok"] and system.ran[-1][-1] == "30%"


def test_remote_callers_without_the_key_cannot_control_but_can_still_ask(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch, client=("192.168.1.50", 40000))
    r = c.post("/v1/assistant", json={"text": "volume up"}).json()
    assert r["ok"] is False and "local or authenticated" in r["text"] and system.ran == []
    assert "Mahasvin S S" in c.post("/v1/assistant", json={"text": "who created you"}).json()["text"]


def test_remote_callers_with_the_key_can_control(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch, client=("192.168.1.50", 40000), key=KEY)
    assert c.post("/v1/assistant", json={"text": "volume up"}).status_code == 401           # key required at all
    r = c.post("/v1/assistant", json={"text": "set the volume to 20"}, headers={"Authorization": f"Bearer {KEY}"}).json()
    assert r["ok"] and system.ran


def test_a_foreign_website_cannot_press_buttons_even_from_localhost_with_the_key(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch, key=KEY)
    hdr = {"Authorization": f"Bearer {KEY}", "Origin": "https://evil.example"}
    r = c.post("/v1/assistant", json={"text": "set the volume to 20"}, headers=hdr).json()
    assert r["ok"] is False and system.ran == []


def test_proxied_requests_are_not_local(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch)
    r = c.post("/v1/assistant", json={"text": "volume up"}, headers={"X-Forwarded-For": "203.0.113.9"}).json()
    assert r["ok"] is False and system.ran == []


def test_confirmation_is_per_session_over_http(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch)
    assert c.post("/v1/assistant", json={"text": "lock the screen", "session": "a"}).json()["kind"] == "confirm"
    assert c.post("/v1/assistant", json={"text": "yes", "session": "b"}).json()["kind"] == "model"
    assert system.ran == []
    assert c.post("/v1/assistant", json={"text": "yes", "session": "a"}).json()["ok"] is True
    assert system.ran[-1] == ["loginctl", "lock-session"]


def test_timer_events_are_polled_by_the_ui(tmp_path, monkeypatch):
    c, rt, _ = make(tmp_path, monkeypatch)
    rt.assistant._speaker_failed = True
    rt.assistant.notify("Your 1 minute timer is done.", kind="timer", speak=False)
    ev = c.get("/v1/assistant/events?since=0").json()["events"]
    assert ev[0]["kind"] == "timer" and c.get(f"/v1/assistant/events?since={ev[0]['id']}").json()["events"] == []


def test_datasets_endpoint_reports_the_folder(tmp_path, monkeypatch):
    c, rt, _ = make(tmp_path, monkeypatch)
    (rt.cfg.datasets_dir / "text").mkdir(parents=True)
    (rt.cfg.datasets_dir / "text" / "a.txt").write_text("hello world")
    d = c.get("/v1/datasets").json()
    assert d["files"] == 1 and d["files_by_folder"]["text"] == 1 and "learner" in d


def test_validation_errors_are_structured_json(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    r = c.post("/v1/assistant", json={"text": ""})
    assert r.status_code == 422 and r.json()["error"] == "invalid request"


def test_voice_stop_is_restricted_to_trusted_callers(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch, client=("192.168.1.50", 40000))
    assert c.post("/v1/voice/stop").status_code == 403
    c, _, _ = make(tmp_path, monkeypatch)
    assert c.post("/v1/voice/stop").json() == {"ok": True}


def test_generate_without_a_model_is_503(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    assert c.post("/v1/generate", json={"prompt": "x"}).status_code == 503
