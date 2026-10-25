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


def make(tmp_path, monkeypatch, client=LOCAL, key=None, rate="0/60"):
    if key:
        monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", key)
    else:
        monkeypatch.delenv("ZIVAWHITE_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("ZIVAWHITE_RATE_LIMIT", rate)          # "0/60" = the general limiter is off
    rt = Runtime(make_cfg(tmp_path), load_model=False, log=lambda *_: None)
    system = FakeSystem("linux", ["pactl", "loginctl"])
    rt.controller.backend = get_backend(system)
    return TestClient(create_app(rt), base_url="http://localhost", client=client), rt, system


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
    first = c.post("/v1/assistant", json={"text": "lock the screen"}).json()
    assert first["kind"] == "confirm"
    mine = first["session"]                                   # the server issued this conversation's id
    assert c.post("/v1/assistant", json={"text": "yes", "session": "made-up-by-someone"}).json()["kind"] == "model"
    assert system.ran == []
    assert c.post("/v1/assistant", json={"text": "yes", "session": mine}).json()["ok"] is True
    assert system.ran[-1] == ["loginctl", "lock-session"]


def test_a_caller_cannot_pick_the_voice_or_cli_session_to_answer_someone_elses_question(tmp_path, monkeypatch):
    c, rt, system = make(tmp_path, monkeypatch)
    rt.assistant.handle("lock the screen", session="voice")          # the voice loop asks "are you sure?"
    r = c.post("/v1/assistant", json={"text": "yes", "session": "voice"}).json()
    assert r["kind"] == "model" and system.ran == []                 # "voice" means nothing to the HTTP layer
    assert r["session"] not in ("voice", "cli")


def test_every_assistant_reply_carries_a_server_issued_session_and_reuses_it(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    a = c.post("/v1/assistant", json={"text": "who created you"}).json()["session"]
    assert len(a) >= 16
    assert c.post("/v1/assistant", json={"text": "who created you", "session": a}).json()["session"] == a


def test_dns_rebinding_is_refused_everywhere(tmp_path, monkeypatch):
    c, _, system = make(tmp_path, monkeypatch)
    evil = {"Host": "evil.example:8000", "Origin": "http://evil.example:8000"}
    assert c.get("/v1/health", headers=evil).status_code == 400            # it cannot even read answers
    assert c.post("/v1/assistant", json={"text": "set the volume to 20"}, headers=evil).status_code == 400
    assert system.ran == []
    monkeypatch.setenv("ZIVAWHITE_ALLOWED_HOSTS", "evil.example")           # ... unless you explicitly trust the name
    assert c.get("/v1/health", headers=evil).status_code == 200


def test_an_unknown_route_is_rejected_instead_of_silently_meaning_web(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    assert c.post("/v1/ask", json={"query": "hi", "route": "banana"}).status_code == 422
    assert c.post("/v1/assistant", json={"text": "hi", "route": "banana"}).status_code == 422
    assert c.post("/v1/ask", json={"query": "who created you", "route": "local"}).status_code == 200


def test_an_oversized_body_is_413(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_MAX_BODY_BYTES", "1000")
    c, _, _ = make(tmp_path, monkeypatch)
    r = c.post("/v1/ask", json={"query": "x" * 5000})
    assert r.status_code == 413


def test_the_web_ui_is_served_with_a_csp_and_every_response_with_nosniff(tmp_path, monkeypatch):
    c, _, _ = make(tmp_path, monkeypatch)
    page = c.get("/")
    assert "default-src 'none'" in page.headers["content-security-policy"] and "'unsafe-inline'" not in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert c.get("/v1/health").headers["x-content-type-options"] == "nosniff"


def test_repeated_wrong_keys_are_locked_out_before_the_right_one_is_tried(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "3/60")
    c, _, _ = make(tmp_path, monkeypatch, key=KEY)
    for i in range(3):
        assert c.get("/v1/status", headers={"Authorization": f"Bearer wrong{i}"}).status_code == 401
    assert c.get("/v1/status", headers={"Authorization": "Bearer wrong3"}).status_code == 429
    assert c.get("/v1/status", headers={"Authorization": f"Bearer {KEY}"}).status_code == 429     # locked out first


def test_the_limiter_runs_before_the_key_check_so_failed_logins_on_ask_are_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("ZIVAWHITE_AUTH_FAIL_LIMIT", "0/60")           # switch the lock-out off to isolate the general limiter
    c, _, _ = make(tmp_path, monkeypatch, key=KEY, rate="3/60")
    codes = [c.post("/v1/ask", json={"query": "hi"}, headers={"Authorization": "Bearer nope"}).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]   # before the fix a 401 came first, so the limiter never saw them


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
