"""/v1/chat/completions and /v1/models: the OpenAI wire format, streaming included.
Needs FastAPI + httpx (pip install -r requirements-dev.txt); skipped otherwise. The model is a fake router."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from helpers import make_cfg  # noqa: E402

from zivawhite.runtime import Runtime  # noqa: E402
from zivawhite.server import create_app  # noqa: E402


class FakeRouter:
    behavior_tuned = True
    model_info = {}
    connectors = {}

    def __init__(self, reply="Hello there, friend!", loaded=True, finish="stop"):
        self.reply, self.loaded, self.finish, self.calls = reply, loaded, finish, []

    def has_local_model(self):
        return self.loaded

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"text": self.reply, "prompt_tokens": 7, "completion_tokens": 5, "finish_reason": self.finish}

    def chat_stream(self, messages, info=None, **kw):
        self.calls.append((messages, kw))
        for piece in (self.reply[:6], self.reply[6:12], self.reply[12:]):
            if piece:
                yield piece
        if info is not None:
            info.update(finish_reason=self.finish, completion_tokens=5)


def make(tmp_path, monkeypatch, router=None, key=None):
    if key:
        monkeypatch.setenv("ZIVAWHITE_SERVER_API_KEY", key)
    else:
        monkeypatch.delenv("ZIVAWHITE_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("ZIVAWHITE_RATE_LIMIT", "0/60")
    rt = Runtime(make_cfg(tmp_path), load_model=False, log=lambda *_: None)
    rt.router = router if router is not None else FakeRouter()
    return TestClient(create_app(rt), client=("127.0.0.1", 50000)), rt


BODY = {"model": "ziva-white", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}


def sse_events(text):
    events = [e for e in text.split("\n\n") if e.strip()]
    assert all(e.startswith("data: ") for e in events)
    return [e[len("data: "):] for e in events]


def test_non_streaming_reply_has_the_openai_shape(tmp_path, monkeypatch):
    c, rt = make(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion" and j["id"].startswith("chatcmpl-") and j["model"] == "ziva-white"
    assert j["choices"][0]["message"] == {"role": "assistant", "content": "Hello there, friend!"}
    assert j["choices"][0]["finish_reason"] == "stop" and j["choices"][0]["index"] == 0
    assert j["usage"] == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    messages, kw = rt.router.calls[0]
    assert messages[0] == {"role": "system", "content": "Be brief."} and kw["max_new_tokens"] == 256


def test_request_parameters_reach_the_model(tmp_path, monkeypatch):
    c, rt = make(tmp_path, monkeypatch)
    c.post("/v1/chat/completions", json={**BODY, "max_tokens": 40, "temperature": 0.3, "top_p": 0.8, "stop": "END"})
    _, kw = rt.router.calls[0]
    assert (kw["max_new_tokens"], kw["temperature"], kw["top_p"], kw["stop"]) == (40, 0.3, 0.8, ["END"])
    c.post("/v1/chat/completions", json={**BODY, "stop": ["a", "b"]})
    assert rt.router.calls[1][1]["stop"] == ["a", "b"]


def test_streaming_sends_role_then_content_deltas_then_finish_then_done(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch)
    r = c.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(ch["object"] == "chat.completion.chunk" and ch["id"] == chunks[0]["id"] for ch in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks) == "Hello there, friend!"
    assert chunks[-1]["choices"][0]["delta"] == {} and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(ch["choices"][0]["finish_reason"] is None for ch in chunks[:-1])


def test_a_reply_that_ran_out_of_tokens_reports_finish_reason_length(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch, router=FakeRouter(finish="length"))
    assert c.post("/v1/chat/completions", json=BODY).json()["choices"][0]["finish_reason"] == "length"
    chunks = [json.loads(e) for e in sse_events(c.post("/v1/chat/completions", json={**BODY, "stream": True}).text)[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_bad_requests_are_400_in_openai_error_shape_before_any_streaming_starts(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch)
    for stream in (False, True):
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "x"}], "stream": stream})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
        assert "last message" in r.json()["error"]["message"]
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "wizard", "content": "x"}]})
    assert r.status_code == 400 and "role" in r.json()["error"]["message"]
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_no_model_loaded_is_a_503(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch, router=FakeRouter(loaded=False))
    r = c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 503 and r.json()["error"]["type"] == "server_error"


def test_models_endpoint_lists_the_model(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch)
    j = c.get("/v1/models").json()
    assert j["object"] == "list" and j["data"][0]["id"] == "ziva-white" and j["data"][0]["object"] == "model"


def test_the_bearer_key_protects_the_openai_endpoints_like_every_other(tmp_path, monkeypatch):
    c, _ = make(tmp_path, monkeypatch, key="k3y")
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401
    assert c.get("/v1/models").status_code == 401
    ok = c.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer k3y"})
    assert ok.status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer k3y"}).status_code == 200


def test_this_endpoint_can_only_generate_text_it_never_reaches_the_device(tmp_path, monkeypatch):
    c, rt = make(tmp_path, monkeypatch)
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "volume up"}]})
    # the text went to the language model verbatim; no control path exists on this route
    assert rt.router.calls[0][0] == [{"role": "user", "content": "volume up"}]
