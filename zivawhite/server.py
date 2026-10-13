"""
server.py - the HTTP API and web UI.

    GET  /                          web UI (chat, voice, hands-free mode)
    POST /v1/ask        {"query", "route"}                -> routed answer (identity-aware); never touches the device
    POST /v1/generate   {"prompt", ...}                   -> raw local-model sampling
    POST /v1/chat/completions                             -> OpenAI-compatible chat (messages, stream:true for SSE)
    GET  /v1/models                                       -> OpenAI-compatible model list
    POST /v1/assistant  {"text", "session", "route"}      -> the full assistant: identity, DEVICE CONTROL, model
    GET  /v1/assistant/info                               -> names, wake words (what the web UI needs)
    GET  /v1/assistant/events?since=N                     -> background notifications (timers)
    POST /v1/voice/stop                                   -> stop speaking
    GET  /v1/datasets                                     -> what is in the datasets folder / what still needs training
    GET  /v1/status                                       -> background jobs, model, learning state
    GET  /v1/health, /v1/version

/v1/chat/completions speaks the OpenAI wire format, so any OpenAI client works against it unchanged:
    from openai import OpenAI; client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything-or-your-key")
    client.chat.completions.create(model="ziva-white", messages=[{"role": "user", "content": "hi"}], stream=True)
It only ever generates text from the local model (like /v1/ask it can never touch the device).

Device control is only reachable through /v1/assistant and only for local or API-key callers whose
Origin is not a foreign website (access.py). /v1/ask can answer "who created you?" but can never
act on the machine.

The request models live at module level, not inside create_app(): with `from __future__ import
annotations`, FastAPI resolves annotations through the function's __globals__ and never sees a
class defined in an enclosing scope, so it would treat `body` as a query parameter.
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
import uuid
from typing import Callable, List, Optional, Union

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .access import control_allowed
from .runtime import Runtime
from .security import get_server_api_key, make_rate_limit_dependency, require_api_key

log = logging.getLogger("zivawhite")


class AskBody(BaseModel):
    # No max_length: the router/tokenizer handle arbitrarily long input (a local model truncates
    # to its own block_size, an architectural fact, not a policy cap). min_length=1 is a real
    # "this endpoint needs a query" requirement, not a capacity limit.
    query: str = Field(..., min_length=1)
    route: str = "auto"


class GenBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    raw: bool = False  # True: plain continuation even for a chat-tuned model


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionBody(BaseModel):
    """The subset of OpenAI's chat-completions request this model can honour. Unknown fields are ignored."""
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


MODEL_ID = "ziva-white"


def _oai_error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class AssistantBody(BaseModel):
    text: str = Field(..., min_length=1)
    session: Optional[str] = Field(default=None, max_length=80)   # keeps "are you sure?" confirmations per browser tab
    route: str = "auto"


def _answer_dict(ans) -> dict:
    return {"text": getattr(ans, "text", str(ans)), "route": getattr(ans, "route", "local"),
            "reason": getattr(ans, "reason", ""), "sources": list(getattr(ans, "sources", []) or []),
            "latency_s": getattr(ans, "latency_s", 0.0), "confidence": getattr(ans, "confidence", None)}


def create_app(rt: Runtime, daemon_snapshot: Optional[Callable[[], dict]] = None) -> FastAPI:
    api = FastAPI(title="Ziva White", version=__version__)
    webui_index = os.path.join(os.path.dirname(__file__), "webui", "index.html")
    rate_limited = make_rate_limit_dependency()
    assistant = rt.assistant

    # CORS: by default only pages served from localhost / 127.0.0.1 may call this API from a
    # browser. The bundled web UI is same-origin and needs no CORS at all; the point is that some
    # other website open in your browser can't read answers from a local server.
    # ZIVAWHITE_CORS_ORIGINS="*" (anyone may READ) or a comma-separated list of origins.
    cors_origins = os.getenv("ZIVAWHITE_CORS_ORIGINS", "").strip()
    if cors_origins == "*":
        cors_kwargs = {"allow_origins": ["*"], "allow_credentials": False}
    elif cors_origins:
        cors_kwargs = {"allow_origins": [o.strip() for o in cors_origins.split(",") if o.strip()], "allow_credentials": True}
    else:
        cors_kwargs = {"allow_origin_regex": r"https?://(localhost|127\.0\.0\.1)(:\d+)?", "allow_credentials": False}
    api.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"], **cors_kwargs)

    @api.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = _time.time()
        response = await call_next(request)
        if request.url.path != "/v1/assistant/events":          # the UI polls this; don't drown the log in it
            log.info(f'{request.client.host if request.client else "-"} "{request.method} {request.url.path}" '
                     f"{response.status_code} {(_time.time() - t0) * 1000:.0f}ms")
        return response

    # Structured JSON errors everywhere, including on a genuinely unexpected exception: a client
    # always gets {"error": "..."} instead of a traceback or an HTML page, and the real exception
    # still goes to the server log.
    @api.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=getattr(exc, "headers", None))

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": "invalid request", "detail": exc.errors()})

    @api.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        log.exception(f"unhandled error on {request.url.path}")
        return JSONResponse(status_code=500, content={"error": "internal server error"})

    def can_control(request: Request) -> bool:
        return control_allowed(request.client.host if request.client else None, dict(request.headers), get_server_api_key())

    # ------------------------------------------------------------------------------ pages / info
    @api.get("/", response_class=HTMLResponse)
    def index():
        if os.path.exists(webui_index):
            return FileResponse(webui_index)
        return HTMLResponse("<p>Ziva White is running. Web UI file missing -- use /v1/assistant or /v1/ask directly.</p>")

    @api.get("/v1/health")
    def health():
        from .device import resolve_runtime

        mw = rt.router
        ws = mw.connectors.get("web_search") if mw is not None else None
        return {"status": "ok", "runtime": resolve_runtime().summary(),
                "local_model_loaded": bool(mw and mw.has_local_model()),
                "model": (mw.model_info or None) if mw is not None else None,
                "model_error": rt.router_error,
                "web_search_configured": bool(ws and ws.available())}

    @api.get("/v1/version")
    def version():
        return {"version": __version__, "auth_required": bool(get_server_api_key()),
                "connectors": list(rt.router.connectors) if rt.router is not None else []}

    @api.get("/v1/assistant/info")
    def assistant_info():
        return assistant.info()

    # ------------------------------------------------------------------------------------ asking
    @api.post("/v1/ask", dependencies=[Depends(require_api_key), Depends(rate_limited)])
    def ask(body: AskBody):
        known = assistant.identity.answer(body.query)
        if known is not None:
            return {"text": known, "route": "identity", "reason": "answered from the assistant's configuration",
                    "sources": [], "latency_s": 0.0, "confidence": 1.0}
        return _answer_dict(rt._ask(body.query, body.route))

    @api.post("/v1/generate", dependencies=[Depends(require_api_key), Depends(rate_limited)])
    def generate(body: GenBody):
        mw = rt.router
        if mw is None or not mw.has_local_model():
            return JSONResponse(status_code=503, content={"error": "no local checkpoint loaded"})
        text = mw.generate(body.prompt, max_new_tokens=body.max_new_tokens, temperature=body.temperature, raw=body.raw)
        return {"text": text}  # only the new text: no echoed prompt, no <bos>

    @api.get("/v1/models", dependencies=[Depends(require_api_key)])
    def list_models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": int(_time.time()),
                                            "owned_by": "local"}]}

    @api.post("/v1/chat/completions", dependencies=[Depends(require_api_key), Depends(rate_limited)])
    def chat_completions(body: ChatCompletionBody):
        mw = rt.router
        if mw is None or not mw.has_local_model():
            return _oai_error(503, rt.router_error or "no local checkpoint loaded", "server_error")
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        try:  # validate now: a bad request must be a 400, not a stream that dies halfway
            from .chat import prepare_chat_request

            prepare_chat_request(messages)
        except ValueError as e:
            return _oai_error(400, str(e))
        stops = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
        kw = dict(max_new_tokens=body.max_tokens or 256, temperature=body.temperature, top_p=body.top_p, stop=stops)
        cid, created, model_id = "chatcmpl-" + uuid.uuid4().hex[:24], int(_time.time()), body.model or MODEL_ID

        def chunk(delta: dict, finish: Optional[str] = None) -> dict:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        if not body.stream:
            try:
                out = mw.chat(messages, **kw)
            except ValueError as e:
                return _oai_error(400, str(e))
            except RuntimeError as e:
                return _oai_error(503, str(e), "server_error")
            return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out["text"]},
                                 "finish_reason": out["finish_reason"]}],
                    "usage": {"prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                              "total_tokens": out["prompt_tokens"] + out["completion_tokens"]}}

        def events():
            info: dict = {}
            yield _sse(chunk({"role": "assistant", "content": ""}))
            try:
                for piece in mw.chat_stream(messages, info=info, **kw):
                    yield _sse(chunk({"content": piece}))
            except Exception:  # noqa: BLE001 - a stream that already started cannot change its HTTP status
                log.exception("error while streaming /v1/chat/completions")
                yield _sse({"error": {"message": "generation failed", "type": "server_error"}})
                yield "data: [DONE]\n\n"
                return
            yield _sse(chunk({}, info.get("finish_reason", "stop")))
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/v1/assistant", dependencies=[Depends(require_api_key), Depends(rate_limited)])
    def assistant_endpoint(body: AssistantBody, request: Request):
        t0 = _time.time()
        session = body.session or (request.client.host if request.client else "anon")
        reply = assistant.handle(body.text, session=session, allow_control=can_control(request), route=body.route)
        return {**reply.to_dict(), "latency_s": round(_time.time() - t0, 3)}

    @api.get("/v1/assistant/events", dependencies=[Depends(require_api_key)])
    def assistant_events(since: int = 0):
        return {"events": assistant.events_since(since)}

    @api.post("/v1/voice/stop", dependencies=[Depends(require_api_key)])
    def voice_stop(request: Request):
        if not can_control(request):
            return JSONResponse(status_code=403, content={"error": "local or authenticated callers only"})
        rt.stop_speaking()
        return {"ok": True}

    # --------------------------------------------------------------------------- learning / status
    @api.get("/v1/datasets", dependencies=[Depends(require_api_key)])
    def datasets():
        return {**rt.learner.folder.status(), "learner": rt.learner.snapshot()}

    @api.get("/v1/status", dependencies=[Depends(require_api_key)])
    def status():
        return {"version": __version__, "daemon": daemon_snapshot() if daemon_snapshot else None, "runtime": rt.status()}

    if get_server_api_key():
        log.info("ZIVAWHITE_SERVER_API_KEY is set -- the /v1 endpoints require a bearer token")
    else:
        log.warning("no ZIVAWHITE_SERVER_API_KEY set -- /v1/ask and /v1/generate are open to anyone who can reach this "
                    "server, and device control is limited to this machine. Fine for localhost; set that env var "
                    "before exposing this beyond your own machine.")
    return api
