"""
runtime.py - builds the whole assistant from configs/ziva.yaml and hands each background job
its entry point.

    cfg --> Runtime --> router (the model; needs torch)      \
                    --> assistant (identity -> control -> model)  > used by the CLI, the web server,
                    --> timers, controller                        > the voice loop and the daemon
                    --> learner (watches datasets/)
                    --> voice loop (microphone)

The model is loaded last and degrades gracefully: if PyTorch is missing or there is no checkpoint
yet, everything else still works (voice, device control, timers, identity answers, the datasets
watcher that will train the first checkpoint), and questions get an honest explanation instead of
a crash.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Callable, Dict, Optional

from .assistant import Assistant
from .autolearn import DatasetLearner
from .checkpoint import find_default_checkpoint
from .config import Config
from .control import DeviceController
from .control.timers import TimerScheduler


class Runtime:
    def __init__(self, cfg: Config, checkpoint: Optional[str] = None, log: Callable[[str], None] = print,
                 load_model: bool = True):
        self.cfg = cfg
        self.log = log
        self.checkpoint = checkpoint
        self.router = None
        self.router_error: Optional[str] = None
        self._router_lock = threading.Lock()

        self.timers = TimerScheduler(cfg.runs_dir / "timers.json")
        self.controller = DeviceController(cfg, timers=self.timers)
        self.assistant = Assistant(cfg, ask=self._ask, controller=self.controller)
        self.timers.on_fire = self.assistant.on_timer
        self.learner = DatasetLearner(cfg, log=log, on_trained=self.reload_model)
        self.voice = None
        if load_model:
            self.load_router()

    # -- the model -------------------------------------------------------------------------------
    def run_dirs(self):
        al = self.cfg.section("autolearn")
        return [str(self.cfg.resolve(al.get("chat_run", "runs/ziva_behavior"))),
                str(self.cfg.resolve(al.get("base_run", "runs/ziva")))]

    def find_checkpoint(self) -> Optional[str]:
        return self.checkpoint or find_default_checkpoint(str(self.cfg.root), run_dirs=self.run_dirs())

    def load_router(self):
        """Import torch and load whichever checkpoint exists. Never raises: problems are kept in
        `router_error` and surfaced as answers."""
        with self._router_lock:
            try:
                from .device import resolve_runtime
                from .router import ZivaWhite

                ckpt = self.find_checkpoint()
                self.router = ZivaWhite(checkpoint_path=ckpt, device=resolve_runtime().device,
                                        queue_dir=str(self.cfg.data_dir / "queue"))
                self.router_error = None
                gen = self.cfg.section("generation")
                for key in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty"):
                    if gen.get(key) is not None:
                        self.router.sampling[key] = gen[key]
                if self.router.has_local_model():
                    i = self.router.model_info
                    self.log(f"[ziva] local model: {i['file']} ({i['n_layer']} layers x {i['n_embd']} dims, step {i['step']}, "
                             f"{'chat-tuned' if self.router.behavior_tuned else 'base, not chat-tuned'})")
                else:
                    self.log("[ziva] no local model yet -- put training files in "
                             f"{self.cfg.datasets_dir} and autolearn will train one")
            except ImportError as e:
                self.router = None
                self.router_error = (f"The language model can't load because a package is missing ({e}). "
                                     "Run: pip install -r requirements.txt")
                self.log(f"[ziva] {self.router_error}")
            except Exception as e:  # noqa: BLE001 - a corrupt checkpoint must not stop voice/control/timers
                self.router = None
                self.router_error = f"The language model failed to load: {type(e).__name__}: {e}"
                self.log(f"[ziva] {self.router_error}")
        return self.router

    def reload_model(self) -> None:
        """Pick up a freshly trained checkpoint without restarting anything."""
        path = self.find_checkpoint()
        if not path:
            return
        if self.router is None:
            self.load_router()
            return
        with self.router._gen_lock:                  # no generation while the weights are swapped
            self.router.load_local_model(path)
        i = self.router.model_info
        self.log(f"[ziva] loaded the new model: step {i['step']}, {'chat-tuned' if self.router.behavior_tuned else 'base'}")

    def _ask(self, text: str, route: str):
        if self.router is None:
            return SimpleNamespace(text=self.router_error or "The language model isn't available.", route="none",
                                   sources=[], reason="no router")
        return self.router.ask(text, route=route)

    # -- background jobs -------------------------------------------------------------------------------
    def jobs(self) -> Dict[str, Callable[[threading.Event], None]]:
        """name -> function(stop_event). Only the jobs named in background.jobs are returned."""
        wanted = list(self.cfg.get("background.jobs", []) or [])
        out: Dict[str, Callable[[threading.Event], None]] = {}
        if "autolearn" in wanted and self.cfg.get("autolearn.enabled", True):
            out["autolearn"] = self.learner.run_forever
        if "timers" in wanted:
            out["timers"] = lambda stop: self.timers.run(stop)
        if "voice" in wanted and self.cfg.get("voice.enabled", False):
            out["voice"] = self._run_voice
        return out

    def _run_voice(self, stop: threading.Event) -> None:
        from .voice.loop import VoiceAssistant

        if self.voice is None:
            speak = bool(self.cfg.get("voice.speak_replies", True))
            self.voice = VoiceAssistant(self.cfg, lambda text: self.assistant.handle(text, session="voice"),
                                        tts=self.assistant.get_speaker() if speak else None, log=self.log)
        self.voice.run(stop)

    def stop_speaking(self) -> None:
        if self.voice is not None:
            self.voice.stop_speaking()
        sp = self.assistant.speaker
        if sp is not None:
            sp.stop()

    def status(self) -> dict:
        return {
            "model": (self.router.model_info if self.router is not None and self.router.has_local_model() else None),
            "model_error": self.router_error,
            "autolearn": self.learner.snapshot(),
            "voice": self.voice.snapshot() if self.voice is not None else None,
            "timers": len(self.timers.list()),
        }
