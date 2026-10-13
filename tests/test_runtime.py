"""Runtime wiring: which jobs run, graceful degradation without a model, hot-reload bookkeeping."""
from types import SimpleNamespace

from helpers import make_cfg

from zivawhite.runtime import Runtime


def rt(tmp_path, **over):
    return Runtime(make_cfg(tmp_path, **over), load_model=False, log=lambda *_: None)


def test_default_jobs_are_autolearn_and_timers_and_voice_is_opt_in(tmp_path):
    assert sorted(rt(tmp_path).jobs()) == ["autolearn", "timers"]
    assert sorted(rt(tmp_path, voice={"enabled": True}).jobs()) == ["autolearn", "timers", "voice"]


def test_jobs_follow_the_config(tmp_path):
    assert sorted(rt(tmp_path, background={"jobs": ["timers"]}).jobs()) == ["timers"]
    assert rt(tmp_path, background={"jobs": []}).jobs() == {}
    assert "autolearn" not in rt(tmp_path, autolearn={"enabled": False}).jobs()


def test_without_a_model_questions_get_an_explanation_but_everything_else_works(tmp_path):
    r = rt(tmp_path)
    r.router_error = "The language model can't load because a package is missing (No module named 'torch')."
    reply = r.assistant.handle("explain gravity")
    assert reply.kind == "model" and "package is missing" in reply.text
    assert "Mahasvin S S" in r.assistant.handle("who created you").text
    assert r.assistant.handle("set a timer for 5 minutes").ok


def test_status_shape(tmp_path):
    st = rt(tmp_path).status()
    assert st["model"] is None and st["autolearn"]["state"] == "starting" and st["voice"] is None and st["timers"] == 0


def test_run_dirs_come_from_config_with_chat_model_first(tmp_path):
    dirs = rt(tmp_path, autolearn={"chat_run": "r/chat", "base_run": "r/base"}).run_dirs()
    assert [d.replace("\\", "/").split("/")[-2:] for d in dirs] == [["r", "chat"], ["r", "base"]]


def test_find_checkpoint_prefers_the_chat_tuned_run(tmp_path):
    r = rt(tmp_path)
    for run in ("ziva", "ziva_behavior"):
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / "checkpoint.pt").write_text("x")
    assert r.find_checkpoint().replace("\\", "/").endswith("runs/ziva_behavior/checkpoint.pt")
    assert Runtime(make_cfg(tmp_path), checkpoint="/explicit.pt", load_model=False).find_checkpoint() == "/explicit.pt"


def test_reload_model_swaps_weights_under_the_generation_lock(tmp_path):
    r = rt(tmp_path)
    (tmp_path / "runs" / "ziva").mkdir(parents=True)
    (tmp_path / "runs" / "ziva" / "checkpoint.pt").write_text("x")
    events = []

    class Lock:
        def __enter__(self): events.append("lock")
        def __exit__(self, *a): events.append("unlock")

    class Router:
        _gen_lock = Lock()
        model_info = {"step": 7}
        behavior_tuned = True
        def load_local_model(self, path): events.append(("load", path.replace("\\", "/").split("/")[-1]))

    r.router = Router()
    r.reload_model()
    assert events == ["lock", ("load", "checkpoint.pt"), "unlock"]


def test_reload_with_nothing_trained_yet_is_a_no_op(tmp_path):
    r = rt(tmp_path)
    r.reload_model()
    assert r.router is None


def test_stop_speaking_reaches_both_voice_loop_and_shared_speaker(tmp_path):
    r = rt(tmp_path)
    stops = []
    r.assistant.speaker = SimpleNamespace(stop=lambda: stops.append("speaker"))
    r.voice = SimpleNamespace(stop_speaking=lambda: stops.append("voice"))
    r.stop_speaking()
    assert stops == ["voice", "speaker"]


def test_learner_is_wired_to_hot_reload(tmp_path):
    r = rt(tmp_path)
    assert r.learner.on_trained == r.reload_model
    assert r.timers.on_fire == r.assistant.on_timer
