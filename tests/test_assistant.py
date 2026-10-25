"""The assistant layer: identity answers, command grammar, confirmation flow, access gate, notifications."""
from types import SimpleNamespace

from helpers import make_cfg
from test_device_control import FakeSystem

from zivawhite.assistant import Assistant
from zivawhite.control import DeviceController
from zivawhite.control.backends import get_backend
from zivawhite.control.timers import TimerScheduler
from zivawhite.identity import Identity


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def make(tmp_path, have=("systemctl", "loginctl", "pactl"), ask=None, **cfg_over):
    cfg = make_cfg(tmp_path, **cfg_over)
    system = FakeSystem("linux", have)
    asked = []
    def default_ask(text, route):
        asked.append((text, route))
        return SimpleNamespace(text=f"model says: {text}", route="local", sources=["s1"], reason="because")
    ctl = DeviceController(cfg, backend=get_backend(system), timers=TimerScheduler(None))
    clock = Clock()
    a = Assistant(cfg, ask=ask or default_ask, controller=ctl, clock=clock)
    a.speaker = None
    a._speaker_failed = True                     # tests never open a real voice
    return a, system, asked, clock


# ------------------------------------------------------------------------------- identity
def test_creator_is_answered_from_the_config_before_the_model_is_consulted(tmp_path):
    a, _, asked, _ = make(tmp_path)
    r = a.handle("Who created you?")
    assert r.kind == "identity" and r.text == "I was created by Mahasvin S S, also known as Snow White."
    assert asked == []


def test_other_identity_questions(tmp_path):
    a, _, _, _ = make(tmp_path)
    assert "Mahasvin S S (Snow White)" in a.handle("who are you").text
    assert a.handle("Who is Mahasvin S S?").text == "Mahasvin S S, also known as Snow White, is my creator."


def test_the_creator_is_configuration_not_code(tmp_path):
    a, _, _, _ = make(tmp_path, identity={"assistant_name": "Nova", "creator": {"name": "Ada Lovelace", "alias": "Countess"}})
    assert a.handle("who made you").text == "I was created by Ada Lovelace, also known as Countess."
    assert "Nova" in a.handle("what's your name").text


def test_questions_that_merely_contain_creator_words_go_to_the_model(tmp_path):
    a, _, asked, _ = make(tmp_path)
    for q in ("who made the Eiffel Tower", "who created python", "what are you doing tomorrow"):
        assert a.handle(q).kind == "model"
    assert len(asked) == 3


# ------------------------------------------------------------------------------- commands
def test_commands_never_reach_the_model(tmp_path):
    a, system, asked, _ = make(tmp_path)
    r = a.handle("set the volume to 40")
    assert r.kind == "control" and r.ok and r.action == "volume_set" and asked == []
    assert system.ran[-1][-1] == "40%"


def test_ordinary_questions_go_to_the_model_with_metadata(tmp_path):
    a, _, asked, _ = make(tmp_path)
    r = a.handle("explain photosynthesis", route="local")
    assert (r.kind, r.text, r.route, r.sources, r.reason) == ("model", "model says: explain photosynthesis", "local", ["s1"], "because")
    assert asked == [("explain photosynthesis", "local")]


def test_model_failure_is_a_reply_not_an_exception(tmp_path):
    def boom(text, route): raise RuntimeError("oom")
    a, _, _, _ = make(tmp_path, ask=boom)
    r = a.handle("hello there friend")
    assert r.kind == "error" and not r.ok and "RuntimeError" in r.text


def test_no_model_loaded(tmp_path):
    cfg = make_cfg(tmp_path)
    a = Assistant(cfg, ask=None)
    assert a.handle("hello").kind == "error"
    assert a.handle("volume up").kind == "error"               # no controller: the grammar is not consulted, it goes to the (absent) model


def test_empty_input_is_ignored(tmp_path):
    a, _, asked, _ = make(tmp_path)
    assert a.handle("   ").kind == "ignored" and asked == []


# ----------------------------------------------------------------------------- confirmation
def test_risky_actions_ask_first_and_run_only_on_yes(tmp_path):
    a, system, _, _ = make(tmp_path)
    r = a.handle("lock the screen", session="s1")
    assert r.kind == "confirm" and "Say yes" in r.text and system.ran == []
    r = a.handle("yes", session="s1")
    assert r.kind == "control" and r.ok and system.ran[-1] == ["loginctl", "lock-session"]
    assert a.handle("yes", session="s1").kind == "model"                 # nothing pending any more


def test_no_cancels(tmp_path):
    a, system, _, _ = make(tmp_path)
    a.handle("lock the screen", session="s1")
    assert a.handle("no", session="s1").text == "Okay, cancelled." and system.ran == []


def test_any_other_message_cancels_the_pending_action(tmp_path):
    a, system, _, _ = make(tmp_path)
    a.handle("lock the screen", session="s1")
    a.handle("what time is it", session="s1")
    assert a.handle("yes", session="s1").kind == "model" and system.ran == []


def test_confirmation_expires(tmp_path):
    a, system, _, clock = make(tmp_path)
    a.handle("lock the screen", session="s1")
    clock.t += 21                                                        # confirm_ttl_s is 20
    assert a.handle("yes", session="s1").kind == "model" and system.ran == []


def test_confirmations_are_per_session(tmp_path):
    a, system, _, _ = make(tmp_path)
    a.handle("lock the screen", session="alice")
    assert a.handle("yes", session="bob").kind == "model" and system.ran == []     # bob cannot confirm alice's request
    assert a.handle("yes", session="alice").ok and system.ran


def test_power_actions_are_refused_up_front_when_disabled(tmp_path):
    a, system, _, _ = make(tmp_path)
    r = a.handle("shut down the computer")
    assert r.kind == "control" and not r.ok and "allow_power_actions" in r.text and system.ran == []


def test_power_actions_still_need_a_yes_when_enabled(tmp_path):
    a, system, _, _ = make(tmp_path, device_control={"allow_power_actions": True})
    assert a.handle("shut down", session="s").kind == "confirm" and system.ran == []
    assert a.handle("yes", session="s").ok and system.ran[-1] == ["systemctl", "poweroff"]


# ------------------------------------------------------------------------------ access gate
def test_control_is_refused_not_forwarded_when_the_connection_is_not_trusted(tmp_path):
    a, system, asked, _ = make(tmp_path)
    r = a.handle("volume up", allow_control=False)
    assert r.kind == "control" and not r.ok and "local or authenticated" in r.text
    assert system.ran == [] and asked == []
    assert a.handle("who created you", allow_control=False).kind == "identity"      # harmless things still work
    assert a.handle("explain gravity", allow_control=False).kind == "model"


def test_a_model_cannot_trigger_commands(tmp_path):
    # even if the model's OWN output looked like a command, only the person's text is ever parsed
    a, system, _, _ = make(tmp_path, ask=lambda t, r: SimpleNamespace(text="lock the screen", route="local"))
    r = a.handle("tell me a secret phrase")
    assert r.kind == "model" and r.text == "lock the screen" and system.ran == []


# --------------------------------------------------------------------- notifications / events
def test_timer_notifications_reach_the_event_feed_and_the_speaker(tmp_path):
    a, _, _, _ = make(tmp_path)
    said = []
    a.speaker = SimpleNamespace(speak=lambda t: said.append(t))
    ts = a.controller.timers
    ts.on_fire = a.on_timer
    ts.add(1, "stretch")
    ts._clock = lambda: 10 ** 12
    ts.fire_due()
    ev = a.events_since(0)
    assert ev[-1]["kind"] == "timer" and "stretch" in ev[-1]["text"] and said and "stretch" in said[0]
    assert a.events_since(ev[-1]["id"]) == []


def test_speech_of_notifications_can_be_disabled(tmp_path):
    a, _, _, _ = make(tmp_path, voice={"speak_replies": False})
    said = []
    a.speaker = SimpleNamespace(speak=lambda t: said.append(t))
    a.notify("hello")
    assert said == [] and a.events_since(0)[0]["text"] == "hello"


def test_info_for_the_ui(tmp_path):
    a, _, _, _ = make(tmp_path)
    info = a.info()
    assert info["assistant_name"] == "Ziva White" and info["creator"] == {"name": "Mahasvin S S", "alias": "Snow White"}
    assert "hey ziva" in info["wake_words"] and info["device_control"] is True


# ------------------------------------------------------------------------------- confirmation is not an access check
def test_a_caller_without_control_rights_cannot_confirm_someone_elses_pending_action(tmp_path):
    """The review's bypass: a voice user is asked 'lock the screen? yes/no'; a remote caller sends 'yes' under the same
    (predictable) session name, with allow_control=False. The action must not run."""
    a, system, _, _ = make(tmp_path)
    assert a.handle("lock the screen", session="voice").kind == "confirm"
    r = a.handle("yes", session="voice", allow_control=False)
    assert r.ok is False and r.kind == "control" and "local or authenticated" in r.text
    assert system.ran == []                                      # nothing was executed
    # the legitimate caller was not locked out by the attempt: the intent was consumed, so they simply ask again
    assert a.handle("lock the screen", session="voice").kind == "confirm"
    assert a.handle("yes", session="voice").ok and system.ran


def test_a_busy_model_is_reported_not_crashed(tmp_path):
    from zivawhite.errors import ModelBusy

    def busy(text, route):
        raise ModelBusy("busy")
    a, _, _, _ = make(tmp_path, ask=busy)
    r = a.handle("what is the capital of France")
    assert r.kind == "error" and "another request" in r.text
