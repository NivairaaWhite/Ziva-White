"""Configuration layering/validation and the creator identity."""
import pytest
from helpers import make_cfg

from zivawhite.config import DEFAULT_CONFIG_PATH, Config, ConfigError, deep_merge, load_config
from zivawhite.identity import Identity, normalize


# --------------------------------------------------------------------------------- config
def test_the_shipped_config_is_valid():
    cfg = load_config(path=DEFAULT_CONFIG_PATH, env={})
    assert cfg.problems() == []
    assert cfg.get("identity.creator.name") == "Mahasvin S S" and cfg.get("identity.creator.alias") == "Snow White"


def test_deep_merge_merges_dicts_and_replaces_lists():
    base = {"a": {"x": 1, "y": 2}, "words": ["a", "b"]}
    out = deep_merge(base, {"a": {"y": 3}, "words": ["c"]})
    assert out == {"a": {"x": 1, "y": 3}, "words": ["c"]} and base["a"]["y"] == 2          # input untouched


def test_local_override_file_is_merged_over_defaults(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "ziva.local.yaml").write_text("voice:\n  wake:\n    words: [computer]\npaths:\n  datasets: my_data\n")
    cfg = load_config(path=DEFAULT_CONFIG_PATH, root=tmp_path, env={})
    assert cfg.get("voice.wake.words") == ["computer"] and cfg.get("voice.wake.sensitivity") == 0.88      # only what you set changed
    assert cfg.datasets_dir == tmp_path / "my_data"


def test_env_selects_the_override_file_and_the_datasets_folder(tmp_path):
    other = tmp_path / "other.yaml"
    other.write_text("background:\n  port: 9100\n")
    cfg = load_config(path=DEFAULT_CONFIG_PATH, root=tmp_path, env={"ZIVAWHITE_CONFIG": str(other), "ZIVAWHITE_DATASETS": "/data/train"})
    assert cfg.get("background.port") == 9100 and str(cfg.datasets_dir).replace("\\", "/") == "/data/train"
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(path=DEFAULT_CONFIG_PATH, root=tmp_path, env={"ZIVAWHITE_CONFIG": str(tmp_path / "nope.yaml")})


def test_broken_yaml_gives_a_readable_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path=bad, root=tmp_path, env={})
    lst = tmp_path / "list.yaml"
    lst.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path=lst, root=tmp_path, env={})
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=tmp_path / "missing.yaml", root=tmp_path, env={})


def test_paths_resolve_relative_to_the_project_not_the_cwd(tmp_path):
    cfg = make_cfg(tmp_path)
    assert cfg.runs_dir == tmp_path / "runs" and cfg.datasets_dir == tmp_path / "datasets"
    assert cfg.resolve("/abs/x").as_posix() == "/abs/x"
    assert str(cfg.resolve("~/pics")).startswith(str(__import__("pathlib").Path.home()))


@pytest.mark.parametrize("override,fragment", [
    ({"voice": {"wake": {"sensitivity": 1.5}}}, "sensitivity"),
    ({"voice": {"wake": {"sensitivity": 0}}}, "sensitivity"),
    ({"voice": {"wake": {"words": []}}}, "wake.words"),
    ({"voice": {"wake": {"words": ["", "  "]}}}, "non-empty"),
    ({"background": {"port": 70000}}, "port"),
    ({"background": {"port": "eight"}}, "port"),
    ({"background": {"jobs": ["autolearn", "teleport"]}}, "teleport"),
    ({"voice": {"stt": {"engine": "magic"}}}, "stt.engine"),
    ({"voice": {"tts": {"engine": "magic"}}}, "tts.engine"),
    ({"autolearn": {"scan_interval_s": "often"}}, "scan_interval_s"),
    ({"identity": {"creator": {"name": 5}}}, "creator.name"),
])
def test_validation_catches_typos_early(tmp_path, override, fragment):
    problems = make_cfg(tmp_path, **override).problems()
    assert any(fragment in p for p in problems), problems


def test_require_and_section_helpers():
    cfg = Config({"a": {"b": 1}})
    assert cfg.get("a.b") == 1 and cfg.get("a.c", 7) == 7 and cfg.section("a") == {"b": 1} and cfg.section("zzz") == {}
    with pytest.raises(ConfigError):
        cfg.require("a.c")


def test_booleans_are_not_accepted_as_numbers(tmp_path):
    assert any("scan_interval_s" in p for p in make_cfg(tmp_path, autolearn={"scan_interval_s": True}).problems())


# -------------------------------------------------------------------------------- identity
def ident(**kw):
    answers = {"creator": "I was created by {creator_name}, also known as {creator_alias}.",
               "who_are_you": "I'm {assistant_name}, built by {creator_name} ({creator_alias}).",
               "about_creator": "{creator_name} ({creator_alias}) is my creator."}
    return Identity(kw.get("assistant", "Ziva White"), kw.get("name", "Mahasvin S S"), kw.get("alias", "Snow White"), answers,
                    kw.get("questions", {"creator": ["Who created you?"], "who_are_you": ["Who are you?"],
                                         "about_creator": ["Who is {creator_name}?"]}))


@pytest.mark.parametrize("q", [
    "Who created you?", "who made you", "Who built you?", "who's your creator", "who is your developer", "Hey, who made you?",
    "tell me who built you", "can you tell me who made you", "Do you know who created you?", "who programmed you", "who do you belong to",
])
def test_creator_phrasings(q):
    assert ident().match(q).kind == "creator"


@pytest.mark.parametrize("q", ["Who are you?", "what are you", "what is your name", "what's your name", "introduce yourself", "tell me about yourself"])
def test_who_are_you_phrasings(q):
    assert ident().match(q).kind == "who_are_you"


@pytest.mark.parametrize("q", ["Who is Mahasvin S S?", "who is mahasvin", "tell me about Mahasvin", "do you know Mahasvin S S", "who is Snow White"])
def test_questions_about_the_creator(q):
    assert ident().match(q).kind == "about_creator"


@pytest.mark.parametrize("q", [
    "who made the Eiffel Tower", "who created python", "what are you doing", "who is the president", "who invented the telephone",
    "who is snow white in the fairy tale", "tell me the story of snow white and the seven dwarfs", "what is your favourite colour",
    "who wrote hamlet", "", "x " * 200,
])
def test_things_that_only_look_similar_are_not_identity_questions(q):
    assert ident().match(q) is None


def test_the_name_and_alias_are_filled_from_config():
    assert ident().answer("who created you") == "I was created by Mahasvin S S, also known as Snow White."
    other = ident(name="Ada Lovelace", alias="Countess", assistant="Nova")
    assert other.answer("who made you") == "I was created by Ada Lovelace, also known as Countess."
    assert "Nova" in other.answer("who are you") and other.answer("who is Ada") is not None


def test_no_alias_falls_back_to_the_name():
    assert ident(alias="").answer("who made you") == "I was created by Mahasvin S S, also known as Mahasvin S S."


def test_no_creator_configured_disables_identity_answers():
    assert ident(name="").match("who created you") is None


def test_training_records_are_built_from_the_config():
    recs = ident().training_records()
    assert len(recs) == 3
    assert recs[0]["messages"][0] == {"role": "user", "content": "Who created you?"}
    assert recs[0]["messages"][1]["content"] == "I was created by Mahasvin S S, also known as Snow White."
    assert recs[2]["messages"][0]["content"] == "Who is Mahasvin S S?"                 # placeholders in questions are filled too


def test_training_records_skip_kinds_without_an_answer():
    i = Identity("A", "B", "C", {"creator": "x"}, {"creator": ["q"], "who_are_you": ["q2"]})
    assert len(i.training_records()) == 1


def test_shipped_config_yields_enough_training_examples():
    recs = Identity.from_config(load_config(path=DEFAULT_CONFIG_PATH, env={})).training_records()
    assert len(recs) >= 20
    assert all("Mahasvin S S" in r["messages"][1]["content"] or "Ziva White" in r["messages"][1]["content"] for r in recs)


def test_normalize():
    assert normalize("Who's  YOUR creator??") == "who is your creator"
    assert normalize("Hey, please tell me who made you") == "who made you"
