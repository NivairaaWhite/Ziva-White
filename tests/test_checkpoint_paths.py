"""Which checkpoint file gets loaded. Pure path logic, no torch work."""
import os

from zivawhite.checkpoint import find_default_checkpoint, resolve_serving_checkpoint


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def test_rolling_checkpoint_resolves_to_best_when_present(tmp_path):
    ck, best = str(tmp_path / "checkpoint.pt"), str(tmp_path / "checkpoint_best.pt")
    _touch(ck)
    assert resolve_serving_checkpoint(ck) == ck          # no best yet
    _touch(best)
    assert resolve_serving_checkpoint(ck) == best


def test_an_explicitly_named_file_is_loaded_as_asked(tmp_path):
    _touch(str(tmp_path / "checkpoint_best.pt"))
    named = str(tmp_path / "my_model.pt")
    _touch(named)
    assert resolve_serving_checkpoint(named) == named    # used to be silently swapped for checkpoint_best.pt
    int8 = str(tmp_path / "checkpoint_int8.pt")
    assert resolve_serving_checkpoint(int8) == int8


def test_default_discovery_prefers_the_chat_tuned_run(tmp_path):
    root = str(tmp_path)
    assert find_default_checkpoint(root) is None
    _touch(os.path.join(root, "runs", "ziva", "checkpoint.pt"))
    assert find_default_checkpoint(root) == os.path.join(root, "runs", "ziva", "checkpoint.pt")
    _touch(os.path.join(root, "runs", "ziva_behavior", "checkpoint.pt"))
    assert find_default_checkpoint(root) == os.path.join(root, "runs", "ziva_behavior", "checkpoint.pt")


def test_default_discovery_accepts_a_folder_with_only_a_best_checkpoint(tmp_path):
    root = str(tmp_path)
    _touch(os.path.join(root, "runs", "ziva", "checkpoint_best.pt"))
    found = find_default_checkpoint(root)
    assert found is not None and resolve_serving_checkpoint(found).endswith("checkpoint_best.pt")
