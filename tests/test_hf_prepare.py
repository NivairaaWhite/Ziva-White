"""hf_prepare: converters, the wrong-schema guard, streaming shards, provenance. Offline: rows come from local files."""
import json
import os

import numpy as np
import pytest

from zivawhite.data import hf_prepare as hp
from zivawhite.data.dataset import token_count
from zivawhite.sft import build_examples, load_conversations
from zivawhite.tokenizer import ByteLevelTokenizer
from zivawhite.tools import parse_tool_call

TOK = ByteLevelTokenizer()


# ---------------------------------------------------------------------------------- recipes
def test_recipes_are_well_formed_and_the_license_status_is_honest_about_what_was_checked():
    assert len(hp.RECIPES) == len({r.key for r in hp.RECIPES.values()})
    for key, r in hp.RECIPES.items():
        assert r.key == key and r.kind in ("text", "chat", "prefs", "glaive", "xlam") and r.hf_id.count("/") == 1
        if r.kind in ("text", "chat"):
            assert r.column
    assert hp.RECIPES["smol-smoltalk"].license == "Apache-2.0" and hp.RECIPES["smol-smoltalk"].license_checked
    assert hp.RECIPES["ultrafeedback"].license == "MIT" and hp.RECIPES["ultrafeedback"].license_checked
    assert "NOT STATED" in hp.RECIPES["open-phi-textbooks"].license                    # read off the live card
    assert not hp.RECIPES["openhermes-2.5"].license_checked and hp.RECIPES["xlam-function-calling"].gated
    hp.main(["list"])                                                                    # prints without error


# -------------------------------------------------------------------------------- converters
def test_chat_rows_keep_roles_and_openhermes_style_from_value_works():
    smol = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "source": "x"}
    assert hp.convert_chat(smol, "messages")["messages"][0] == {"role": "system", "content": "s"}
    hermes = {"conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
    assert [m["role"] for m in hp.convert_chat(hermes, "conversations")["messages"]] == ["system", "user", "assistant"]
    assert hp.convert_chat({"messages": [{"role": "user", "content": "only"}]}, "messages") is None
    assert hp.convert_chat({"nope": 1}, "messages") is None


def test_ultrafeedback_style_preference_rows():
    row = {"prompt": "Why is the sky blue?", "chosen": [{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": "Rayleigh."}],
           "rejected": [{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": "Paint."}], "chosen-rating": 5}
    out = hp.convert_prefs(row)
    assert out == {"prompt": [{"role": "user", "content": "Why is the sky blue?"}], "chosen": "Rayleigh.", "rejected": "Paint."}
    assert hp.convert_prefs({"prompt": "q", "chosen": "a", "rejected": "a"}) is None


GLAIVE_SYSTEM = 'SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{"name": "get_rate", "parameters": {"type": "object", "properties": {"base": {"type": "string"}}}}'
GLAIVE_CHAT = ("USER: What is the USD to EUR rate?\n\n\nASSISTANT: <functioncall> {\"name\": \"get_rate\", \"arguments\": '{\"base\": \"USD\", \"target\": \"EUR\"}'} <|endoftext|>\n\n\n"
               "FUNCTION RESPONSE: {\"rate\": 0.92}\n\n\nASSISTANT: The rate is 0.92. <|endoftext|>\n\n\n")


def test_glaive_rows_become_the_tool_protocol_with_the_function_result_as_a_tool_turn():
    out = hp.convert_glaive({"system": GLAIVE_SYSTEM, "chat": GLAIVE_CHAT})
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert out["messages"][0]["content"].startswith("You can use tools.") and "get_rate" in out["messages"][0]["content"]
    assert "SYSTEM:" not in out["messages"][0]["content"]
    assert parse_tool_call(out["messages"][2]["content"]) == {"name": "get_rate", "arguments": {"base": "USD", "target": "EUR"}}
    assert out["messages"][3]["content"] == '{"rate": 0.92}' and out["messages"][4]["content"] == "The rate is 0.92."
    assert len(build_examples([out["messages"]], TOK, 1024)) == 2                      # usable SFT data


def test_glaive_accepts_unquoted_json_arguments_and_plain_chat_and_drops_unparseable_calls():
    dict_form = 'USER: hi\n\n\nASSISTANT: <functioncall> {"name": "f", "arguments": {"a": 1}} <|endoftext|>\n\n\nFUNCTION RESPONSE: ok\n\n\nASSISTANT: done'
    assert parse_tool_call(hp.convert_glaive({"system": "SYSTEM: s", "chat": dict_form})["messages"][2]["content"]) == {"name": "f", "arguments": {"a": 1}}
    plain = hp.convert_glaive({"system": "SYSTEM: s", "chat": "USER: hello\n\n\nASSISTANT: Hi there! <|endoftext|>"})
    assert [m["role"] for m in plain["messages"]] == ["system", "user", "assistant"]
    broken = 'USER: hi\n\n\nASSISTANT: <functioncall> {"name": "f", "arguments": \'{oops\'} <|endoftext|>'
    assert hp.convert_glaive({"system": "SYSTEM: s", "chat": broken}) is None
    assert hp.convert_glaive({"system": 1, "chat": 2}) is None


def test_xlam_single_call_rows_only():
    tools = [{"name": "weather", "description": "Get weather", "parameters": {"city": {"type": "str", "description": "the city"}}}]
    row = {"query": "Weather in Paris?", "tools": json.dumps(tools), "answers": json.dumps([{"name": "weather", "arguments": {"city": "Paris"}}])}
    out = hp.convert_xlam(row)
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert "- weather(city: the city): Get weather" in out["messages"][0]["content"]
    assert parse_tool_call(out["messages"][2]["content"]) == {"name": "weather", "arguments": {"city": "Paris"}}
    two = {**row, "answers": json.dumps([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}])}
    assert hp.convert_xlam(two) is None and hp.convert_xlam({**row, "tools": "not json"}) is None


# ---------------------------------------------------------------------------------- writers
def test_the_wrong_schema_guard_refuses_to_leave_a_nearly_empty_dataset_behind(tmp_path):
    rows = [{"wrong_column": i} for i in range(40)] + [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    out = str(tmp_path / "o.jsonl")
    with pytest.raises(SystemExit, match="inspect"):
        hp.write_jsonl(out, rows, "chat", "messages")
    assert not os.path.exists(out) and not os.path.exists(out + ".tmp")
    with pytest.raises(SystemExit, match="0 usable"):
        hp.write_jsonl(out, [{"x": 1}], "chat", "messages")


def test_jsonl_output_provenance_and_row_limit(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(30)]
    out = str(tmp_path / "d" / "o.jsonl")
    res = hp.write_jsonl(out, rows, "chat", "messages", max_rows=10, recipe=hp.RECIPES["smol-smoltalk"])
    assert res["kept"] == 10 and sum(1 for _ in open(out)) == 10
    prov = json.load(open(res["provenance"]))
    assert prov["source"]["license"] == "Apache-2.0" and prov["rows_kept"] == 10 and "terms of any model" in prov["reminder"]
    assert len(load_conversations([out])) == 10                                          # feeds sft.py directly


def test_text_shards_match_the_trainers_format_and_split_whole_documents(tmp_path):
    docs = [f"document number {i} " + "word " * (i % 7 + 3) for i in range(400)]
    out = str(tmp_path / "shards")
    res = hp.write_text_shards(out, iter(docs), TOK, val_permille=50, batch_docs=32, recipe=hp.RECIPES["open-phi-textbooks"])
    train = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
    assert (len(train), len(val)) == (res["train_tokens"], res["val_tokens"]) == (token_count(out, "train"), token_count(out, "val"))
    expected_val = [i for i in range(400) if hp.is_val_doc(i, 50)]
    assert 5 <= len(expected_val) <= 40                                                   # ~5% of documents
    want_train = [t for i, d in enumerate(docs) if i not in set(expected_val) for t in TOK.encode(d, add_eos=True)]
    want_val = [t for i in expected_val for t in TOK.encode(docs[i], add_eos=True)]
    assert train.tolist() == want_train and val.tolist() == want_val                      # whole docs, in order, <eos> after each
    assert (train == TOK.eos_id).sum() == 400 - len(expected_val)
    prov = json.load(open(os.path.join(out, "shards.provenance.json")))          # travels with the shards
    assert prov["train_tokens"] == len(train) and prov["source"]["key"] == "open-phi-textbooks"
    assert "NOT STATED" in prov["source"]["license"]


def test_text_shards_can_stop_at_a_token_budget_without_reading_everything(tmp_path):
    consumed = []

    def docs():
        for i in range(100_000):
            consumed.append(i)
            yield "x" * 200
    res = hp.write_text_shards(str(tmp_path / "s"), docs(), TOK, max_tokens=20_000, batch_docs=16)
    assert 20_000 <= res["train_tokens"] + res["val_tokens"] < 40_000 and len(consumed) < 500


def test_validation_membership_is_deterministic_and_roughly_the_requested_fraction():
    flags = [hp.is_val_doc(i, 10) for i in range(20_000)]
    assert flags == [hp.is_val_doc(i, 10) for i in range(20_000)]
    assert 0.007 < sum(flags) / len(flags) < 0.013
    assert hp.is_val_doc(5, 10, seed=1) == hp.is_val_doc(5, 10, seed=1)


# ------------------------------------------------------------------------- the CLI, offline
def test_cli_converts_a_local_file_end_to_end_and_inspect_works(tmp_path):
    src = tmp_path / "raw.jsonl"
    src.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}) for i in range(25)))
    out = str(tmp_path / "chat.jsonl")
    hp.main(["chat", "--recipe", "smol-smoltalk", "--local", str(src), "--out", out, "--max_rows", "20"])
    assert sum(1 for _ in open(out)) == 20
    hp.main(["inspect", "--recipe", "smol-smoltalk", "--local", str(src)])
    with pytest.raises(SystemExit, match="cannot produce"):
        hp.main(["prefs", "--recipe", "smoltalk", "--local", str(src), "--out", out])
    with pytest.raises(SystemExit, match="unknown recipe"):
        hp.main(["chat", "--recipe", "nope", "--out", out])


def test_cli_builds_shards_from_a_folder_of_text_files(tmp_path):
    d = tmp_path / "books"
    d.mkdir()
    for i in range(30):
        (d / f"{i}.txt").write_text(f"book {i} " + "lorem ipsum " * 30)
    out = str(tmp_path / "processed")
    hp.main(["text", "--recipe", "open-phi-textbooks", "--local", str(d), "--out_dir", out, "--val_permille", "100"])
    assert token_count(out, "train") > 1000 and token_count(out, "val") > 0
