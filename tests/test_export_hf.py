"""export_hf: the files that describe the model are checked here without any ML dependency; the full
export -> load-with-transformers round trip runs wherever torch, safetensors and transformers are installed."""
import json

import pytest

from zivawhite.export_hf import build_model_card, build_modelfile, build_text_files
from zivawhite.modelconfig import ModelConfig

CFG = ModelConfig.preset("small", vocab_size=12000)
META = {"behavior_tuned": True, "step": 12000, "val_loss": 2.5}


def _files(**kw):
    base = dict(bos_id=1, eos_id=2, pad_id=0, dtype="float16", name="Ziva-White-25M", license_id=None,
                data_note=None, ckpt_meta=META)
    base.update(kw)
    return build_text_files(CFG, **base)


def test_every_expected_text_file_is_produced_and_json_files_parse():
    files = _files()
    assert set(files) == {"config.json", "generation_config.json", "tokenizer_config.json",
                          "special_tokens_map.json", "Modelfile", "README.md"}
    for name, text in files.items():
        if name.endswith(".json"):
            json.loads(text)


def test_config_json_is_a_llama_config_with_the_special_token_ids():
    cfg = json.loads(_files()["config.json"])
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert (cfg["bos_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]) == (1, 2, 0)
    assert cfg["num_key_value_heads"] == 2 and cfg["vocab_size"] == 12000


def test_tokenizer_config_carries_the_chat_template_and_context_length():
    tc = json.loads(_files()["tokenizer_config.json"])
    assert "<|assistant|>" in tc["chat_template"] and tc["model_max_length"] == 512
    assert tc["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_modelfile_reproduces_the_chat_format_and_stops_on_every_role_tag():
    mf = build_modelfile(CFG, chat=True)
    assert mf.startswith("# Ollama Modelfile") and "FROM ." in mf
    assert "<|{{ .Role }}|>" in mf and "<|assistant|>" in mf
    for tag in ("<|user|>", "<|assistant|>", "<|system|>", "<|tool|>"):
        assert f'PARAMETER stop "{tag}"' in mf
    assert "PARAMETER num_ctx 512" in mf


def test_a_base_model_gets_a_warning_instead_of_a_chat_template():
    mf = build_modelfile(CFG, chat=False)
    assert "BASE model" in mf and "TEMPLATE" not in mf


def test_model_card_is_honest_about_what_it_does_not_know():
    card = build_model_card(CFG, name="Ziva-White-25M", license_id=None, data_note=None, ckpt_meta=META, dtype="float16")
    assert card.startswith("---") and "license:" not in card.split("---")[1]     # no license invented
    assert "TODO before publishing" in card and "not set" in card
    assert "may be wrong" in card or "often **wrong**" in card
    assert "Validation loss at export: 2.500" in card and "12,000" in card
    assert "22.6M" in card or "22.5M" in card                                      # excl. embedding, from the real config
    assert "instruction/chat-tuned" in card and "apply_chat_template" in card


def test_model_card_for_a_base_model_does_not_pretend_to_chat():
    card = build_model_card(CFG, name="x", license_id="apache-2.0", data_note="my data, CC0",
                            ckpt_meta={"behavior_tuned": False}, dtype="float32")
    assert "BASE language model" in card and "apply_chat_template" not in card
    assert "license: apache-2.0" in card.split("---")[1] and "my data, CC0" in card and "TODO" not in card


def test_exporting_an_unsupported_architecture_is_refused_before_anything_is_written():
    from zivawhite.hf_compat import hf_config_dict

    with pytest.raises(ValueError):
        hf_config_dict(ModelConfig(n_head=2, n_embd=32, use_moe=True))


def test_full_export_round_trip_matches_the_original_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from zivawhite.checkpoint import save_checkpoint
    from zivawhite.export_hf import export_hf
    from zivawhite.model import ZivaGPT
    from zivawhite.tokenizer import load_tokenizer

    tok = load_tokenizer(None)                          # the bundled BPE tokenizer
    cfg = ModelConfig.preset("tiny", vocab_size=tok.vocab_size, block_size=64)
    torch.manual_seed(0)
    model = ZivaGPT(cfg)
    ck = tmp_path / "checkpoint.pt"
    save_checkpoint(str(ck), model=model, step=7, cfg=cfg, extra={"behavior_tuned": True, "val_loss": 3.0})
    res = export_hf(str(ck), str(tmp_path / "out"), dtype="float32", name="t", verify=True)
    assert res["verify"]["ok"] and res["verify"]["max_abs_diff"] < 1e-3
    assert {"model.safetensors", "config.json", "tokenizer.json", "README.md", "Modelfile"} <= set(res["files"])
