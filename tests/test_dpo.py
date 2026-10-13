"""dpo.py: preference-data handling (pure Python) and the DPO maths (needs torch; skipped without it)."""
import json
import math

import pytest

from zivawhite.dpo import (encode_pair, load_preferences, normalize_preference, split_pairs, write_pairs)
from zivawhite.sft import IGNORE
from zivawhite.tokenizer import ByteLevelTokenizer

TOK = ByteLevelTokenizer()
U = lambda t: {"role": "user", "content": t}
A = lambda t: {"role": "assistant", "content": t}


# ------------------------------------------------------------------------------ data shapes
def test_plain_string_prompt():
    assert normalize_preference({"prompt": "q", "chosen": " good ", "rejected": "bad"}) == ([U("q")], "good", "bad")


def test_message_list_prompt_keeps_system_and_history():
    p = [{"role": "system", "content": "s"}, U("a"), A("b"), U("c")]
    hist, c, r = normalize_preference({"prompt": p, "chosen": "x", "rejected": "y"})
    assert [m["role"] for m in hist] == ["system", "user", "assistant", "user"] and (c, r) == ("x", "y")


def test_ultrafeedback_style_prompt_is_implied_by_the_chosen_conversation():
    rec = {"chosen": [U("Why is the sky blue?"), A("Rayleigh scattering.")],
           "rejected": [U("Why is the sky blue?"), A("Because it reflects the sea.")]}
    assert normalize_preference(rec) == ([U("Why is the sky blue?")], "Rayleigh scattering.", "Because it reflects the sea.")


def test_unusable_records_are_dropped():
    assert normalize_preference({"prompt": "q", "chosen": "same", "rejected": "same"}) is None
    assert normalize_preference({"prompt": "q", "chosen": "", "rejected": "bad"}) is None
    assert normalize_preference({"chosen": "a", "rejected": "b"}) is None            # no prompt anywhere
    assert normalize_preference("nope") is None


def test_load_preferences_reads_jsonl_and_skips_junk(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text("\n".join([json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}), "{bad", json.dumps({"prompt": "q"})]))
    assert len(load_preferences([str(f)])) == 1


def test_written_self_play_pairs_load_back(tmp_path):
    pairs = [([U("q1")], "gold", "sample"), ([U("q2"), A("a"), U("q3")], "gold2", "sample2")]
    path = str(tmp_path / "out" / "pairs.jsonl")
    write_pairs(path, pairs)
    assert load_preferences([path]) == pairs


# --------------------------------------------------------------------------------- encoding
def test_chosen_and_rejected_share_the_exact_prompt_and_differ_only_in_the_reply():
    (ci, cl), (ri, rl) = encode_pair(TOK, ([U("What is 2+2?")], "It is 4.", "No idea, sorry."), 256)
    n = sum(1 for t in cl if t == IGNORE)
    assert n == sum(1 for t in rl if t == IGNORE) and ci[:n] == ri[:n]               # identical prompt tokens
    assert TOK.decode([t for t in cl if t != IGNORE]) == "It is 4."
    assert TOK.decode([t for t in rl if t != IGNORE]) == "No idea, sorry."
    assert cl[-1] == rl[-1] == TOK.eos_id                                            # both end their turn


def test_split_pairs_is_deterministic_and_skipped_for_tiny_sets():
    items = list(range(100))
    a, b = split_pairs(items, 0.1, 5)
    assert len(b) == 10 and set(a).isdisjoint(b) and split_pairs(items, 0.1, 5) == (a, b)
    assert split_pairs(items[:10], 0.1, 5) == (items[:10], [])


# ------------------------------------------------------------------------ the maths (torch)
def test_loss_is_ln2_when_policy_equals_reference_and_falls_as_the_margin_grows():
    torch = pytest.importorskip("torch")
    from zivawhite.dpo import dpo_loss

    z = torch.zeros(3)
    loss, margin = dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.full((3,), math.log(2)), atol=1e-6) and torch.allclose(margin, z)
    better, m = dpo_loss(torch.tensor([5.0]), torch.tensor([-5.0]), torch.zeros(1), torch.zeros(1), beta=0.1)
    worse, m2 = dpo_loss(torch.tensor([-5.0]), torch.tensor([5.0]), torch.zeros(1), torch.zeros(1), beta=0.1)
    assert better < math.log(2) < worse and m > 0 > m2
    # matches the closed form -log(sigmoid(beta * delta)) for a hand-picked delta
    d = 0.1 * ((3.0 - 1.0) - (-2.0 - 0.5))
    one, _ = dpo_loss(torch.tensor([3.0]), torch.tensor([-2.0]), torch.tensor([1.0]), torch.tensor([0.5]), beta=0.1)
    assert float(one) == pytest.approx(math.log1p(math.exp(-d)), rel=1e-5)


def test_sequence_logps_matches_a_manual_log_softmax_and_ignores_masked_positions():
    torch = pytest.importorskip("torch")
    from zivawhite.dpo import sequence_logps
    from zivawhite.model import ModelConfig, ZivaGPT

    torch.manual_seed(0)
    model = ZivaGPT(ModelConfig(vocab_size=40, n_layer=2, n_head=2, n_kv_head=1, n_embd=32, block_size=16)).eval()
    x = torch.randint(0, 40, (2, 10))
    y = torch.randint(0, 40, (2, 10))
    y[:, :4] = IGNORE
    y[1, 8:] = IGNORE
    got, n = sequence_logps(model, x, y)
    ref = torch.log_softmax(model.hidden_logits(x).float(), -1)
    for b in range(2):
        want = sum(float(ref[b, t, y[b, t]]) for t in range(10) if y[b, t] != IGNORE)
        assert float(got[b]) == pytest.approx(want, rel=1e-4) and int(n[b]) == int((y[b] != IGNORE).sum())


def test_a_few_dpo_steps_raise_the_chosen_over_rejected_margin_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    from zivawhite.dpo import dpo_loss, sequence_logps
    from zivawhite.model import ModelConfig, ZivaGPT
    from zivawhite.sft import pad_batch

    torch.manual_seed(0)
    model = ZivaGPT(ModelConfig(vocab_size=ByteLevelTokenizer.vocab_size, n_layer=2, n_head=2, n_kv_head=1, n_embd=32, block_size=64))
    pair = encode_pair(TOK, ([U("hi")], "hello there", "go away"), 64)
    xs, ys = pad_batch([pair[0], pair[1]], TOK.pad_id)
    x, y = torch.tensor(xs), torch.tensor(ys)

    def margins():
        lp, _ = sequence_logps(model, x, y)
        return lp[:1], lp[1:]
    with torch.no_grad():
        rc, rr = [t.clone() for t in margins()]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        c, r = margins()
        loss, m = dpo_loss(c, r, rc, rr, beta=0.5)
        first = float(m) if first is None else first
        opt.zero_grad(); loss.mean().backward(); opt.step()
    assert float(m) > first + 0.5
