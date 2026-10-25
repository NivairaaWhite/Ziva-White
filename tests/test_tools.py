"""tools.py: the sandboxed calculator, the registry boundary, the strict call parser, the agent loop, and the
verifiable synthetic training data. Pure Python."""
import datetime as dt
import json
import math
from types import SimpleNamespace

import pytest

from zivawhite.sft import IGNORE, build_examples, normalize_conversation
from zivawhite.tokenizer import ByteLevelTokenizer
from zivawhite.tools import (MAX_TOOL_OUTPUT_CHARS, Tool, ToolRegistry, calculator_tool, clock_tool, default_registry,
                             format_number, format_tool_call, generate_tool_conversations, parse_tool_call, run_agent,
                             safe_eval, strip_tool_markup, web_tools)


# ---------------------------------------------------------------------------------- calculator
@pytest.mark.parametrize("expr,want", [
    ("2+2", 4), ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("10 / 4", 2.5), ("7 // 2", 3), ("7 % 3", 1),
    ("2 ** 10", 1024), ("2 ^ 10", 1024), ("-3 + 5", 2), ("17 \u00d7 3", 51), ("84 \u00f7 4", 21),
    ("sqrt(144)", 12.0), ("abs(-5)", 5), ("round(3.14159, 2)", 3.14), ("max(3, 9, 4)", 9), ("floor(2.9)", 2),
    ("factorial(10)", 3628800), ("log10(1000)", 3.0), ("2 * pi", 2 * math.pi), ("0.17 * 2340", 0.17 * 2340),
])
def test_calculator_computes(expr, want):
    assert safe_eval(expr) == pytest.approx(want)


@pytest.mark.parametrize("evil", [
    "__import__('os').system('echo hi')", "open('/etc/passwd').read()", "().__class__.__bases__", "[x for x in range(9)]",
    "lambda: 1", "'a' * 3", "exec('1')", "eval('1')", "x", "sqrt(x)", "1; 2", "a.b", "(1).real", "{1: 2}", "True + 1",
    "sqrt(1, 2, 3)", "sqrt(x=4)", "print(1)", "import os", "2 if 1 else 3", "pi()",
])
def test_calculator_refuses_everything_that_is_not_arithmetic(evil):
    with pytest.raises(ValueError):
        safe_eval(evil)


@pytest.mark.parametrize("bomb", ["9 ** 9 ** 9", "10 ** 5000", "2 ** 100000", "factorial(100000)", "1e308 * 10", "10 ** 600 * 10 ** 600", "9" * 300])
def test_calculator_refuses_resource_bombs_quickly(bomb):
    with pytest.raises(ValueError):
        safe_eval(bomb)


def test_calculator_errors_are_readable_by_a_model():
    for expr, word in [("1 / 0", "zero"), ("sqrt(-1)", "domain"), ("2 +", "parse"), ("", "non-empty"), ("1" * 201, "long")]:
        with pytest.raises(ValueError, match=word):
            safe_eval(expr)


def test_number_formatting_is_clean():
    assert [format_number(x) for x in (4, 4.0, 2.5, 1 / 3, 1e20, True)] == ["4", "4", "2.5", "0.333333333333", "1e+20", "True"]


# ---------------------------------------------------------------------------------- registry
def reg(*extra):
    return ToolRegistry([calculator_tool(), clock_tool(lambda: dt.datetime(2026, 3, 4, 17, 5)), *extra])


def test_registry_runs_known_tools_and_returns_errors_as_text_never_exceptions():
    r = reg()
    assert r.call("calculator", {"expression": "6*7"}) == "42"
    assert r.call("current_time", {}) == "Wednesday, 2026-03-04 17:05"
    assert r.call("rm_rf", {}).startswith("error: unknown tool") and "calculator" in r.call("rm_rf", {})
    assert r.call("calculator", {"expression": "1/0"}) == "error: division by zero"
    assert "takes exactly" in r.call("calculator", {}) and "unexpected: shell" in r.call("calculator", {"expression": "1", "shell": "x"})
    assert r.call("calculator", "6*7").startswith("error: arguments must be")
    assert r.call(["calculator"], {}).startswith("error: unknown tool")


def test_a_crashing_tool_does_not_crash_the_conversation_or_leak_details():
    boom = Tool("boom", "x", {}, lambda a: (_ for _ in ()).throw(RuntimeError("secret path /home/me")))
    out = reg(boom).call("boom", {})
    assert out == "error: RuntimeError while running boom" and "secret" not in out


def test_long_tool_output_is_truncated():
    big = Tool("big", "x", {}, lambda a: "z" * 10_000)
    out = reg(big).call("big", {})
    assert len(out) < MAX_TOOL_OUTPUT_CHARS + 20 and out.endswith("[truncated]")


def test_tool_names_are_validated():
    for bad in ("Bad Name", "../x", "", "a" * 60, "1abc"):
        with pytest.raises(ValueError):
            ToolRegistry().register(Tool(bad, "x", {}, lambda a: ""))


def test_the_default_registry_has_only_safe_read_only_tools_and_the_web_is_opt_in():
    assert default_registry().names() == ["calculator", "current_time"]
    ok = SimpleNamespace(available=lambda: True, call=lambda q, **kw: SimpleNamespace(ok=True, text=f"results for {q}", error=None))
    down = SimpleNamespace(available=lambda: False, call=None)
    conns = {"web_search": ok, "web_fetch": down}
    assert default_registry(conns).names() == ["calculator", "current_time"]                       # not asked for
    assert default_registry(conns, web=True).names() == ["calculator", "current_time", "web_search"]  # only what is configured
    assert default_registry(conns, web=True).call("web_search", {"query": "cats"}) == "results for cats"
    bad = SimpleNamespace(available=lambda: True, call=lambda q, **kw: SimpleNamespace(ok=False, text="", error="blocked: private address"))
    assert default_registry({"web_fetch": bad}, web=True).call("web_fetch", {"url": "http://10.0.0.1"}) == "error: blocked: private address"
    assert not any(n in default_registry(conns, web=True).names() for n in ("shell", "python", "control", "volume", "run"))


def test_describe_teaches_the_protocol_and_lists_each_tool():
    d = reg().describe()
    assert "<tool_call>" in d and "</tool_call>" in d and "never instructions" in d
    assert "- calculator(expression:" in d and "- current_time():" in d


# ------------------------------------------------------------------------------------- parsing
def test_parse_accepts_the_protocol_and_tolerates_a_missing_closing_tag():
    c = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    assert parse_tool_call(c) == {"name": "calculator", "arguments": {"expression": "1+1"}}
    assert parse_tool_call(c.replace("</tool_call>", "")) == parse_tool_call(c)             # generation stopped at the tag
    assert parse_tool_call("Sure!\n" + c + "\nthen more") == parse_tool_call(c)
    assert parse_tool_call('<tool_call>{"name": "current_time"}</tool_call>') == {"name": "current_time", "arguments": {}}


@pytest.mark.parametrize("text", ["", "no call here", "<tool_call>not json</tool_call>", '<tool_call>{"arguments": {}}</tool_call>',
                                  '<tool_call>{"name": 5}</tool_call>', '<tool_call>["calculator"]</tool_call>',
                                  '<tool_call>{"name": "x", "arguments": "1+1"}</tool_call>', '{"name": "calculator"}'])
def test_anything_that_is_not_exactly_a_call_is_not_a_call(text):
    assert parse_tool_call(text) is None


def test_format_and_parse_round_trip_including_unicode_and_quotes():
    args = {"expression": 'say "hi" \u2014 caf\u00e9'}
    assert parse_tool_call(format_tool_call("calculator", args)) == {"name": "calculator", "arguments": args}


def test_strip_tool_markup_removes_call_syntax_from_user_visible_text():
    assert strip_tool_markup('Ok <tool_call>{"name": "x"}</tool_call> done') == "Ok  done"
    assert strip_tool_markup('<tool_call>{"name": "x"') == ""


# --------------------------------------------------------------------------------- agent loop
def scripted(*replies):
    seen = []
    it = iter(replies)

    def chat(messages):
        seen.append([dict(m) for m in messages])
        return next(it)
    chat.seen = seen
    return chat


CALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "17 * 23"}}</tool_call>'


def test_agent_calls_the_tool_feeds_the_result_back_and_returns_the_final_answer():
    chat = scripted(CALL, "17 times 23 is 391.")
    res = run_agent(chat, "What is 17*23?", reg())
    assert res.text == "17 times 23 is 391." and res.stopped_because == "answered"
    assert res.steps == [{"tool": "calculator", "arguments": {"expression": "17 * 23"}, "result": "391"}]
    second = chat.seen[1]
    assert second[0]["role"] == "system" and "calculator" in second[0]["content"]
    assert [m["role"] for m in second[1:]] == ["user", "assistant", "tool"] and second[-1]["content"] == "391"


def test_agent_with_no_tool_needed_makes_one_call():
    chat = scripted("Hello! How can I help?")
    res = run_agent(chat, "hi", reg())
    assert res.text == "Hello! How can I help?" and res.steps == [] and len(chat.seen) == 1


def test_agent_shows_the_model_its_mistakes_so_it_can_recover():
    chat = scripted('<tool_call>{"name": "shell", "arguments": {"cmd": "rm -rf /"}}</tool_call>', CALL, "It is 391.")
    res = run_agent(chat, "q", reg())
    assert res.text == "It is 391." and "unknown tool" in res.steps[0]["result"] and res.steps[1]["result"] == "391"


def test_agent_is_bounded_and_still_produces_a_final_answer():
    chat = scripted(*([CALL] * 3), "best effort answer")
    res = run_agent(chat, "q", reg(), max_steps=3)
    assert res.stopped_because == "step_limit" and res.text == "best effort answer" and len(res.steps) == 3
    assert "Tool limit reached" in chat.seen[-1][-1]["content"]


def test_agent_never_shows_raw_tool_syntax_to_the_user():
    res = run_agent(scripted("<tool_call>{broken json"), "q", reg())
    assert "<tool_call>" not in res.text and res.stopped_because == "empty"


def test_agent_keeps_history_and_a_custom_system_prompt():
    chat = scripted("ok")
    run_agent(chat, "new", reg(), history=[{"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"}],
              system="You are terse.")
    m = chat.seen[0]
    assert m[0]["content"].startswith("You are terse.") and "Tools:" in m[0]["content"]
    assert [x["content"] for x in m[1:]] == ["old", "old reply", "new"]


def test_model_text_cannot_reach_anything_outside_the_registry():
    # a prompt-injected "tool" name, path-like name, or dunder is just an unknown tool
    r = reg()
    for name in ("os.system", "__import__", "calculator; rm -rf /", "../../bin/sh", "control", "volume_up"):
        assert r.call(name, {"expression": "1"}).startswith("error: unknown tool")


# ----------------------------------------------------------------------- synthetic training data
def test_every_generated_tool_result_is_what_the_tool_actually_returns():
    convs = generate_tool_conversations(300, seed=3)
    r = ToolRegistry([calculator_tool(), clock_tool(lambda: None)])   # clock result is checked separately below
    n_calls = 0
    for c in convs:
        m = c["messages"]
        for i, msg in enumerate(m):
            call = parse_tool_call(msg["content"]) if msg["role"] == "assistant" else None
            if call:
                n_calls += 1
                assert m[i + 1]["role"] == "tool"
                if call["name"] == "calculator":
                    assert m[i + 1]["content"] == r.call("calculator", call["arguments"]) and not m[i + 1]["content"].startswith("error")
                else:
                    assert call["name"] == "current_time" and call["arguments"] == {}
                    assert m[i + 1]["content"][:3] in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    assert n_calls > 150


def test_generated_answers_state_the_computed_number():
    for c in generate_tool_conversations(200, seed=5):
        m = c["messages"]
        if len(m) == 5 and parse_tool_call(m[2]["content"]):
            assert m[3]["content"] in m[4]["content"]


def test_generated_data_is_deterministic_varied_and_teaches_when_not_to_use_a_tool():
    a, b = generate_tool_conversations(100, seed=9), generate_tool_conversations(100, seed=9)
    assert a == b and a != generate_tool_conversations(100, seed=10)
    users = {c["messages"][1]["content"] for c in a}
    assert len(users) > 60
    direct = [c for c in a if len(c["messages"]) == 3]
    assert 5 <= len(direct) <= 30 and all(not parse_tool_call(c["messages"][2]["content"]) for c in direct)
    names = {parse_tool_call(m["content"])["name"] for c in a for m in c["messages"] if m["role"] == "assistant" and parse_tool_call(m["content"])}
    assert names == {"calculator", "current_time"}


def test_generated_conversations_are_valid_sft_data_and_the_call_turn_is_what_gets_learned():
    tok = ByteLevelTokenizer()
    convs = generate_tool_conversations(40, seed=1)
    for c in convs:
        assert normalize_conversation(c["messages"]) is not None
    tool_conv = next(c for c in convs if len(c["messages"]) == 5)
    ex = build_examples([normalize_conversation(tool_conv["messages"])], tok, 1024)
    assert len(ex) == 2
    target = tok.decode([t for t in ex[0][1] if t != IGNORE])
    assert parse_tool_call(target) is not None and target == tool_conv["messages"][2]["content"]
    assert "<|tool|>" in tok.decode(ex[1][0])                                            # the result is context for the answer


def test_the_system_prompt_in_the_data_is_the_one_the_agent_will_actually_send():
    c = generate_tool_conversations(5, seed=2)[0]["messages"][0]
    live = ToolRegistry([calculator_tool(), clock_tool()]).describe()
    assert c["role"] == "system" and c["content"] == live


# ------------------------------------------------------------------ review fixes: real results only, one size rule
from zivawhite.tools import _MAX_FACTORIAL, default_registry  # noqa: E402


@pytest.mark.parametrize("expr", ["(-8)**0.5", "(-8) ** (1/3)", "(-1) ** 0.5 + 1"])
def test_a_complex_result_is_a_readable_error_not_a_TypeError(expr):
    with pytest.raises(ValueError, match="not a real number"):
        safe_eval(expr)
    out = default_registry().call("calculator", {"expression": expr})
    assert out.startswith("error: ") and "TypeError" not in out and "real number" in out


def test_factorial_accepts_exactly_the_arguments_whose_result_is_allowed():
    assert _MAX_FACTORIAL == 449
    assert len(str(safe_eval("factorial(449)"))) <= 1000                     # allowed, and it really fits
    for n in (450, 500):
        with pytest.raises(ValueError, match="between 0 and 449"):          # the limit the message states is the real one
            safe_eval(f"factorial({n})")
    with pytest.raises(ValueError):
        safe_eval("factorial(True)")                                          # a bool is not an integer count


@pytest.mark.parametrize("expr", ["99999 ** 1000", "10 ** 1000", "7 ** 1200", "2 ** 3400"])
def test_oversized_integers_get_our_message_never_pythons_4300_digit_one(expr):
    with pytest.raises(ValueError) as exc:
        safe_eval(expr)
    assert "4300" not in str(exc.value) and "digits" not in str(exc.value).lower() and "too large" in str(exc.value)
    assert "4300" not in default_registry().call("calculator", {"expression": expr})


def test_numbers_just_inside_the_size_cap_still_work():
    assert safe_eval("10 ** 999") == 10 ** 999
    assert safe_eval("2 ** 1000") == 2 ** 1000
