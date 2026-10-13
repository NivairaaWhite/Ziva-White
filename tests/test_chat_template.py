"""The exported Hugging Face chat_template must build exactly the prompt chat.py builds."""
import pytest

from zivawhite.chat import HF_CHAT_TEMPLATE, build_chat_prompt, render_conversation, render_turn

jinja2 = pytest.importorskip("jinja2")


def _render(messages, add_generation_prompt=True):
    # Hugging Face compiles chat templates in exactly this environment
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(HF_CHAT_TEMPLATE).render(messages=messages, bos_token="<bos>",
                                                    add_generation_prompt=add_generation_prompt)


CASES = [
    [{"role": "user", "content": "hello"}],
    [{"role": "user", "content": "  padded  "}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}],
    [{"role": "user", "content": "multi\nline\n\ncontent"}],
    [{"role": "user", "content": "what is 2+2?"},
     {"role": "assistant", "content": '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'},
     {"role": "tool", "content": "4"}],
]


@pytest.mark.parametrize("messages", CASES)
def test_template_equals_build_chat_prompt(messages):
    # inference builds: render_conversation(history + new turn) + "\n" + assistant tag; reproduce that directly
    prompt = "<bos>" + render_conversation(messages) + "\n" + "<|assistant|>" + "\n"
    assert _render(messages) == prompt


def test_user_turn_prompt_matches_build_chat_prompt_with_history():
    history = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "a"},
               {"role": "assistant", "content": "b"}]
    assert _render(history + [{"role": "user", "content": "c"}]) == "<bos>" + build_chat_prompt("c", history)


def test_without_generation_prompt_the_conversation_is_just_rendered_turns():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert _render(msgs, add_generation_prompt=False) == "<bos>" + render_conversation(msgs) + "\n"


def test_unknown_roles_are_rejected_instead_of_becoming_assistant_turns():
    with pytest.raises(ValueError):
        render_turn("bot", "hi")
    assert render_turn("system", " rules ") == "<|system|>\nrules"
    assert render_turn("tool", "4") == "<|tool|>\n4"
