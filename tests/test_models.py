from ctxc.models import (
    head_len,
    round_boundary_at_or_before,
    validate_chain,
)


def _call(cid, name="read_file"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_valid_chain_passes(session_messages):
    assert validate_chain(session_messages) == []


def test_orphan_tool_is_flagged():
    chain = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "nope", "content": "x"},
    ]
    errs = validate_chain(chain)
    assert any("orphan" in e for e in errs)


def test_unanswered_call_before_user_is_flagged():
    chain = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
        {"role": "user", "content": "next"},
    ]
    errs = validate_chain(chain)
    assert any("unanswered" in e for e in errs)


def test_trailing_open_call_is_legal():
    chain = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
    ]
    assert validate_chain(chain) == []


def test_system_after_content_is_flagged():
    chain = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    errs = validate_chain(chain)
    assert any("system" in e for e in errs)


def test_head_len_covers_system_and_task():
    chain = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "x"},
    ]
    assert head_len(chain) == 3


def test_round_boundary_never_lands_on_tool():
    chain = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    # index 2 is a tool message; the boundary must back up to its assistant
    assert round_boundary_at_or_before(chain, 2, 0) == 1
