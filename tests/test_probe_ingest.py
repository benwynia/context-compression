"""Tests for retention probes (rung 4) and transcript import (rung 5)."""

import json

import pytest

from ctxc.ingest import claude_code_jsonl, claude_export, detect_and_convert
from ctxc.models import validate_chain
from ctxc.probe import plant_probes, render_probe_report, run_probes
from ctxc.synth import synth_session


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def test_plant_probes_preserves_structure(counter):
    msgs = synth_session(rounds=30, seed=7)
    planted, probes = plant_probes(msgs, n=8, counter=counter)
    assert validate_chain(planted) == []
    assert len(probes) == 8
    assert len(planted) == len(msgs)  # appended into messages, none added
    depths = [p.depth_pct for p in probes]
    assert depths == sorted(depths) and depths[0] < 20 and depths[-1] > 80
    for p in probes:
        assert p.nonce in planted[p.msg_index]["content"]
    # deterministic
    planted2, probes2 = plant_probes(msgs, n=8, counter=counter)
    assert [p.nonce for p in probes2] == [p.nonce for p in probes]


def test_survival_profile_shape(counter):
    msgs = synth_session(rounds=30, seed=7)
    report = run_probes(msgs, budget=8_000, n=8, counter=counter)
    assert report.compressed_tokens <= 8_000
    regions = {o.region for o in report.outcomes}
    assert "lost" in regions or "digest" in regions  # deep compression loses things
    # the shallowest probes live in the protected head/task region... the head
    # is only system+task here, so assert instead: every outcome classified
    assert all(o.region in ("head", "verbatim", "excerpt", "digest", "lost")
               for o in report.outcomes)
    # survival must be honest: lost <=> not survived
    for o in report.outcomes:
        assert o.survived == (o.region != "lost")
    text = render_probe_report(report)
    assert "survival:" in text


def test_retrieval_with_grep_model_equals_survival(counter):
    """A mock model that answers by searching its own prompt: retrieval must
    then exactly equal survival for compressed, and 100% for original."""
    msgs = synth_session(rounds=30, seed=7)

    def grep_model(context, question):
        label = question.split(" clearance")[0].split()[-1]
        flat = json.dumps(context)
        marker = f"the {label} clearance code is "
        i = flat.find(marker)
        if i >= 0:
            return flat[i + len(marker): i + len(marker) + 10]
        return "unknown"

    report = run_probes(msgs, budget=8_000, n=6, counter=counter, ask=grep_model)
    for o in report.outcomes:
        assert o.retrieved_original is True  # everything is in the full chain
        assert o.retrieved_compressed == o.survived
    text = render_probe_report(report)
    assert "retrieval:" in text


def test_generous_budget_survives_everything(counter):
    msgs = synth_session(rounds=8, seed=3, result_lines=(10, 30))
    report = run_probes(msgs, budget=10_000_000, n=5, counter=counter)
    assert report.survival_rate == 1.0


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def _claude_code_lines():
    return [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Let me read the file."},
        ]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu1", "name": "Read",
             "input": {"path": "a.py"}},
        ]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1",
             "content": [{"type": "text", "text": "print('hello')"}]},
        ]}},
        {"type": "attachment", "message": {}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "Done."}]}},
        {"type": "user", "message": {"role": "user", "content": "thanks"}},
    ]


def test_claude_code_jsonl_conversion(tmp_path):
    f = tmp_path / "agent-x.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in _claude_code_lines()))
    msgs = claude_code_jsonl(f)
    assert validate_chain(msgs) == []
    # consecutive assistant lines merged into ONE turn with text + tool_call
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Let me read the file."
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "Read"
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert msgs[1] == {"role": "tool", "tool_call_id": "tu1", "content": "print('hello')"}
    assert msgs[2]["content"] == "Done."
    assert msgs[3] == {"role": "user", "content": "thanks"}


def test_claude_export_conversion(tmp_path):
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps([
        {"name": "My chat", "chat_messages": [
            {"sender": "human", "text": "hello"},
            {"sender": "assistant", "text": "hi there"},
        ]},
        {"name": "", "chat_messages": [{"sender": "human", "text": "solo"}]},
    ]))
    sessions = claude_export(f)
    assert sessions["My chat"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert "conversation-1" in sessions


def _copilot_oplog_lines():
    node = {"node": {"children": [
        {"type": 2, "text": "line one\n"},
        {"children": [{"type": 2, "text": "line two"}]},
    ]}}
    snapshot = {"version": 3, "sessionId": "s1", "requests": []}
    request = {
        "requestId": "r1",
        "message": {"text": "raw box text"},
        "result": {"metadata": {
            "renderedUserMessage": [
                {"type": 1, "text": "rendered user turn"},
                {"type": 3, "cacheType": "ephemeral"},
            ],
            "toolCallRounds": [
                {"response": "", "toolCalls": [
                    {"name": "read_file", "arguments": '{"filePath": "a.md"}',
                     "id": "call_1__vscode-9"},
                ]},
                {"response": "All done.", "toolCalls": []},
            ],
            "toolCallResults": {
                "call_1__vscode-9": {"$mid": 20, "content": [{"$mid": 23, "value": node}]},
            },
        }},
    }
    return [
        {"kind": 0, "v": snapshot},
        {"kind": 2, "k": ["requests"], "v": [request]},
        {"kind": 1, "k": ["customTitle"], "v": "t"},
    ]


def test_vscode_copilot_jsonl_conversion(tmp_path):
    from ctxc.ingest import vscode_copilot_jsonl

    f = tmp_path / "copilot.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in _copilot_oplog_lines()))
    msgs = vscode_copilot_jsonl(f)
    assert validate_chain(msgs) == []
    # renderedUserMessage wins over the raw chat-box text
    assert msgs[0] == {"role": "user", "content": "rendered user turn"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "read_file"
    # tool result flattened from the prompt-tsx node tree, in order
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1__vscode-9",
                       "content": "line one\nline two"}
    assert msgs[3] == {"role": "assistant", "content": "All done."}
    # detect_and_convert routes op-logs by the "kind" discriminator
    assert detect_and_convert(f)["copilot"] == msgs


def test_detect_and_convert_roundtrip(tmp_path):
    f = tmp_path / "agent-y.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in _claude_code_lines()))
    sessions = detect_and_convert(f)
    assert list(sessions) == ["agent-y"]
    # a ctxc session file passes straight through
    sf = tmp_path / "s.json"
    sf.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}))
    assert detect_and_convert(sf)["s"] == [{"role": "user", "content": "x"}]


def test_cli_import_and_probe(tmp_path, capsys):
    from ctxc.cli import main

    f = tmp_path / "agent-z.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in _claude_code_lines()))
    assert main(["import", str(f), "--out", str(tmp_path / "out")]) == 0
    converted = tmp_path / "out" / "agent-z.json"
    assert converted.exists()

    session = tmp_path / "big.json"
    session.write_text(json.dumps({"messages": synth_session(rounds=20, seed=4)}))
    assert main(["probe", str(session), "--budget", "8k", "--n", "5"]) == 0
    assert "survival:" in capsys.readouterr().out
