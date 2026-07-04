"""Import real transcripts into ctxc's OpenAI-dialect session format.

Two sources today:

* **Claude Code session JSONL** (``~/.claude/projects/<proj>/<session>.jsonl``,
  including subagent transcripts) — real tool-heavy agent traffic. Anthropic
  dialect: assistant turns carry ``tool_use`` blocks; the following user line
  carries ``tool_result`` blocks.
* **claude.ai data export** (``conversations.json`` from Settings → Privacy →
  Export data) — chat conversations; mostly text, little tool bulk.

Both convert to the plain ``{"messages": [...]}`` session files every ctxc
command consumes (``verify``, ``probe``, ``compress``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Message


def _flatten_block_content(content) -> str:
    """tool_result content is a string or a list of {type: text} parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text") or part.get("content") or ""))
            else:
                out.append(str(part))
        return "\n".join(out)
    return "" if content is None else str(content)


def _repair_tool_pairing(messages: list[Message]) -> list[Message]:
    """Reorder so every tool result immediately follows the assistant that
    issued its tool_call.

    Claude Code transcripts are a message *tree* (each line has a parentUuid),
    and the file isn't always in conversation order — a tool_result can be
    written before the tool_use that created it. tool_call ids are unique, so
    we can repair deterministically: index tool results by id, then walk the
    messages emitting each assistant followed by the results answering its
    calls (in call order). Results whose call never appears are genuine
    orphans (sidechain leakage, truncated captures) and are dropped.
    """
    from .models import tool_call_ids

    results_by_id: dict[str, list[Message]] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results_by_id.setdefault(m["tool_call_id"], []).append(m)

    out: list[Message] = []
    emitted_results: set[int] = set()
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue  # emitted next to its issuing assistant, below
        out.append(m)
        if role == "assistant":
            for tcid in tool_call_ids(m):
                for res in results_by_id.get(tcid, []):
                    if id(res) not in emitted_results:
                        out.append(res)
                        emitted_results.add(id(res))
    return out


def claude_code_jsonl(path: str | Path) -> list[Message]:
    """One Claude Code session JSONL -> OpenAI-dialect messages."""
    out: list[Message] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue  # attachments, summaries, meta lines
        m = d.get("message") or {}
        content = m.get("content")
        if isinstance(content, str):
            role = "assistant" if d["type"] == "assistant" else "user"
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        if d["type"] == "assistant":
            texts: list[str] = []
            calls: list[dict] = []
            for b in content:
                kind = b.get("type")
                if kind == "text" and b.get("text"):
                    texts.append(b["text"])
                elif kind == "tool_use":
                    calls.append({
                        "id": b.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": b.get("name") or "",
                            "arguments": json.dumps(b.get("input") or {},
                                                    ensure_ascii=False),
                        },
                    })
                # thinking blocks are dropped: not part of the billable prompt
            # Claude Code splits one API turn across JSONL lines — merge
            # consecutive assistant lines back into one OpenAI assistant turn
            if out and out[-1].get("role") == "assistant" and (texts or calls):
                prev = out[-1]
                if texts:
                    prev["content"] = "\n".join(
                        t for t in [prev.get("content") or "", *texts] if t
                    )
                if calls:
                    prev.setdefault("tool_calls", []).extend(calls)
            elif texts or calls:
                msg: Message = {"role": "assistant",
                                "content": "\n".join(texts) or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        else:  # user line: tool results and/or user text
            for b in content:
                kind = b.get("type")
                if kind == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id") or "",
                        "content": _flatten_block_content(b.get("content")),
                    })
                elif kind == "text" and b.get("text"):
                    out.append({"role": "user", "content": b["text"]})
    return _repair_tool_pairing(out)


def claude_export(path: str | Path) -> dict[str, list[Message]]:
    """claude.ai conversations.json -> {conversation_name: messages}."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    sessions: dict[str, list[Message]] = {}
    for i, convo in enumerate(data):
        msgs: list[Message] = []
        for cm in convo.get("chat_messages") or []:
            role = "user" if cm.get("sender") == "human" else "assistant"
            text = cm.get("text") or _flatten_block_content(cm.get("content"))
            if text:
                msgs.append({"role": role, "content": text})
        name = (convo.get("name") or f"conversation-{i}").strip() or f"conversation-{i}"
        if msgs:
            sessions[name] = msgs
    return sessions


def detect_and_convert(path: str | Path) -> dict[str, list[Message]]:
    """Auto-detect the source format; returns {session_name: messages}."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return {p.stem: claude_code_jsonl(p)}
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "messages" in data:
        return {p.stem: data["messages"]}  # already a ctxc session file
    return claude_export(p)
