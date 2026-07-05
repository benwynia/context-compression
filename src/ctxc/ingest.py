"""Import real transcripts into ctxc's OpenAI-dialect session format.

Three sources today:

* **Claude Code session JSONL** (``~/.claude/projects/<proj>/<session>.jsonl``,
  including subagent transcripts) — real tool-heavy agent traffic. Anthropic
  dialect: assistant turns carry ``tool_use`` blocks; the following user line
  carries ``tool_result`` blocks.
* **claude.ai data export** (``conversations.json`` from Settings → Privacy →
  Export data) — chat conversations; mostly text, little tool bulk.
* **VS Code Copilot Chat session JSONL** (``~/Library/Application Support/
  Code/User/workspaceStorage/<ws>/chatSessions/<id>.jsonl``) — an op-log of
  the chat widget's state. The API-level conversation lives in each request's
  ``result.metadata``: ``toolCallRounds`` (assistant text + OpenAI-dialect
  tool calls per round) and ``toolCallResults`` (tool outputs as serialized
  prompt-tsx node trees, keyed by tool-call id).

All convert to the plain ``{"messages": [...]}`` session files every ctxc
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
    return out


def _copilot_replay_oplog(path: str | Path) -> dict:
    """Replay a VS Code chat-session op-log JSONL into the session object.

    Line kinds: 0 = initial snapshot, 1 = set value at path ``k``,
    2 = extend the list at path ``k`` with the items in ``v``.
    """
    obj: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("kind")
        if kind == 0:
            obj = d.get("v") or {}
        elif kind in (1, 2):
            keys = d.get("k") or []
            if not keys:
                continue
            target = obj
            try:
                for k in keys[:-1]:
                    target = target[k] if isinstance(target, list) else target.setdefault(k, {})
                last = keys[-1]
                if kind == 1:
                    target[last] = d.get("v")
                else:
                    cur = target[last] if isinstance(target, list) else target.setdefault(last, [])
                    if not isinstance(cur, list):
                        cur = []
                        target[last] = cur
                    v = d.get("v")
                    cur.extend(v if isinstance(v, list) else [v])
            except (KeyError, IndexError, TypeError):
                continue  # a malformed op should not sink the whole import
    return obj


def _copilot_flatten_tool_result(entry) -> str:
    """Flatten a serialized prompt-tsx node tree to its text, in order."""
    out: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str):
                out.append(text)
            value = node.get("value")
            if isinstance(value, str):
                out.append(value)
            for key in ("node", "children", "content", "value"):
                child = node.get(key)
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(entry)
    return "".join(out)


def _copilot_user_text(request: dict) -> str:
    """The user turn as the model saw it: renderedUserMessage (attachments and
    all) when persisted, else the raw chat-box text."""
    md = ((request.get("result") or {}).get("metadata") or {})
    rendered = md.get("renderedUserMessage")
    if isinstance(rendered, list):
        parts = [p.get("text") for p in rendered
                 if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if parts:
            return "\n".join(parts)
    return ((request.get("message") or {}).get("text") or "")


def vscode_copilot_jsonl(path: str | Path) -> list[Message]:
    """One VS Code Copilot Chat session op-log JSONL -> OpenAI-dialect messages."""
    obj = _copilot_replay_oplog(path)
    out: list[Message] = []
    for request in obj.get("requests") or []:
        text = _copilot_user_text(request)
        if text:
            out.append({"role": "user", "content": text})
        md = ((request.get("result") or {}).get("metadata") or {})
        results = md.get("toolCallResults") or {}
        for rnd in md.get("toolCallRounds") or []:
            calls = []
            for tc in rnd.get("toolCalls") or []:
                calls.append({
                    "id": tc.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": tc.get("name") or "",
                        "arguments": tc.get("arguments") or "{}",
                    },
                })
            response = rnd.get("response") or None
            if calls or response:
                msg: Message = {"role": "assistant", "content": response}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            for tc in calls:
                out.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _copilot_flatten_tool_result(results.get(tc["id"])),
                })
    return out


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
        first = next((ln for ln in p.read_text().splitlines() if ln.strip()), "")
        try:
            head = json.loads(first)
        except json.JSONDecodeError:
            head = {}
        # VS Code chat op-logs open with {"kind": 0, "v": {...}}; Claude Code
        # lines carry a "type" field instead
        if isinstance(head, dict) and "kind" in head and "type" not in head:
            return {p.stem: vscode_copilot_jsonl(p)}
        return {p.stem: claude_code_jsonl(p)}
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "messages" in data:
        return {p.stem: data["messages"]}  # already a ctxc session file
    return claude_export(p)
