"""Why doesn't the agent call prune_context? Test invitation variants offline
against a real recorded advisor-smoke session.

Variants:
  A  tool description only (what the smoke run did)
  B  one-line nudge appended to the system message
  C  just-in-time reminder appended to the newest tool result (deterministic,
     head untouched — the janitor can inject this only under budget pressure)
"""
import json
import os
import sys

import httpx

sys.path.insert(0, "src")
from ctxc.advisor import PRUNE_TOOL, annotate  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4"
N_POINTS = 3

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the repo.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor", "description": "View or edit files.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "path": {"type": "string"},
            "old_str": {"type": "string"}, "new_str": {"type": "string"}},
            "required": ["command", "path"]}}},
    PRUNE_TOOL,
]

NUDGE_SYS = ("\nContext hygiene: earlier tool outputs are labeled [block tN]. "
             "Whenever some are clearly no longer needed, ALSO call "
             "prune_context in the same turn as your main action.")
NUDGE_JIT = ("\n\n[context manager: window is filling up. If any [block tN] "
             "outputs above are no longer needed, also call prune_context "
             "with their ids in this same turn.]")


def decision_prefixes(messages, n):
    """Prefixes ending right where an assistant turn happened, deep enough to
    have several annotated blocks."""
    idxs = [i for i, m in enumerate(messages)
            if m.get("role") == "assistant" and i > len(messages) // 3]
    step = max(1, len(idxs) // n)
    return [messages[:i] for i in idxs[::step][:n]]


def run(client, prefix, variant):
    msgs = annotate(prefix)
    if variant == "B":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = msgs[0]["content"] + NUDGE_SYS
    if variant == "C":
        for j in range(len(msgs) - 1, -1, -1):
            if msgs[j].get("role") == "tool":
                msgs[j] = dict(msgs[j])
                msgs[j]["content"] = str(msgs[j]["content"]) + NUDGE_JIT
                break
    r = client.post("https://api.openai.com/v1/chat/completions", json={
        "model": MODEL, "messages": msgs, "tools": TOOLS,
    }, headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=180)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    calls = [tc["function"]["name"] for tc in msg.get("tool_calls") or []]
    pruned = []
    for tc in msg.get("tool_calls") or []:
        if tc["function"]["name"] == "prune_context":
            try:
                pruned = [b.get("id") for b in
                          json.loads(tc["function"]["arguments"]).get("blocks", [])]
            except json.JSONDecodeError:
                pruned = ["<malformed>"]
    return calls, pruned


messages = json.load(open("runs/advisor-smoke/sessions/django__django-16139.json"))["messages"]
prefixes = decision_prefixes(messages, N_POINTS)
client = httpx.Client()
for variant in ("A", "B", "C"):
    hits = 0
    for k, p in enumerate(prefixes):
        calls, pruned = run(client, p, variant)
        hit = "prune_context" in calls
        hits += hit
        print(f"{variant} point{k} depth={len(p):3d} msgs -> calls={calls} "
              f"{'PRUNED ' + str(pruned) if hit else ''}")
    print(f"variant {variant}: prune_context called at {hits}/{len(prefixes)} points\n")
