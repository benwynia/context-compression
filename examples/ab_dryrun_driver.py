"""A/B/B2 dry-run driver: a real tool-calling agent on two objectively-graded
planted-bug tasks, through three live ctxc proxies to OpenAI (gpt-4o-mini).

Not a quality benchmark (n=2, toy tasks) — a live end-to-end test of the
pipeline: proxy modes, checkpoints on real traffic, per-session accounting,
and the scrape -> resolve -> ab toolchain."""

import json
import os
import sys

import httpx

MODEL = "gpt-4o-mini"
KEY = os.environ["OPENAI_API_KEY"]

# ---- fake repo with planted bugs (bulky, distinct content per file) --------- #
def _bulk(name: str, lines: int = 260) -> str:
    return "\n".join(
        f"    # {name} module: handler branch {i} validates payload schema, "
        f"emits metric {name}_{i}, retries transient failures with backoff"
        for i in range(lines)
    )

REPO = {
    "calculator.py": f"def add(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n{_bulk('calculator')}",
    "inventory.py": f"def is_in_stock(count):\n    return count < 0\n\ndef reorder(count):\n    return max(0, 100 - count)\n{_bulk('inventory')}",
    "payments.py": f"def charge(amount):\n    return round(amount, 2)\n{_bulk('payments')}",
    "utils.py": f"def slug(s):\n    return s.lower().replace(' ', '-')\n{_bulk('utils')}",
    "config.py": f"TIMEOUT = 30\nRETRIES = 3\n{_bulk('config')}",
}

TASKS = {
    "task-1": {
        "prompt": "One Python file in this repo has an arithmetic bug (a wrong "
                  "operator). Read ALL the files with your tools, then answer "
                  "exactly: BUG: <filename> <function_name>",
        "grade": lambda ans: "calculator" in ans.lower() and "add" in ans.lower(),
    },
    "task-2": {
        "prompt": "One Python file in this repo has an inverted comparison bug. "
                  "Read ALL the files with your tools, then answer exactly: "
                  "BUG: <filename> <function_name>",
        "grade": lambda ans: "inventory" in ans.lower() and "is_in_stock" in ans.lower(),
    },
}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List repository files.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read one file's full content.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]


def run_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return "\n".join(sorted(REPO))
    if name == "read_file":
        return REPO.get(args.get("path", ""), f"ERROR: no such file {args.get('path')!r}")
    return f"ERROR: unknown tool {name}"


def run_task(port: int, task_id: str, prompt: str) -> str:
    """Real agent loop through the proxy; returns the final answer text."""
    messages = [
        {"role": "system", "content": "You are a code-review agent. Use the tools "
                                      "to read every file before answering."},
        {"role": "user", "content": prompt},
    ]
    client = httpx.Client(timeout=120.0)
    for _ in range(10):
        resp = client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "tools": TOOLS,
                  "temperature": 0, "max_tokens": 300},
            headers={"authorization": f"Bearer {KEY}",
                     "x-ctxc-session-id": task_id},
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        entry = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        messages.append(entry)
        if not msg.get("tool_calls"):
            return msg.get("content") or ""
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": run_tool(fn["name"], args)})
    return ""


if __name__ == "__main__":
    arm, port = sys.argv[1], int(sys.argv[2])
    resolved = []
    for task_id, spec in TASKS.items():
        answer = run_task(port, task_id, spec["prompt"])
        ok = spec["grade"](answer)
        print(f"[{arm}] {task_id}: resolved={ok}  answer={answer[:80]!r}")
        if ok:
            resolved.append(task_id)
    out = f"runs-dry/{arm}/resolved_ids.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(resolved))
