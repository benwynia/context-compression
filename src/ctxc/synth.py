"""Deterministic synthetic coding-agent sessions (OpenAI dialect).

Realistic shape for tests and the demo: a system prompt, a task statement, then
many rounds of assistant tool calls (Read/Bash/Grep/Edit) with large tool results
— some duplicated, some errors — and occasional user follow-ups. Fully seeded, so
tests are reproducible and no fixture files need committing.
"""

from __future__ import annotations

import json
import random

from .models import Message

_SYSTEM = (
    "You are a coding agent operating on a repository. Use the available tools to "
    "read, search and edit files, and run commands. Be precise and keep changes "
    "minimal. Cite files by path."
)

_TASK = (
    "Please fix the failing tests in the payments service and refactor the retry "
    "logic in the client so transient failures back off exponentially."
)

_TOOLS = ["read_file", "run_bash", "grep_search", "edit_file"]

_WORDS = (
    "retry backoff payment client timeout error handler queue worker commit "
    "index token session cache header request response parse stream chunk "
    "buffer socket flush metric trace deploy config schema migration lint"
).split()


def _blob(rng: random.Random, lines: int) -> str:
    out = []
    for i in range(lines):
        words = rng.choices(_WORDS, k=rng.randint(6, 14))
        out.append(f"{i + 1:4d} | " + " ".join(words))
    return "\n".join(out)


def synth_session(
    *,
    rounds: int = 40,
    seed: int = 7,
    result_lines: tuple[int, int] = (30, 220),
    duplicate_every: int = 9,
    error_every: int = 13,
    follow_up_every: int = 11,
) -> list[Message]:
    """Return a full message list: system, task, then ``rounds`` tool rounds."""
    rng = random.Random(seed)
    messages: list[Message] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _TASK},
    ]
    dup_pool: list[str] = []
    call_no = 0
    for r in range(rounds):
        tool = _TOOLS[rng.randrange(len(_TOOLS))]
        path = f"src/{rng.choice(_WORDS)}/{rng.choice(_WORDS)}.py"
        call_no += 1
        cid = f"call_{call_no:04d}"
        messages.append(
            {
                "role": "assistant",
                "content": f"Inspecting {path} for the {rng.choice(_WORDS)} issue.",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps({"path": path, "cmd": f"pytest {path}"}),
                        },
                    }
                ],
            }
        )
        if error_every and (r + 1) % error_every == 0:
            content = (
                f"ERROR: command failed with exit code 2\nTraceback (most recent "
                f"call last):\n  File \"{path}\", line {rng.randint(3, 99)}\n"
                f"AssertionError: expected retry after {rng.randint(2, 30)}s"
            )
        elif duplicate_every and dup_pool and (r + 1) % duplicate_every == 0:
            content = rng.choice(dup_pool)
        else:
            content = _blob(rng, rng.randint(*result_lines))
            dup_pool.append(content)
        messages.append({"role": "tool", "tool_call_id": cid, "content": content})
        if follow_up_every and (r + 1) % follow_up_every == 0:
            messages.append(
                {
                    "role": "user",
                    "content": f"Also check the {rng.choice(_WORDS)} module while you are at it.",
                }
            )
    messages.append(
        {"role": "assistant", "content": "Summary of the changes made so far: retry logic now backs off exponentially."}
    )
    return messages
