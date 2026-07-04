"""Paired A/B comparison of agent-benchmark runs: ctxc arm vs control arm.

Input: two result sets — one row per task per arm. A row is a JSON object:

    {
      "task_id": "sympy__sympy-20590",        # pairing key (required)
      "resolved": true,                        # objective grader verdict (required)
      "requests": 42,
      "prompt_tokens": 1830042,                # provider-reported, all turns
      "output_tokens": 20411,
      "cache_read": 1520000,                   # optional: cache-tier split
      "cache_write": 310042,
      "wall_seconds": 512.0,                   # optional
      "checkpoints": 3                         # optional: ctxc arm engagement
    }

Load from a directory of ``*.json`` files (one per task) or a ``.jsonl`` file.
Rows come from the benchmark harness (resolved) joined with the proxy's
``GET /stats/sessions`` (cost), keyed by the ``x-ctxc-session-id`` the runner
set per task.

Output: paired deltas only over tasks present in BOTH arms — resolved-rate
delta with an exact McNemar test on the discordant pairs, token/AIC deltas
with a bootstrap CI, and a segmentation by whether compression actually
engaged (checkpoints > 0), because tasks that never crossed the budget can
only dilute the comparison.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import httpx

from .aic import AicRate, aic_cached_for, aic_for, usd_for


def scrape_row(
    proxy_url: str, task_id: str, client: httpx.Client | None = None
) -> dict:
    """Build one result row from a running proxy's ``/stats/sessions``.

    ``prompt_tokens`` prefers the upstream's own reported usage (what was
    actually billed); it falls back to the local emitted count when the
    upstream reported nothing (e.g. streamed turns). ``resolved`` is left
    False — merge the grader verdict in afterwards with :func:`mark_resolved`.
    """
    http = client or httpx.Client(timeout=30.0)
    data = http.get(proxy_url.rstrip("/") + "/stats/sessions").json()
    row = data["sessions"].get(task_id)
    if row is None:
        raise KeyError(
            f"no session {task_id!r} on {proxy_url} "
            f"(known: {sorted(data['sessions'])[:5]}…)"
        )
    out = {
        "task_id": task_id,
        "resolved": False,
        "requests": row["requests"],
        "prompt_tokens": row["upstream_prompt_tokens"] or row["emitted_tokens"],
        "output_tokens": row["upstream_completion_tokens"],
        "checkpoints": row["checkpoints"],
        "original_tokens": row["original_tokens"],
        "emitted_tokens": row["emitted_tokens"],
    }
    if row.get("upstream_cached_tokens"):
        out["cache_read"] = row["upstream_cached_tokens"]
    return out


def mark_resolved(results_dir: str | Path, resolved_ids: set[str]) -> int:
    """Set ``resolved`` on every row file from the grader's id list (True when
    listed, False otherwise — grader truth for all). Returns rows updated."""
    n = 0
    for f in sorted(Path(results_dir).glob("*.json")):
        row = json.loads(f.read_text())
        row["resolved"] = row["task_id"] in resolved_ids
        f.write_text(json.dumps(row, indent=1))
        n += 1
    return n


def load_results(path: str | Path) -> dict[str, dict]:
    """``{task_id: row}`` from a dir of *.json rows or a .jsonl file."""
    p = Path(path)
    rows: list[dict] = []
    if p.is_dir():
        for f in sorted(p.glob("*.json")):
            rows.append(json.loads(f.read_text()))
    elif p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else [data]
    out: dict[str, dict] = {}
    for r in rows:
        tid = r.get("task_id")
        if not tid:
            raise ValueError(f"{path}: row without task_id: {r}")
        if tid in out:
            raise ValueError(f"{path}: duplicate task_id {tid!r}")
        out[tid] = r
    return out


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test on the discordant pair counts (binomial
    with p=0.5). 1.0 when there are no discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(
    deltas: list[float], iterations: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile 95% CI for the mean of paired deltas (seeded, reproducible)."""
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(iterations)
    )
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations)])


def _row_aic(row: dict, rate: AicRate) -> float:
    flat = aic_for(
        rate,
        input_tokens=int(row.get("prompt_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        requests=int(row.get("requests") or 0),
    )
    if rate.cache_aware and ("cache_read" in row or "cache_write" in row):
        return (
            aic_cached_for(
                rate,
                cache_read=int(row.get("cache_read") or 0),
                cache_write=int(row.get("cache_write") or 0),
                requests=int(row.get("requests") or 0),
            )
            + int(row.get("output_tokens") or 0) / 1_000_000 * rate.per_1m_output
        )
    return flat


@dataclass
class ArmTotals:
    tasks: int = 0
    resolved: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    aic: float = 0.0

    @property
    def resolved_rate(self) -> float:
        return self.resolved / self.tasks if self.tasks else 0.0


@dataclass
class AbReport:
    paired_tasks: list[str] = field(default_factory=list)
    unpaired_a: list[str] = field(default_factory=list)
    unpaired_b: list[str] = field(default_factory=list)
    ctxc: ArmTotals = field(default_factory=ArmTotals)
    control: ArmTotals = field(default_factory=ArmTotals)
    both_resolved: int = 0
    neither_resolved: int = 0
    only_ctxc_resolved: int = 0
    only_control_resolved: int = 0
    mcnemar_p: float = 1.0
    aic_delta_mean: float = 0.0          # ctxc - control, per task (negative = saves)
    aic_delta_ci: tuple[float, float] = (0.0, 0.0)
    prompt_tokens_saved_pct: float = 0.0
    aic_saved_pct: float = 0.0
    engaged_tasks: int = 0               # ctxc checkpoints > 0
    engaged_prompt_tokens_saved_pct: float = 0.0
    engaged_only_ctxc: int = 0
    engaged_only_control: int = 0


def compare(
    ctxc_results: dict[str, dict],
    control_results: dict[str, dict],
    rate: AicRate,
    *,
    bootstrap_seed: int = 0,
) -> AbReport:
    r = AbReport()
    ids = sorted(set(ctxc_results) & set(control_results))
    r.paired_tasks = ids
    r.unpaired_a = sorted(set(ctxc_results) - set(control_results))
    r.unpaired_b = sorted(set(control_results) - set(ctxc_results))

    deltas: list[float] = []
    engaged_prompt = [0, 0]  # control, ctxc — over engaged tasks only
    for tid in ids:
        a, b = ctxc_results[tid], control_results[tid]
        ra, rb = bool(a.get("resolved")), bool(b.get("resolved"))
        r.ctxc.tasks += 1
        r.control.tasks += 1
        r.ctxc.resolved += ra
        r.control.resolved += rb
        r.both_resolved += ra and rb
        r.neither_resolved += not ra and not rb
        r.only_ctxc_resolved += ra and not rb
        r.only_control_resolved += rb and not ra
        for arm, row in ((r.ctxc, a), (r.control, b)):
            arm.prompt_tokens += int(row.get("prompt_tokens") or 0)
            arm.output_tokens += int(row.get("output_tokens") or 0)
            arm.requests += int(row.get("requests") or 0)
        aa, ab_ = _row_aic(a, rate), _row_aic(b, rate)
        r.ctxc.aic += aa
        r.control.aic += ab_
        deltas.append(aa - ab_)
        if int(a.get("checkpoints") or 0) > 0:
            r.engaged_tasks += 1
            engaged_prompt[0] += int(b.get("prompt_tokens") or 0)
            engaged_prompt[1] += int(a.get("prompt_tokens") or 0)
            r.engaged_only_ctxc += ra and not rb
            r.engaged_only_control += rb and not ra

    r.mcnemar_p = mcnemar_exact_p(r.only_ctxc_resolved, r.only_control_resolved)
    if deltas:
        r.aic_delta_mean = sum(deltas) / len(deltas)
        r.aic_delta_ci = bootstrap_mean_ci(deltas, seed=bootstrap_seed)
    if r.control.prompt_tokens:
        r.prompt_tokens_saved_pct = 100.0 * (
            1 - r.ctxc.prompt_tokens / r.control.prompt_tokens
        )
    if r.control.aic:
        r.aic_saved_pct = 100.0 * (1 - r.ctxc.aic / r.control.aic)
    if engaged_prompt[0]:
        r.engaged_prompt_tokens_saved_pct = 100.0 * (
            1 - engaged_prompt[1] / engaged_prompt[0]
        )
    return r


def render_ab(r: AbReport) -> str:
    lines = [
        "ctxc A/B report — ctxc (compressed) vs control (direct)",
        f"  paired tasks            : {len(r.paired_tasks)}"
        + (f"  (unpaired dropped: {len(r.unpaired_a)} ctxc, {len(r.unpaired_b)} control)"
           if r.unpaired_a or r.unpaired_b else ""),
        "",
        f"  resolved  ctxc          : {r.ctxc.resolved}/{r.ctxc.tasks}"
        f" ({100 * r.ctxc.resolved_rate:.1f}%)",
        f"  resolved  control       : {r.control.resolved}/{r.control.tasks}"
        f" ({100 * r.control.resolved_rate:.1f}%)",
        f"  discordant pairs        : ctxc-only {r.only_ctxc_resolved},"
        f" control-only {r.only_control_resolved}",
        f"  McNemar exact p         : {r.mcnemar_p:.4f}"
        f"  ({'no significant quality difference detected' if r.mcnemar_p >= 0.05 else 'SIGNIFICANT quality difference — investigate before shipping'})",
        "",
        f"  prompt tokens           : ctxc {r.ctxc.prompt_tokens:,} vs"
        f" control {r.control.prompt_tokens:,}  (saved {r.prompt_tokens_saved_pct:.1f}%)",
        f"  AIC                     : ctxc {r.ctxc.aic:,.1f} (${usd_for(r.ctxc.aic):,.2f}) vs"
        f" control {r.control.aic:,.1f} (${usd_for(r.control.aic):,.2f})"
        f"  (saved {r.aic_saved_pct:.1f}%)",
        f"  per-task AIC delta      : {r.aic_delta_mean:+.2f} mean,"
        f" 95% CI [{r.aic_delta_ci[0]:+.2f}, {r.aic_delta_ci[1]:+.2f}]"
        f"  (negative = ctxc cheaper)",
        "",
        f"  compression engaged     : {r.engaged_tasks}/{len(r.paired_tasks)} tasks"
        f" (checkpoints > 0)",
    ]
    if r.engaged_tasks:
        lines += [
            f"  engaged-only tokens     : saved {r.engaged_prompt_tokens_saved_pct:.1f}%",
            f"  engaged-only discordant : ctxc-only {r.engaged_only_ctxc},"
            f" control-only {r.engaged_only_control}",
        ]
    else:
        lines += [
            "  WARNING: compression never engaged — the budget is above every",
            "  task's chain length; this run measures nothing about ctxc.",
        ]
    return "\n".join(lines)
