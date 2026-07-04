"""Retention probes — the information-loss profile of compression (rung 4).

Plant uniquely-labeled verifiable facts at spread depths in a (real or
synthetic) session, compress, and measure two things per probe:

* **survival** — does the fact physically exist anywhere in the compressed
  chain (verbatim, truncated-excerpt, or digest)? Free, offline, exact.
* **retrieval** (``--live``) — can the model actually answer a question whose
  answer is that fact, given only the compressed context? Paired against the
  uncompressed context, graded by exact string match — no judge model.

Survival is the ceiling for retrieval: a fact that didn't survive cannot be
retrieved. The gap between the two (survived-but-not-retrieved) measures how
usable the compressed representations are, which is the quality question the
deterministic digest lives or dies on.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .compressor import CompressConfig, compress
from .models import Message, copy_chain, head_len, is_digest
from .tokens import TokenCounter

_LABELS = [
    "amber", "cobalt", "crimson", "jade", "onyx", "pearl", "sienna", "teal",
    "ivory", "maroon", "olive", "quartz",
]


def _nonce(rng: random.Random) -> str:
    return f"{rng.choice(['FROST', 'EMBER', 'RIDGE', 'DELTA'])}-{rng.randint(1000, 9999)}"


@dataclass
class Probe:
    label: str
    fact: str          # the full planted line
    nonce: str         # the answer a correct retrieval must contain
    msg_index: int     # where it was planted (index in the planted chain)
    depth_pct: float   # token position as % of total chain

    @property
    def question(self) -> str:
        return (
            f"Earlier in this conversation an AUDIT-NOTE stated the {self.label} "
            f"clearance code. Reply with ONLY that code and nothing else."
        )


@dataclass
class ProbeOutcome:
    probe: Probe
    region: str                    # head | verbatim | excerpt | digest | lost
    survived: bool
    retrieved_compressed: bool | None = None
    retrieved_original: bool | None = None


@dataclass
class ProbeReport:
    budget: int
    original_tokens: int = 0
    compressed_tokens: int = 0
    outcomes: list[ProbeOutcome] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        n = len(self.outcomes)
        return sum(o.survived for o in self.outcomes) / n if n else 0.0


def plant_probes(
    messages: list[Message],
    n: int = 8,
    seed: int = 0,
    counter: TokenCounter | None = None,
) -> tuple[list[Message], list[Probe]]:
    """Copy of the chain with ``n`` labeled facts appended to plain-string
    messages at evenly spread token depths (structure untouched)."""
    counter = counter or TokenCounter()
    rng = random.Random(seed)
    msgs = copy_chain(messages)
    candidates = [
        i for i, m in enumerate(msgs)
        if isinstance(m.get("content"), str) and m["content"]
        and m.get("role") in ("user", "tool", "assistant")
    ]
    if not candidates:
        raise ValueError("no plain-text messages to plant probes into")
    cum, running = [], 0
    for m in msgs:
        running += counter.count_message(m)
        cum.append(running)
    total = running

    probes: list[Probe] = []
    n = min(n, len(_LABELS), len(candidates))
    for k in range(n):
        target_pct = (k + 0.5) / n  # spread: 1/2n, 3/2n, ...
        idx = min(candidates, key=lambda i: abs(cum[i] / total - target_pct))
        candidates.remove(idx)
        label = _LABELS[k]
        nonce = _nonce(rng)
        fact = f"AUDIT-NOTE: the {label} clearance code is {nonce}."
        msgs[idx] = dict(msgs[idx])
        msgs[idx]["content"] = f"{msgs[idx]['content']}\n{fact}"
        probes.append(Probe(label, fact, nonce, idx, 100.0 * cum[idx] / total))
    return msgs, probes


def _classify(probe: Probe, planted: list[Message], compressed: list[Message]) -> str:
    original_msg = planted[probe.msg_index]
    h = head_len(planted)
    for m in compressed:
        if m == original_msg:
            return "head" if probe.msg_index < h else "verbatim"
    for m in compressed:
        content = m.get("content")
        if isinstance(content, str) and probe.nonce in content:
            return "digest" if is_digest(m) else "excerpt"
    return "lost"


def run_probes(
    messages: list[Message],
    budget: int,
    *,
    n: int = 8,
    seed: int = 0,
    config: CompressConfig | None = None,
    counter: TokenCounter | None = None,
    ask: Callable[[list[Message], str], str] | None = None,
) -> ProbeReport:
    """Plant, compress, classify; if ``ask(context_messages, question) -> answer``
    is given, also measure paired retrieval (compressed vs original)."""
    counter = counter or TokenCounter()
    planted, probes = plant_probes(messages, n=n, seed=seed, counter=counter)
    res = compress(planted, budget, config, counter)
    report = ProbeReport(
        budget=budget,
        original_tokens=res.original_tokens,
        compressed_tokens=res.compressed_tokens,
    )
    for p in probes:
        region = _classify(p, planted, res.messages)
        out = ProbeOutcome(p, region, survived=region != "lost")
        if ask is not None:
            out.retrieved_compressed = p.nonce in ask(res.messages, p.question)
            out.retrieved_original = p.nonce in ask(planted, p.question)
        report.outcomes.append(out)
    return report


def render_probe_report(r: ProbeReport) -> str:
    live = any(o.retrieved_compressed is not None for o in r.outcomes)
    lines = [
        f"ctxc probe — retention profile at budget {r.budget:,} "
        f"({r.original_tokens:,} -> {r.compressed_tokens:,} tokens)",
        "",
        f"  {'depth':>6}  {'region':<9} {'survived':<9}"
        + ("compressed  original" if live else ""),
    ]
    for o in sorted(r.outcomes, key=lambda o: o.probe.depth_pct):
        row = (
            f"  {o.probe.depth_pct:>5.0f}%  {o.region:<9} "
            f"{'yes' if o.survived else 'NO':<9}"
        )
        if live:
            row += (
                f"{'yes' if o.retrieved_compressed else 'NO':<12}"
                f"{'yes' if o.retrieved_original else 'NO'}"
            )
        lines.append(row)
    lines.append("")
    lines.append(f"  survival: {sum(o.survived for o in r.outcomes)}"
                 f"/{len(r.outcomes)} ({100 * r.survival_rate:.0f}%)")
    if live:
        rc = sum(bool(o.retrieved_compressed) for o in r.outcomes)
        ro = sum(bool(o.retrieved_original) for o in r.outcomes)
        lines.append(f"  retrieval: compressed {rc}/{len(r.outcomes)}, "
                     f"original {ro}/{len(r.outcomes)}"
                     f"  (gap = usability of compressed representations)")
    lines.append("  regions: head/verbatim = untouched; excerpt = survived "
                 "truncation; digest = summary line only; lost = evicted")
    return "\n".join(lines)
