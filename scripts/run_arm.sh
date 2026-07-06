#!/usr/bin/env bash
# Run one A/B arm: SWE-agent per-instance through a ctxc proxy, cost row after each.
# Usage: scripts/run_arm.sh <ARM> <PORT> <instances.txt> <phase-dir>
#   e.g. scripts/run_arm.sh A 8791 runs/instances-pilot.txt runs/pilot
# Per-instance launches so each task gets its own x-ctxc-session-id header
# (SWE-agent's first message is a static system prompt, so the proxy's
# first-message fallback would collapse all instances into one session).
# run-batch skips instances that already have a trajectory, so reruns resume.
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=$1 PORT=$2 LIST=$3 DIR=$4
CONFIG="${CONFIG:-$HOME/SWE-agent/config/default.yaml}"
MODEL="${MODEL:-gpt-5.4-mini}"
COST_LIMIT="${COST_LIMIT:-1.50}"
set -a; source .env; set +a

while read -r ID; do
  [ -z "$ID" ] && continue
  echo "=== [$ARM] $ID $(date +%H:%M:%S) ==="
  "$HOME/SWE-agent/.venv/bin/sweagent" run-batch \
    --config "$CONFIG" \
    --agent.model.name "$MODEL" \
    --agent.model.api_base "http://localhost:$PORT/v1" \
    --agent.model.per_instance_cost_limit "$COST_LIMIT" \
    --agent.model.completion_kwargs "{\"extra_headers\":{\"x-ctxc-session-id\":\"$ID\"}}" \
    --instances.type swe_bench --instances.subset lite --instances.split test \
    --instances.filter "$ID" \
    --output_dir "$DIR/$ARM/agent" --num_workers 1 \
    || echo "!!! [$ARM] $ID agent run failed — recorded, continuing"
  uv run ctxc scrape --proxy "http://localhost:$PORT" --task-id "$ID" \
    --out "$DIR/$ARM/results" \
    || echo "!!! [$ARM] $ID scrape failed"
done < "$LIST"
echo "=== [$ARM] arm complete ==="
