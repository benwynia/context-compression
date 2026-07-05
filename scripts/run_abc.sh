#!/usr/bin/env bash
# Three-arm graded run on the frozen rung-8 instance list, fully unattended:
#   prepull -> arm A (passthrough) -> arm B (deterministic 5k)
#           -> arm C (advisor 5k)  -> grade all -> resolve + ab reports
# All arms: gpt-5.4 (arm C's sidecar judgment is frontier-tier-gated, and a
# fair comparison needs one model everywhere). Each arm gets a main pass and
# one sweep pass (run_arm.sh skips completed instances). Every phase is
# resumable: rerunning this script skips finished work.
# Usage: scripts/run_abc.sh [instances.txt] [out-dir]
set -uo pipefail
cd "$(dirname "$0")/.."
LIST="${1:-runs/instances-main.txt}"
DIR="${2:-runs/abc}"
export MODEL=gpt-5.4 COST_LIMIT=2.00 CONFIG="$PWD/scripts/sweagent-nofilemap.yaml"
set -a; source .env; set +a
mkdir -p "$DIR"

echo "### prepull $(date +%F\ %T)"
bash scripts/prepull.sh "$LIST"

run_one_arm() {  # name port extra-proxy-flags...
  local ARM=$1 PORT=$2; shift 2
  mkdir -p "$DIR/$ARM"/{sessions,results}
  echo "### arm $ARM $(date +%F\ %T)"
  pkill -f "ctxc proxy" 2>/dev/null; sleep 2
  nohup uv run ctxc proxy --upstream https://api.openai.com \
    --record "$DIR/$ARM/sessions" --port "$PORT" "$@" \
    > "$DIR/$ARM/proxy.log" 2>&1 &
  local PROXY_PID=$!
  sleep 5
  bash scripts/run_arm.sh "$ARM" "$PORT" "$LIST" "$DIR"   # main pass
  bash scripts/run_arm.sh "$ARM" "$PORT" "$LIST" "$DIR"   # sweep pass
  curl -s "http://localhost:$PORT/stats" > "$DIR/$ARM/proxy-stats.json" || true
  curl -s "http://localhost:$PORT/stats/sessions" > "$DIR/$ARM/proxy-session-stats.json" || true
  kill "$PROXY_PID" 2>/dev/null
}

run_one_arm A 8791 --budget 60k --passthrough
run_one_arm B 8790 --budget 5k
run_one_arm C 8793 --budget 5k --advisor

echo "### grading $(date +%F\ %T)"
for ARM in A B C; do
  : > "$DIR/$ARM/preds.jsonl"
  for f in "$DIR/$ARM"/agent/*/*.pred; do
    [ -f "$f" ] && python3 -c "import json;print(json.dumps(json.load(open('$f'))))" >> "$DIR/$ARM/preds.jsonl"
  done
  echo "grading $ARM: $(wc -l < "$DIR/$ARM/preds.jsonl") preds"
  "$HOME/swebench-harness/.venv/bin/python" -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-bench_Lite -p "$DIR/$ARM/preds.jsonl" \
    --run_id "abc$ARM" --max_workers 4 2>&1 | tail -3
  python3 -c "
import json
r = json.load(open('agent.abc$ARM.json'))
open('$DIR/$ARM/resolved_ids.txt','w').write('\n'.join(r['resolved_ids'])+'\n')
print('$ARM resolved:', r['resolved_instances'])
"
  mv "agent.abc$ARM.json" "$DIR/$ARM/" 2>/dev/null || true
  uv run ctxc resolve "$DIR/$ARM/results" --ids-file "$DIR/$ARM/resolved_ids.txt"
done

echo "### reports $(date +%F\ %T)"
for pair in "B A" "C A" "C B"; do
  set -- $pair
  echo "===== ctxc ab: $1 vs $2 ====="
  uv run ctxc ab "$DIR/$1/results" "$DIR/$2/results" \
    --rates scripts/rates.json --model gpt-5.4 \
    --json "$DIR/ab-$1-vs-$2.json" || true
done
echo "### ALL DONE $(date +%F\ %T)"
