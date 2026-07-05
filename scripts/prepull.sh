#!/usr/bin/env bash
# Pre-pull all SWE-bench eval images for an instance list, with retries.
# Bash on purpose: zsh parses `$var:latest` as a `:l` modifier and mangles
# the tag — that bug burned a whole evening once.
# Usage: scripts/prepull.sh <instances.txt>
set -uo pipefail
LIST=$1
fail=0
while read -r ID; do
  [ -z "$ID" ] && continue
  img="docker.io/swebench/sweb.eval.x86_64.${ID//__/_1776_}:latest"
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "have   $ID"
    continue
  fi
  ok=0
  for try in 1 2 3 4; do
    if docker pull "$img" >/dev/null 2>&1; then ok=1; break; fi
    sleep $((try * 20))
  done
  if [ "$ok" = 1 ]; then echo "pulled $ID"; else echo "FAILED $ID"; fail=$((fail+1)); fi
done < "$LIST"
echo "prepull done, failures: $fail"
exit 0
