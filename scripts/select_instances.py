"""Pre-registered instance selection. Seed fixed BEFORE any results are seen.

Pilot: seeded shuffle of SWE-bench Lite (test), take instances one per repo
until 5, skipping the 5 rung-7 astropy instances (first 5 by instance_id).
Main run: next 50 by the same shuffle, max 8 per repo, disjoint from pilot.
"""
import random
from datasets import load_dataset

SEED = 42
rows = sorted(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"),
              key=lambda r: r["instance_id"])
rung7 = {r["instance_id"] for r in rows[:5]}
rng = random.Random(SEED)
shuffled = rows[:]
rng.shuffle(shuffled)

pilot, seen_repos = [], set()
for r in shuffled:
    if r["instance_id"] in rung7 or r["repo"] in seen_repos:
        continue
    pilot.append(r["instance_id"]); seen_repos.add(r["repo"])
    if len(pilot) == 5:
        break

main, per_repo = [], {}
taken = set(pilot)
for r in shuffled:
    if r["instance_id"] in rung7 or r["instance_id"] in taken:
        continue
    if per_repo.get(r["repo"], 0) >= 8:
        continue
    main.append(r["instance_id"]); per_repo[r["repo"]] = per_repo.get(r["repo"], 0) + 1
    if len(main) == 50:
        break

open("runs/instances-pilot.txt", "w").write("\n".join(pilot) + "\n")
open("runs/instances-main.txt", "w").write("\n".join(main) + "\n")
print("pilot:", *pilot, sep="\n  ")
print("main: %d instances, repos: %s" % (len(main), sorted(per_repo)))
