"""
stats.py
--------
Reads results_log.jsonl (built up by running checker.py multiple times)
and prints your real hallucination/failure rate.

Usage:
    python stats.py
"""

import os
import json

here = os.path.dirname(__file__)
log_path = os.path.join(here, "results_log.jsonl")

if not os.path.exists(log_path):
    print("No results yet. Run agent_runner.py then checker.py a few times first.")
    raise SystemExit

total = 0
fails = 0
fail_types = {"unsupported_claims": 0, "unflagged_contradictions": 0,
              "skipped_files": 0, "invented_info": 0}

with open(log_path) as f:
    for line in f:
        entry = json.loads(line)
        v = entry["verdict"]
        if "error" in v:
            continue
        total += 1
        if v.get("verdict") == "FAIL":
            fails += 1
            for key in fail_types:
                if v.get(key):
                    fail_types[key] += 1

print(f"Total runs: {total}")
if total > 0:
    print(f"Failed runs: {fails} ({fails / total * 100:.1f}%)")
    print("\nBreakdown of failure types (a run can have more than one):")
    for k, v in fail_types.items():
        print(f"  {k}: {v}")
