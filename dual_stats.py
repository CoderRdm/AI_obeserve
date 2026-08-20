"""
dual_stats.py
--------------
Reads dual_results_log.jsonl (built up by running run_dual_pipeline.py
multiple times) and prints how often the LLM-judge and the hidden-state
probe agree/disagree, plus each one's own fail rate.

This is the actual "did the paper's idea buy us anything" number: if the
probe agrees with the judge often once it's trained, that's a signal the
hidden states really do carry hallucination-relevant information, cheaply
extracted, no second LLM call needed. If it disagrees a lot or the probe
never trains, that's useful to know too -- see the README's honest
limitations section for why that's a plausible outcome at this scale.

Usage:
    python dual_stats.py
"""

import os
import json

here = os.path.dirname(__file__)
log_path = os.path.join(here, "dual_results_log.jsonl")

if not os.path.exists(log_path):
    print("No results yet. Run run_dual_pipeline.py a few times first.")
    raise SystemExit

total = 0
judge_fails = 0
probe_fails = 0
probe_scored = 0
agreements = 0

with open(log_path) as f:
    for line in f:
        entry = json.loads(line)
        judge = entry["llm_judge"]
        probe = entry["hidden_state_probe"]

        if "error" in judge:
            continue
        total += 1

        judge_is_fail = judge.get("verdict") == "FAIL"
        if judge_is_fail:
            judge_fails += 1

        if probe is not None:
            probe_scored += 1
            probe_is_fail = probe.get("verdict") == "FAIL"
            if probe_is_fail:
                probe_fails += 1
            if probe_is_fail == judge_is_fail:
                agreements += 1

print(f"Total runs: {total}")
if total > 0:
    print(f"LLM-judge fail rate: {judge_fails}/{total} ({judge_fails / total * 100:.1f}%)")

if probe_scored > 0:
    print(f"Hidden-state probe fail rate: {probe_fails}/{probe_scored} ({probe_fails / probe_scored * 100:.1f}%)")
    print(f"Agreement with LLM-judge: {agreements}/{probe_scored} ({agreements / probe_scored * 100:.1f}%)")
else:
    print("Hidden-state probe hasn't produced any predictions yet (needs training data first).")
