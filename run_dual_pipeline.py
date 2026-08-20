"""
run_dual_pipeline.py
----------------------
Runs ONE task through both hallucination detectors and logs both
verdicts side by side, so you can compare them:

  1. LLM-judge (checker.py, unchanged): a second LLM reads the answer +
     source docs and gives a PASS/FAIL verdict with reasons.
  2. Hidden-state probe (hidden_state_probe.py +
     hallucination_classifier.py, new): a small classifier reads
     statistics of the local model's internal activations while it
     generated the answer, and gives its own PASS/FAIL verdict -- no
     second LLM call involved.

Every run also logs the hidden-state features + the LLM-judge's verdict
as a training example (see hallucination_classifier.log_labeled_example),
so the probe classifier keeps improving as you run this more.

Usage:
    python run_dual_pipeline.py "Summarize each file and give total Q3 revenue."

Run it repeatedly with different task phrasings (same idea as the
original README's advice for checker.py) to build up both:
  (a) real LLM-judge fail-rate stats (already supported by stats.py)
  (b) enough labeled examples for the hidden-state probe to train on
"""

import os
import sys
import json

import local_agent_runner as lar
import hidden_state_probe as probe
import hallucination_classifier as clf
import checker as ck

HERE = os.path.dirname(__file__)
DUAL_LOG_PATH = os.path.join(HERE, "dual_results_log.jsonl")


def run_once(task):
    print(f"\n=== TASK ===\n{task}\n")

    # 1. Run the local model, capturing hidden states as it generates.
    print("Running local model...")
    docs, answer, hidden_states = lar.run_local(task, capture_hidden_states=True)
    print("=== AGENT ANSWER ===")
    print(answer)

    # 2. LLM-judge verdict (same checker.py used by the Groq pipeline;
    #    it doesn't care which model produced the answer).
    print("\nRunning LLM-judge audit...")
    run_data = {"task": task, "docs": docs, "agent_answer": answer}
    judge_verdict = ck.run_check(run_data)
    judge_is_fail = judge_verdict.get("verdict") == "FAIL"
    print(f"LLM-judge verdict: {judge_verdict.get('verdict', '?')}")

    # 3. Hidden-state features for this run.
    features, feature_names = probe.extract_features(hidden_states)

    # 4. Log this as a training example for the probe classifier, using
    #    the judge's verdict as a (weak/noisy) label. See the big comment
    #    in hallucination_classifier.py for why this is a limitation
    #    worth knowing about.
    if "error" not in judge_verdict:
        clf.log_labeled_example(features, feature_names, label=int(judge_is_fail), task=task)
        clf.retrain_if_stale()
    else:
        print("(Judge output could not be parsed -- skipping this run as a training example.)")

    # 5. Hidden-state probe's own prediction, if a model has been trained.
    probe_result = clf.predict(features)
    if probe_result is None:
        print("\nHidden-state probe: not trained yet (needs a few more runs across both PASS and FAIL).")
    else:
        print(
            f"\nHidden-state probe verdict: {probe_result['verdict']} "
            f"(P(fail)={probe_result['probability']:.2f}, "
            f"trained on {probe_result['trained_on']} examples)"
        )

    # 6. Log both verdicts side by side.
    entry = {
        "task": task,
        "agent_answer": answer,
        "llm_judge": judge_verdict,
        "hidden_state_probe": probe_result,
    }
    with open(DUAL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nAppended to {DUAL_LOG_PATH}")

    return entry


if __name__ == "__main__":
    try:
        task = sys.argv[1] if len(sys.argv) > 1 else "Summarize each file in one paragraph."
        run_once(task)
    except Exception as error:
        print(f"run_dual_pipeline.py failed: {error}")
        raise SystemExit(1)
