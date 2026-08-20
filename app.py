"""
app.py
------
Streamlit UI for AgentAudit. Lets you paste a task, run it against the
agent, see the raw answer, and see the audit flags highlighted - plus a
running stats dashboard across everything logged in results_log.jsonl.

Requires: streamlit, and agent_runner.py + checker.py in the same folder.
Requires: GROQ_API_KEY set as an environment variable or in a .env file.

Usage:
    streamlit run app.py
"""

import os
import json
import streamlit as st

import agent_runner as ar
import checker as ck

st.set_page_config(page_title="AgentAudit", page_icon="🔎", layout="wide")

HERE = os.path.dirname(__file__)
LOG_PATH = os.path.join(HERE, "results_log.jsonl")
DUAL_LOG_PATH = os.path.join(HERE, "dual_results_log.jsonl")

st.title("🔎 AgentAudit")
st.caption("Watches an LLM agent's answers and checks them against the real source documents.")

tab_run, tab_dashboard, tab_compare = st.tabs(
    ["Run a test", "Stats dashboard", "LLM-judge vs hidden-state probe"]
)

# ---------------------------------------------------------------------------
# TAB 1: Run a single test live
# ---------------------------------------------------------------------------
with tab_run:
    docs = ar.load_source_documents()

    with st.expander(f"Source documents in test_files/ ({len(docs)} files)", expanded=False):
        for name, content in docs.items():
            st.markdown(f"**{name}**")
            st.code(content, language="text")

    task = st.text_input(
        "Task to give the agent",
        value="Summarize each file and give total Q3 revenue.",
    )

    if st.button("Run agent + audit", type="primary"):
        if not docs:
            st.error("No files found in test_files/. Add some .txt files there first.")
        else:
            with st.spinner("Running agent..."):
                try:
                    prompt = ar.build_prompt(task, docs)
                    answer = ar.call_groq(prompt)
                except Exception as e:
                    st.error(f"Agent call failed: {e}")
                    answer = None

            if answer:
                st.subheader("Agent answer")
                st.write(answer)

                run_data = {"task": task, "docs": docs, "agent_answer": answer}

                with st.spinner("Running audit..."):
                    try:
                        verdict = ck.run_check(run_data)
                    except Exception as e:
                        st.error(f"Checker call failed: {e}")
                        verdict = None

                if verdict:
                    st.subheader("Audit result")

                    if verdict.get("verdict") == "FAIL":
                        st.error("FAIL — issues found")
                    elif verdict.get("verdict") == "PASS":
                        st.success("PASS — no issues found")
                    else:
                        st.warning("Could not determine a clean verdict")

                    flag_labels = {
                        "unsupported_claims": "Unsupported claims",
                        "unflagged_contradictions": "Unflagged contradictions",
                        "skipped_files": "Skipped files",
                        "invented_info": "Invented info",
                    }
                    for key, label in flag_labels.items():
                        items = verdict.get(key) or []
                        if items:
                            st.markdown(f"**🚩 {label}:**")
                            for item in items:
                                st.markdown(f"- {item}")

                    if verdict.get("notes"):
                        st.caption(verdict["notes"])

                    # log it, same format the CLI checker.py uses
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps({"task": task, "verdict": verdict}) + "\n")
                    st.caption("Logged to results_log.jsonl")

# ---------------------------------------------------------------------------
# TAB 2: Stats dashboard across everything logged so far
# ---------------------------------------------------------------------------
with tab_dashboard:
    if not os.path.exists(LOG_PATH):
        st.info("No runs logged yet. Run a few tests in the other tab first.")
    else:
        total = 0
        fails = 0
        fail_types = {
            "unsupported_claims": 0,
            "unflagged_contradictions": 0,
            "skipped_files": 0,
            "invented_info": 0,
        }
        rows = []

        with open(LOG_PATH) as f:
            for line in f:
                entry = json.loads(line)
                v = entry["verdict"]
                if "error" in v:
                    continue
                total += 1
                is_fail = v.get("verdict") == "FAIL"
                if is_fail:
                    fails += 1
                    for key in fail_types:
                        if v.get(key):
                            fail_types[key] += 1
                rows.append({
                    "task": entry["task"],
                    "verdict": v.get("verdict", "?"),
                    "notes": v.get("notes", ""),
                })

        col1, col2 = st.columns(2)
        col1.metric("Total runs", total)
        if total > 0:
            col2.metric("Fail rate", f"{fails / total * 100:.1f}%")

            st.subheader("Failure type breakdown")
            st.bar_chart(fail_types)

            st.subheader("All runs")
            st.dataframe(rows, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Compare the LLM-judge against the local hidden-state probe
# ---------------------------------------------------------------------------
with tab_compare:
    st.caption(
        "Runs from run_dual_pipeline.py (uses the local Qwen2.5-0.5B model + "
        "hidden-state probing, not Groq). Run that script from a terminal a "
        "few times first -- it needs `torch`/`transformers` installed and "
        "isn't triggered from this UI, since the first run downloads a model."
    )

    if not os.path.exists(DUAL_LOG_PATH):
        st.info(
            "No dual-pipeline runs logged yet. From a terminal, run:\n\n"
            "`python run_dual_pipeline.py \"your task here\"`\n\n"
            "a handful of times with different tasks, then reload this tab."
        )
    else:
        total = 0
        judge_fails = 0
        probe_fails = 0
        probe_scored = 0
        agreements = 0
        rows = []

        with open(DUAL_LOG_PATH) as f:
            for line in f:
                entry = json.loads(line)
                judge = entry["llm_judge"]
                probe = entry["hidden_state_probe"]

                if "error" in judge:
                    continue
                total += 1

                judge_verdict = judge.get("verdict", "?")
                judge_is_fail = judge_verdict == "FAIL"
                if judge_is_fail:
                    judge_fails += 1

                probe_verdict = "not trained yet"
                probe_prob = None
                if probe is not None:
                    probe_scored += 1
                    probe_verdict = probe.get("verdict", "?")
                    probe_prob = probe.get("probability")
                    probe_is_fail = probe_verdict == "FAIL"
                    if probe_is_fail:
                        probe_fails += 1
                    if probe_is_fail == judge_is_fail:
                        agreements += 1

                rows.append({
                    "task": entry["task"],
                    "llm_judge_verdict": judge_verdict,
                    "probe_verdict": probe_verdict,
                    "probe_P(fail)": f"{probe_prob:.2f}" if probe_prob is not None else "-",
                    "agree": (probe_verdict == judge_verdict) if probe is not None else None,
                })

        col1, col2, col3 = st.columns(3)
        col1.metric("Total runs", total)
        if total > 0:
            col2.metric("LLM-judge fail rate", f"{judge_fails / total * 100:.1f}%")
        if probe_scored > 0:
            col3.metric("Probe/judge agreement", f"{agreements / probe_scored * 100:.1f}%")
        else:
            col3.metric("Probe/judge agreement", "n/a (probe not trained yet)")

        st.subheader("Run-by-run comparison")
        st.dataframe(rows, use_container_width=True)

        st.caption(
            "Agreement measures whether the hidden-state probe, working only "
            "from internal activations of the local model, reaches the same "
            "verdict as the LLM-judge, working only from reading the final "
            "text. High agreement suggests hidden states really do carry a "
            "hallucination-relevant signal (the paper's core claim, tested "
            "cheaply here). The probe was trained on the judge's own "
            "verdicts as weak labels, so this measures agreement with the "
            "judge, not agreement with ground truth -- see the README."
        )