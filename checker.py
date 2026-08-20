# """
# checker.py
# ----------
# The "audit" step. Takes the agent's answer + the real source documents,
# and asks Claude to act as a strict fact-checker: does the answer contain
# claims not supported by the source text? Does it silently pick a side on
# a contradiction instead of flagging it? Does it invent info that isn't
# in any file?

# Requires: ANTHROPIC_API_KEY environment variable set.
# Usage:
#     python checker.py
# (reads last_run.json produced by agent_runner.py)
# """

# import os
# import json
# import urllib.request

# API_URL = "https://api.anthropic.com/v1/messages"
# MODEL = "claude-sonnet-4-6"

# CHECKER_SYSTEM_PROMPT = """You are a strict fact-checking auditor for AI agent outputs.
# You will be given the REAL SOURCE DOCUMENTS and an AGENT ANSWER that claims to be
# based on them. Your job:

# 1. List any specific claim in the agent answer that is NOT supported by the source
#    documents (hallucination).
# 2. Check if the source documents contain any contradicting information on the same
#    fact (e.g. two different numbers for the same thing). If so, check whether the
#    agent answer silently picked one value without flagging the conflict.
# 3. Check if any source file's content was ignored/skipped in the answer.
# 4. Check if the agent invented an answer to something the documents do not mention,
#    instead of saying it isn't available.

# Respond ONLY in this exact JSON format, nothing else, no markdown fences:
# {
#   "unsupported_claims": ["..."],
#   "unflagged_contradictions": ["..."],
#   "skipped_files": ["..."],
#   "invented_info": ["..."],
#   "verdict": "PASS" or "FAIL",
#   "notes": "one sentence summary"
# }
# "verdict" is FAIL if any of the four lists above is non-empty, otherwise PASS.
# """


# def call_claude(system, user_content):
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("Set ANTHROPIC_API_KEY environment variable first.")

#     body = json.dumps({
#         "model": MODEL,
#         "max_tokens": 1000,
#         "system": system,
#         "messages": [{"role": "user", "content": user_content}],
#     }).encode("utf-8")

#     req = urllib.request.Request(
#         API_URL,
#         data=body,
#         headers={
#             "Content-Type": "application/json",
#             "x-api-key": api_key,
#             "anthropic-version": "2023-06-01",
#         },
#         method="POST",
#     )
#     with urllib.request.urlopen(req) as resp:
#         data = json.loads(resp.read())
#     return "".join(block["text"] for block in data["content"] if block["type"] == "text")


# def run_check(run_data):
#     doc_block = "\n\n".join(
#         f"--- FILE: {name} ---\n{content}" for name, content in run_data["docs"].items()
#     )
#     user_content = (
#         f"SOURCE DOCUMENTS:\n{doc_block}\n\n"
#         f"TASK GIVEN TO AGENT: {run_data['task']}\n\n"
#         f"AGENT ANSWER:\n{run_data['agent_answer']}"
#     )
#     raw = call_claude(CHECKER_SYSTEM_PROMPT, user_content)
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         return {"error": "Could not parse checker output", "raw": raw}


# if __name__ == "__main__":
#     here = os.path.dirname(__file__)
#     with open(os.path.join(here, "last_run.json")) as f:
#         run_data = json.load(f)

#     verdict = run_check(run_data)
#     print("=== AUDIT RESULT ===")
#     print(json.dumps(verdict, indent=2))

#     # Append to a running results log so you can compute stats across many runs
#     log_path = os.path.join(here, "results_log.jsonl")
#     with open(log_path, "a") as f:
#         f.write(json.dumps({"task": run_data["task"], "verdict": verdict}) + "\n")
#     print(f"\nAppended to {log_path}")
"""
checker.py (Groq version)
---------------------------
The "audit" step. Takes the agent's answer + the real source documents,
and asks an LLM (via Groq) to act as a strict fact-checker.

Requires: GROQ_API_KEY environment variable set (or a .env file with it).
Usage:
    python checker.py
(reads last_run.json produced by agent_runner.py)
"""

import os
import json
import urllib.request
import urllib.error

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_file()


def get_groq_models():
    configured_models = os.environ.get("GROQ_MODELS", "").strip()
    if configured_models:
        return [model.strip() for model in configured_models.split(",") if model.strip()]

    configured_model = os.environ.get("GROQ_MODEL", "").strip()
    if configured_model:
        return [configured_model]

    return DEFAULT_MODELS

CHECKER_SYSTEM_PROMPT = """You are a strict fact-checking auditor for AI agent outputs.
You will be given the REAL SOURCE DOCUMENTS and an AGENT ANSWER that claims to be
based on them. Your job:

1. List any specific claim in the agent answer that is NOT supported by the source
   documents (hallucination).
2. Check if the source documents contain any contradicting information on the same
   fact (e.g. two different numbers for the same thing). If so, check whether the
   agent answer silently picked one value without flagging the conflict.
3. Check if any source file's content was ignored/skipped in the answer.
4. Check if the agent invented an answer to something the documents do not mention,
   instead of saying it isn't available.

Respond ONLY in this exact JSON format, nothing else, no markdown fences, no
explanation before or after the JSON:
{
  "unsupported_claims": ["..."],
  "unflagged_contradictions": ["..."],
  "skipped_files": ["..."],
  "invented_info": ["..."],
  "verdict": "PASS" or "FAIL",
  "notes": "one sentence summary"
}
"verdict" is FAIL if any of the four lists above is non-empty, otherwise PASS.
"""


def call_groq(system, user_content):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY environment variable first.")

    models = get_groq_models()
    last_error = None
    for model in models:
        body = json.dumps({
            "model": model,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "AgentAudit/1.0 (+https://console.groq.com)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            last_error = f"{error.code} {error.reason} while calling Groq model {model}: {details}"
            if error.code == 403 and model != models[-1]:
                continue
            raise RuntimeError(last_error) from error

    configured_models = ", ".join(models)
    raise RuntimeError(
        last_error
        or f"Groq request failed unexpectedly. Check GROQ_MODEL/GROQ_MODELS in .env. Tried: {configured_models}."
    )


def run_check(run_data):
    doc_block = "\n\n".join(
        f"--- FILE: {name} ---\n{content}" for name, content in run_data["docs"].items()
    )
    user_content = (
        f"SOURCE DOCUMENTS:\n{doc_block}\n\n"
        f"TASK GIVEN TO AGENT: {run_data['task']}\n\n"
        f"AGENT ANSWER:\n{run_data['agent_answer']}"
    )
    raw = call_groq(CHECKER_SYSTEM_PROMPT, user_content)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1) if cleaned.startswith("json\n") else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Could not parse checker output", "raw": raw}


if __name__ == "__main__":
    try:
        here = os.path.dirname(__file__)
        last_run_path = os.path.join(here, "last_run.json")
        if not os.path.exists(last_run_path):
            print("No last_run.json found. Run agent_runner.py first to generate one.")
            raise SystemExit(1)

        with open(last_run_path) as f:
            run_data = json.load(f)

        verdict = run_check(run_data)
        print("=== AUDIT RESULT ===")
        print(json.dumps(verdict, indent=2))

        log_path = os.path.join(here, "results_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps({"task": run_data["task"], "verdict": verdict}) + "\n")
        print(f"\nAppended to {log_path}")
    except Exception as error:
        print(f"checker.py failed: {error}")
        raise SystemExit(1)