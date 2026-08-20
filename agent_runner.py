# """
# agent_runner.py
# ----------------
# Simulates the "agent" step: sends a task prompt + a set of source files to
# Claude, and returns the raw answer. This is the thing we are going to audit.

# Requires: ANTHROPIC_API_KEY environment variable set.
# Usage:
#     python agent_runner.py "Summarize each file and give total Q3 revenue."
# """

# import os
# import sys
# import glob
# import json
# import urllib.request

# API_URL = "https://api.anthropic.com/v1/messages"
# MODEL = "claude-sonnet-4-6"
# TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")


# def load_source_documents():
#     docs = {}
#     for path in sorted(glob.glob(os.path.join(TEST_FILES_DIR, "*.txt"))):
#         with open(path, "r", encoding="utf-8") as f:
#             docs[os.path.basename(path)] = f.read()
#     return docs


# def build_prompt(task, docs):
#     doc_block = "\n\n".join(
#         f"--- FILE: {name} ---\n{content}" for name, content in docs.items()
#     )
#     return f"{doc_block}\n\n---\n\nTASK: {task}"


# def call_claude(prompt):
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("Set ANTHROPIC_API_KEY environment variable first.")

#     body = json.dumps({
#         "model": MODEL,
#         "max_tokens": 1000,
#         "messages": [{"role": "user", "content": prompt}],
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


# def run(task):
#     docs = load_source_documents()
#     prompt = build_prompt(task, docs)
#     answer = call_claude(prompt)
#     return docs, answer


# if __name__ == "__main__":
#     task = sys.argv[1] if len(sys.argv) > 1 else "Summarize each file in one paragraph."
#     docs, answer = run(task)
#     print("=== AGENT ANSWER ===")
#     print(answer)

#     # Save for the checker to use
#     out = {"task": task, "docs": docs, "agent_answer": answer}
#     with open(os.path.join(os.path.dirname(__file__), "last_run.json"), "w") as f:
#         json.dump(out, f, indent=2)
#     print("\nSaved to last_run.json")
"""
agent_runner.py (Groq version)
--------------------------------
Simulates the "agent" step: sends a task prompt + a set of source files to
an LLM via Groq's free API, and returns the raw answer. This is the thing
we are going to audit.

Requires: GROQ_API_KEY environment variable set (or a .env file with it).
Get a free key at https://console.groq.com/keys

Usage:
    python agent_runner.py "Summarize each file and give total Q3 revenue."
"""

import os
import sys
import glob
import json
import urllib.request
import urllib.error

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")


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


def load_source_documents():
    docs = {}
    for path in sorted(glob.glob(os.path.join(TEST_FILES_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


def build_prompt(task, docs):
    doc_block = "\n\n".join(
        f"--- FILE: {name} ---\n{content}" for name, content in docs.items()
    )
    return f"{doc_block}\n\n---\n\nTASK: {task}"


def call_groq(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY environment variable first.")

    models = get_groq_models()
    last_error = None
    for model in models:
        body = json.dumps({
            "model": model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
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


def run(task):
    docs = load_source_documents()
    prompt = build_prompt(task, docs)
    answer = call_groq(prompt)
    return docs, answer


if __name__ == "__main__":
    try:
        task = sys.argv[1] if len(sys.argv) > 1 else "Summarize each file in one paragraph."
        docs, answer = run(task)
        print("=== AGENT ANSWER ===")
        print(answer)

        out = {"task": task, "docs": docs, "agent_answer": answer}
        with open(os.path.join(os.path.dirname(__file__), "last_run.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved to last_run.json")
    except Exception as error:
        print(f"agent_runner.py failed: {error}")
        raise SystemExit(1)