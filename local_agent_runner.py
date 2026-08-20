"""
local_agent_runner.py
----------------------
Same job as agent_runner.py (send a task + source files to an LLM and get
back an answer) but runs a small model LOCALLY on your own machine via
Hugging Face `transformers`, instead of calling a hosted API.

Why local? The hidden-state probing idea from the MultiHaluDet paper
(arXiv 2605.24919) needs access to the model's imessanternal per-layer
activations while it generates text. Hosted APIs (Groq, Claude, OpenAI,
...) never expose that -- you only ever get the final text back. A local
model is the only way to actually see inside.

Model: Qwen2.5-0.5B-Instruct. Picked because it's small enough to run on
a CPU with no GPU (~1GB download, ~1-2GB RAM while running), so this
costs nothing beyond electricity and a one-time download. It will be
slower and lower quality than the 7B models the paper uses, and it will
hallucinate more -- which for a *study/demo* pipeline like this is
actually useful: more hallucinations to look at.

First run downloads the model from Hugging Face Hub automatically and
caches it (usually under ~/.cache/huggingface). Every run after that is
fully offline.

Requires: `pip install torch transformers` (see README).
Usage:
    python local_agent_runner.py "Summarize each file and give total Q3 revenue."
"""

import os
import sys
import glob
import json

MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
TEST_FILES_DIR = os.path.join(os.path.dirname(__file__), "test_files")
MAX_NEW_TOKENS = 400

# Module-level cache so the model is only loaded once per process (loading
# takes a few seconds; you don't want to pay that cost per-call if this
# gets imported by app.py or run_dual_pipeline.py).
_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    # Imported lazily so that scripts which don't need the local model
    # (e.g. plain stats.py) don't require torch/transformers to be
    # installed at all.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} (first run downloads it, then it's cached)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # plain fp32 for CPU; no GPU assumed
    )
    model.eval()

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    _MODEL, _TOKENIZER = model, tokenizer
    return model, tokenizer


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


def run_local(task, capture_hidden_states=True):
    """
    Runs the task through the local model.

    Returns:
        docs: dict of source documents used
        answer: the generated text
        hidden_states: None, or a list of per-generation-step hidden state
            tuples if capture_hidden_states=True. Each element corresponds
            to one generated token; each tuple has one tensor per layer
            (including the embedding layer), shape (1, 1, hidden_size) for
            the newly generated token at that step.
    """
    import torch

    model, tokenizer = _load_model()
    docs = load_source_documents()
    prompt = build_prompt(task, docs)
    messages = [{"role": "user", "content": prompt}]
    chat_encoding = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = chat_encoding["input_ids"]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # deterministic-ish; keeps runs comparable
            output_hidden_states=capture_hidden_states,
            return_dict_in_generate=capture_hidden_states,
            pad_token_id=tokenizer.eos_token_id,
        )

    if capture_hidden_states:
        generated_ids = output.sequences[0][input_ids.shape[1]:]
        hidden_states = output.hidden_states  # tuple (per step) of tuple (per layer) of tensors
    else:
        generated_ids = output[0][input_ids.shape[1]:]
        hidden_states = None
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return docs, answer, hidden_states


if __name__ == "__main__":
    try:
        task = sys.argv[1] if len(sys.argv) > 1 else "Summarize each file in one paragraph."
        docs, answer, hidden_states = run_local(task, capture_hidden_states=True)

        print("=== LOCAL AGENT ANSWER ===")
        print(answer)

        out = {"task": task, "docs": docs, "agent_answer": answer}
        with open(os.path.join(os.path.dirname(__file__), "last_run_local.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved answer to last_run_local.json")

        # Hidden states are large tensors -- not JSON-serializable and not
        # meant to be kept around as files. hidden_state_probe.py expects to
        # receive them directly from run_local() in the same process (see
        # run_dual_pipeline.py), not read them back from disk.
        if hidden_states is not None:
            n_steps = len(hidden_states)
            n_layers = len(hidden_states[0])
            print(f"Captured hidden states: {n_steps} generated tokens x {n_layers} layers")
    except Exception as error:
        print(f"local_agent_runner.py failed: {error}")
        raise SystemExit(1)
