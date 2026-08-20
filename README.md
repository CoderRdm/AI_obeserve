# AgentAudit

A small tool that watches an AI agent's answer and checks it against real
source documents — catching hallucinated claims, silently-resolved
contradictions, skipped files, and invented info.

## Setup

1. Get a Groq API key from https://console.groq.com/keys
2. Set it as an environment variable:

   **Windows (Command Prompt):**
   ```
   set GROQ_API_KEY=your-key-here
   ```

   **Mac/Linux:**
   ```
   export GROQ_API_KEY=your-key-here
   ```

3. If your account only allows specific Groq models, set one in `.env`:

   ```
   GROQ_MODEL=llama-3.1-8b-instant
   ```

   You can also set a fallback list:

   ```
   GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
   ```

4. Install the only dependency needed (none beyond Python's built-ins — this
   uses `urllib` on purpose so there's nothing extra to install).

## How to run one test

```
python agent_runner.py "Summarize each file and give total Q3 revenue."
python checker.py
```

`agent_runner.py` sends your task + the 5 files in `test_files/` to Groq,
saves the raw answer to `last_run.json`.

`checker.py` reads that file, asks Groq to audit the answer against the
real source documents, prints a verdict, and appends the result to
`results_log.jsonl`.

## How to build up real numbers

Run the two scripts above repeatedly with different task phrasings, e.g.:

```
python agent_runner.py "Summarize each file and give total Q3 revenue."
python checker.py

python agent_runner.py "What were the employee satisfaction survey results?"
python checker.py

python agent_runner.py "What is Q4's projected revenue growth based on these files?"
python checker.py
```

Reword each task a few times, run 15-45 times total, then run:

```
python stats.py
```

This prints your real fail rate and a breakdown of failure types — this is
the number that goes in your resume bullet. Don't use a placeholder number;
this script gives you the actual one in a couple minutes of runs.

## Test files

`test_files/` contains 5 short files with built-in traps:
- Two files with a contradicting Q3 revenue figure
- One file with an explicitly missing data point (HR survey results)
- Two clean files with no traps, as a baseline

Feel free to add your own files/traps as you extend this.

---

## Extension: hidden-state hallucination probing (study addition)

Everything above is a **black-box** detector: `checker.py` asks a second
LLM to *read the answer* and judge it. That's fast and needs zero setup
beyond an API key, but it can only ever be as good as the judge model's
reading comprehension.

There's a different family of approaches that instead looks at a model's
**internal activations while it generates** — the idea explored in
["MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State
Probing"](https://arxiv.org/abs/2605.24919) (arXiv 2605.24919). Their
method probes per-layer hidden states across a full generation, extracts
statistical features from them, and classifies with a large stacked
ensemble trained on ~10,000 human-labeled examples, hitting ~98.5% AUROC
on English hallucination benchmarks with strong cross-lingual transfer.

This repo now includes a **scaled-down, study-purpose version** of that
idea, added as a second, independent detector you can compare against the
existing LLM-judge. It's built to be readable and cheap to run, not to
reproduce the paper's numbers.

### New files

| File | Role |
|---|---|
| `local_agent_runner.py` | Runs the "agent" step on a small **local** model (Qwen2.5-0.5B-Instruct) instead of Groq, capturing hidden states during generation. Local is required here — hosted APIs (Groq, Claude, OpenAI, ...) never expose internal activations, only final text. |
| `hidden_state_probe.py` | Turns raw per-layer hidden states into a flat numeric feature vector (per-layer mean norm, std, kurtosis, median absolute deviation, plus a norm-trajectory slope across layers). |
| `hallucination_classifier.py` | A small 2-model ensemble (logistic regression + gradient boosting) trained to predict hallucination from those features. |
| `run_dual_pipeline.py` | Runs one task through **both** detectors — the LLM-judge and the hidden-state probe — and logs both verdicts side by side. |
| `dual_stats.py` | Prints each detector's fail rate and how often they agree. |
| `app.py` (new tab) | "LLM-judge vs hidden-state probe" tab showing the same comparison in the dashboard. |

### How this differs from the paper (read this before drawing conclusions)

This is explicitly a **simplified, cheap version** of the paper's idea,
not a reproduction of it. The differences matter:

- **Model size**: the paper uses 7B-parameter models (Mistral-7B,
  LLaMA2-7B). This uses Qwen2.5-0.5B (~14x smaller) so it runs on a CPU
  with no GPU. Smaller models have fewer, less expressive layers, so the
  signal in their hidden states is weaker and noisier.
- **Classifier**: the paper trains 5 base models via out-of-fold stacking
  with a learned meta-regressor. This uses 2 base models combined by
  simple averaging — no stacking, no meta-learner. That needs far less
  data to avoid overfitting, at the cost of being a cruder ensemble.
- **Training labels — the big one**: the paper trains on ~10,000
  **human-annotated** hallucination labels (HaluEval) plus more from
  TriviaQA. This repo has no human-labeled dataset, so it bootstraps from
  **weak labels**: every time you run `run_dual_pipeline.py`, the
  existing LLM-judge's PASS/FAIL verdict is used as the label for that
  example. This means the hidden-state probe is learning to *predict the
  judge*, not learning ground-truth hallucination directly. High
  agreement between the two detectors is still a meaningful, real signal
  (it suggests hidden states carry information correlated with the
  judge's reasoning) — but it is not the same claim as the paper's
  "98.5% AUROC against human labels."
- **No multilingual angle**: the paper's other major contribution is
  cross-lingual generalization (English/French/Bangla/Amharic). This repo
  only tests English test files.

### Setup

```
pip install torch transformers scikit-learn scipy numpy
```

No API key needed for this part — everything runs locally. The **first**
run of `local_agent_runner.py` or `run_dual_pipeline.py` downloads
Qwen2.5-0.5B-Instruct from Hugging Face (~1GB) and caches it, usually
under `~/.cache/huggingface`. Every run after that is fully offline and
free.

CPU-only is fine; expect each run to take anywhere from several seconds
to a couple minutes depending on your machine, since there's no GPU
acceleration assumed.

### How to run it

```
python run_dual_pipeline.py "Summarize each file and give total Q3 revenue."
```

This runs the local model, gets the LLM-judge's verdict (still via
Groq/`checker.py` — you still need `GROQ_API_KEY` set for this half),
extracts hidden-state features, logs them as a training example, and (once
there are at least a few examples of each class) trains and queries the
hidden-state probe classifier too.

Run it repeatedly with varied task phrasings — same idea as the original
usage pattern above — to build up both **enough labeled examples for the
probe to train on** and **a real agreement-rate number**:

```
python run_dual_pipeline.py "Summarize each file and give total Q3 revenue."
python run_dual_pipeline.py "What were the employee satisfaction survey results?"
python run_dual_pipeline.py "What is Q4's projected revenue growth based on these files?"
```

Then:

```
python dual_stats.py
```

or open the Streamlit dashboard (`streamlit run app.py`) and check the
"LLM-judge vs hidden-state probe" tab.

### What to actually look at / study

- `hidden_state_probe.py`'s `extract_features()` is the part worth
  reading closely if you want to understand what "probing hidden states"
  concretely means — it's the smallest possible version of the paper's
  Stage 1.
- Compare a FAIL-verdict run's hidden-state feature values against a
  PASS-verdict run's (both get logged in `hidden_state_features_log.jsonl`)
  — do any layers show a consistent difference? That's the empirical
  question the paper is built around.
- Watch the agreement rate in `dual_stats.py` as you accumulate more
  runs. It starting low and creeping up as the probe gets more training
  data is expected and is itself the interesting result.
