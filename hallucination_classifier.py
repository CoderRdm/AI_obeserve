"""
hallucination_classifier.py
-----------------------------
Trains a small classifier on the hidden-state features from
hidden_state_probe.py, so the pipeline can predict "hallucinated /
not" directly from a model's internal activations -- no second LLM
judge call needed at prediction time.

What the paper does (full version):
    A big stacked ensemble: gradient-boosted trees, random forest,
    kernel SVM, MLP, and logistic regression as base classifiers, each
    trained via out-of-fold cross-validation to avoid leakage, then a
    meta-regressor learns to combine their outputs. Trained on ~10,000
    human-annotated examples (HaluEval) plus more from TriviaQA.

What this file does (simplified version):
    Two base classifiers (logistic regression + gradient boosting, both
    from scikit-learn) combined by simple averaging of their predicted
    probabilities -- a "poor man's ensemble" with no meta-learner, no
    out-of-fold stacking, no SVM/MLP. This is intentional: those
    techniques need hundreds+ of labeled examples per class to avoid
    just overfitting noise, and this project's dataset is small.

Where do labels come from?
    The paper trains on real human-annotated hallucination labels. We
    don't have that. Instead, this uses WEAK / NOISY labels: every time
    you run run_dual_pipeline.py, the existing LLM-judge (checker.py)
    already produces a PASS/FAIL verdict for that answer. We treat that
    verdict as a (noisy) label and train the hidden-state classifier to
    predict it from the hidden-state features.

    This means the hidden-state classifier's ceiling is bounded by how
    good the LLM-judge is -- it is learning to approximate the judge
    from internal activations, not learning "ground truth" hallucination
    like the paper's human-labeled version does. That's a real
    limitation, and it's explained again in the README. It is still a
    meaningful thing to study: do internal activations carry a signal
    that correlates with the judge's verdict at all? If yes, that's
    evidence hidden states carry hallucination-relevant information,
    which is the paper's central claim, just measured more cheaply.

Persistence: the trained classifier is pickled to
`probe_classifier.pkl` so you don't have to retrain from scratch every
run -- retrain_if_stale() below handles that automatically as your
results log grows.

Requires: scikit-learn, numpy.
"""

import os
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

HERE = os.path.dirname(__file__)
FEATURES_LOG_PATH = os.path.join(HERE, "hidden_state_features_log.jsonl")
MODEL_PATH = os.path.join(HERE, "probe_classifier.pkl")

MIN_EXAMPLES_PER_CLASS = 3  # below this, there's nothing meaningful to fit


def log_labeled_example(features, feature_names, label, task):
    """
    Appends one (features, label) example to the training log. `label`
    is 1 for FAIL (judge flagged something), 0 for PASS. Call this every
    time you have both a feature vector and a judge verdict for the same
    run (see run_dual_pipeline.py).
    """
    entry = {
        "task": task,
        "features": features.tolist(),
        "feature_names": feature_names,
        "label": int(label),
    }
    with open(FEATURES_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_training_data():
    if not os.path.exists(FEATURES_LOG_PATH):
        return None, None

    X, y = [], []
    with open(FEATURES_LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            X.append(entry["features"])
            y.append(entry["label"])

    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    """
    Trains fresh on everything in hidden_state_features_log.jsonl and
    saves the result. Returns None (and prints why) if there isn't
    enough labeled data yet in one or both classes.
    """
    X, y = _load_training_data()
    if X is None or len(X) == 0:
        print("No training examples logged yet -- run run_dual_pipeline.py a few times first.")
        return None

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos < MIN_EXAMPLES_PER_CLASS or n_neg < MIN_EXAMPLES_PER_CLASS:
        print(
            f"Not enough examples per class yet to train "
            f"(need >= {MIN_EXAMPLES_PER_CLASS} each; have {n_pos} FAIL, {n_neg} PASS). "
            f"Keep running the pipeline with varied tasks."
        )
        return None

    logreg = LogisticRegression(max_iter=1000)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=2)

    logreg.fit(X, y)
    gbm.fit(X, y)

    bundle = {"logreg": logreg, "gbm": gbm, "n_examples": len(X)}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Trained probe classifier on {len(X)} examples ({n_pos} FAIL, {n_neg} PASS).")
    return bundle


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(features):
    """
    Returns a dict: {"probability": float 0-1, "verdict": "FAIL"/"PASS",
    "trained_on": int} or None if no trained model exists yet.

    probability is the simple average of the two base classifiers'
    predicted probability of "FAIL" (label=1) -- the "poor man's
    ensemble" mentioned above.
    """
    bundle = _load_model()
    if bundle is None:
        return None

    x = features.reshape(1, -1)
    p_logreg = bundle["logreg"].predict_proba(x)[0][1]
    p_gbm = bundle["gbm"].predict_proba(x)[0][1]
    probability = float((p_logreg + p_gbm) / 2)

    return {
        "probability": probability,
        "verdict": "FAIL" if probability >= 0.5 else "PASS",
        "trained_on": bundle["n_examples"],
    }


def retrain_if_stale(min_new_examples=5):
    """
    Convenience helper: retrains only if the training log has grown by
    at least `min_new_examples` since the currently saved model, or if
    there is no saved model yet. Called from run_dual_pipeline.py after
    every run so the probe classifier gradually improves as you collect
    more runs, without retraining from scratch every single time.
    """
    X, y = _load_training_data()
    if X is None:
        return

    bundle = _load_model()
    already_trained_on = bundle["n_examples"] if bundle else 0

    if len(X) - already_trained_on >= min_new_examples or bundle is None:
        train()


if __name__ == "__main__":
    train()
