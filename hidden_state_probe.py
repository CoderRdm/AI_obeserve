"""
hidden_state_probe.py
-----------------------
Turns raw per-token, per-layer hidden states (as produced by
local_agent_runner.run_local) into a fixed-size numeric feature vector,
the way MultiHaluDet (arXiv 2605.24919) does in its "LLM Feature
Extraction" stage -- simplified a lot to fit a small model and a small
project.

What the paper does (full version):
    - Samples every model down to K=32 uniform layer indices (so a 32,
      48, or 80-layer model all produce comparable feature vectors).
    - Per layer, computes distributional statistics of the hidden state
      vectors across the generated sequence: norms, kurtosis, median
      absolute deviation.
    - Also pulls global features: logit statistics, norm trajectories
      across layers.
    - Feeds all of that into a multi-scale attention transformer encoder
      before classification.

What this file does (simplified version, same spirit):
    - Qwen2.5-0.5B only has 24 transformer layers (+ 1 embedding layer),
      so instead of sampling down to a fixed K we just use every layer
      directly -- no need for dynamic sampling at this scale, but the
      function is written so you can raise K later if you swap in a
      bigger/deeper model.
    - Per layer, per generated token, we already have a hidden vector.
      We compute simple statistics across the token dimension (mean
      norm, std of norms, kurtosis, median absolute deviation) -- same
      four statistics as the paper, no fancy attention encoder, just a
      flat feature vector.
    - No transformer encoder / stacked ensemble on top -- that lives in
      hallucination_classifier.py, using plain scikit-learn instead of a
      learned attention layer over these features (would need far more
      training data than a personal project can gather).

This is "the same idea, scaled down" -- not a reimplementation of the
whole paper. The README explains the simplifications explicitly.

Requires: numpy, scipy, torch (for the tensors coming in).
"""

import numpy as np
from scipy.stats import kurtosis


def dynamic_layer_sample(num_layers, k=None):
    """
    Returns k evenly-spaced layer indices out of num_layers, so models of
    different depths produce feature vectors of the same length.

    If k is None or k >= num_layers, just returns every layer index
    (this is the common case for a 24-layer model like Qwen2.5-0.5B --
    there's nothing to downsample).
    """
    if k is None or k >= num_layers:
        return list(range(num_layers))
    return list(np.linspace(0, num_layers - 1, num=k, dtype=int))


def _layer_stats(vectors):
    """
    vectors: numpy array of shape (num_tokens, hidden_size) -- one hidden
    state vector per generated token, for a single layer.

    Returns four scalars: mean norm, std of norms, kurtosis of norms,
    median absolute deviation of norms. These mirror the "norms,
    kurtosis, median absolute deviation" statistics the paper computes
    per layer.
    """
    norms = np.linalg.norm(vectors, axis=1)  # one norm per token

    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    # kurtosis needs at least a few points to mean anything; short
    # answers (few generated tokens) can make this noisy -- that's a
    # known limitation, not a bug.
    kurt = float(kurtosis(norms)) if len(norms) >= 4 else 0.0
    mad = float(np.median(np.abs(norms - np.median(norms))))

    return [mean_norm, std_norm, kurt, mad]


def extract_features(hidden_states, k_layers=None):
    """
    hidden_states: the `hidden_states` object returned by
        local_agent_runner.run_local(..., capture_hidden_states=True).
        Structure: tuple (one entry per generated token) of tuple (one
        entry per layer, including the embedding layer) of tensors
        shaped (1, seq_len_at_that_step, hidden_size).

        Note: HF's generate() gives you the *prompt* hidden states on the
        first generation step and single-token hidden states on every
        step after. We only use the last token's vector at each step
        (the newly generated one) so every step contributes exactly one
        vector per layer, keeping things simple and consistent.

    k_layers: how many layers to sample down to (see dynamic_layer_sample).
        Default None = use every layer.

    Returns: a flat 1D numpy array of features, and a list of human
    readable names for each entry (for debugging / interpretability).
    """
    if not hidden_states or len(hidden_states) == 0:
        raise ValueError("No hidden states to extract features from (empty generation?)")

    num_layers = len(hidden_states[0])
    layer_indices = dynamic_layer_sample(num_layers, k=k_layers)

    # Collect, per layer, the vector for the last (newly generated) token
    # at every generation step.
    per_layer_vectors = {layer_idx: [] for layer_idx in layer_indices}
    for step_hidden_states in hidden_states:
        for layer_idx in layer_indices:
            layer_tensor = step_hidden_states[layer_idx]  # (1, seq_len, hidden)
            last_token_vec = layer_tensor[0, -1, :].detach().cpu().numpy()
            per_layer_vectors[layer_idx].append(last_token_vec)

    features = []
    feature_names = []
    for layer_idx in layer_indices:
        vectors = np.stack(per_layer_vectors[layer_idx], axis=0)  # (num_tokens, hidden)
        stats = _layer_stats(vectors)
        features.extend(stats)
        feature_names.extend([
            f"layer{layer_idx}_mean_norm",
            f"layer{layer_idx}_std_norm",
            f"layer{layer_idx}_kurtosis",
            f"layer{layer_idx}_mad",
        ])

    # Global features across all sampled layers: how does the mean norm
    # trend from early to late layers? The paper calls this a "norm
    # trajectory" -- we approximate it with the slope of a linear fit.
    mean_norms_by_layer = [features[i * 4] for i in range(len(layer_indices))]
    if len(mean_norms_by_layer) >= 2:
        slope = float(np.polyfit(range(len(mean_norms_by_layer)), mean_norms_by_layer, 1)[0])
    else:
        slope = 0.0
    features.append(slope)
    feature_names.append("norm_trajectory_slope")

    return np.array(features, dtype=np.float32), feature_names


if __name__ == "__main__":
    print(
        "This module is a library, not meant to run standalone.\n"
        "Use run_dual_pipeline.py to see it in action end-to-end."
    )
