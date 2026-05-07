# mia/scores.py
# Computes MIA (Membership Inference Attack) scores from per-token loss lists.
# Each function takes loss_list (output of models/pythia.py) as input and
# returns a list of scalar scores, one per text.

import zlib
import numpy as np
import parameters as P


# ── Group A: Thresholding-based ───────────────────────────────────────────────

def compute_perplexity(loss_list: list) -> list:
    """
    Compute perplexity for each text.

    Perplexity is defined as exp(mean token loss). Lower perplexity indicates
    the model finds the text more familiar, which suggests membership.

    Args:
        loss_list: List of per-token loss lists, one list per text.

    Returns:
        List of perplexity values, one per text.
    """
    import math
    return [math.exp(sum(entry) / len(entry)) for entry in loss_list]


def compute_zlib_ratio(loss_list: list, texts: list) -> list:
    """
    Compute the ratio of mean token loss to zlib compression entropy.

    The zlib entropy captures the compressibility of the raw text, providing
    a text-complexity baseline. The ratio normalizes the model loss by this
    baseline, reducing false positives on highly predictable text.

    Args:
        loss_list: List of per-token loss lists, one list per text.
        texts:     List of raw input strings corresponding to loss_list.

    Returns:
        List of loss-to-zlib ratios, one per text.
    """
    ratios = []
    for entry, text in zip(loss_list, texts):
        mean_loss    = sum(entry) / len(entry)
        zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
        ratios.append(mean_loss / zlib_entropy)
    return ratios


# ── Groups B & C: Min-k% and Max-k% ──────────────────────────────────────────

def compute_k_prob(loss_list: list, k: float, reverse: bool = False) -> list:
    """
    Compute the mean loss of the k% of tokens with the lowest (or highest) loss.

    Min-k% Prob (Shi et al., ICLR 2024): focuses on the tokens with the
    lowest probability (highest loss), filtering out trivially predictable
    tokens that inflate naive perplexity estimates.

    Max-k%: focuses on the tokens with the highest probability (lowest loss),
    capturing the tokens the model is most confident about.

    Args:
        loss_list: List of per-token loss lists, one list per text.
        k:         Fraction of tokens to select (e.g. 0.1 = bottom 10%).
        reverse:   If True, selects the top-k% (highest loss) instead.

    Returns:
        List of mean k%-loss values, one per text.
    """
    results = []
    for entry in loss_list:
        sorted_entry = sorted(entry, reverse=reverse)
        n = max(1, int(len(sorted_entry) * k))
        results.append(sum(sorted_entry[:n]) / n)
    return results


# ── Main function ─────────────────────────────────────────────────────────────

def compute_all_scores(loss_list: list,
                       texts: list) -> tuple:
    """
    Compute all enabled MIA scores and return a feature matrix.

    The set of computed scores is controlled by the flags in parameters.py
    (COMPUTE_PERPLEXITY, COMPUTE_ZLIB, COMPUTE_MIN_K, COMPUTE_MAX_K).
    For Phase 1 this produces 16 scores: 2 thresholding + 7 min-k + 7 max-k.
    Perturbation-based and reference-model-based scores are added in Phase 2.

    Args:
        loss_list: List of per-token loss lists produced by get_token_losses().
        texts:     List of raw input strings corresponding to loss_list.

    Returns:
        features:      np.ndarray of shape (n_samples, n_scores).
        feature_names: List of score names corresponding to each column.
    """
    scores: dict[str, list] = {}

    # Group A — Thresholding (2 scores)
    if P.COMPUTE_PERPLEXITY:
        scores["perplexity"] = compute_perplexity(loss_list)

    if P.COMPUTE_ZLIB:
        scores["zlib_ratio"] = compute_zlib_ratio(loss_list, texts)

    # Group B — Min-k% (7 scores)
    if P.COMPUTE_MIN_K:
        for k in P.K_VALUES:
            name = f"min_{int(k * 100)}pct"
            scores[name] = compute_k_prob(loss_list, k=k, reverse=False)

    # Group C — Max-k% (7 scores)
    if P.COMPUTE_MAX_K:
        for k in P.K_VALUES:
            name = f"max_{int(k * 100)}pct"
            scores[name] = compute_k_prob(loss_list, k=k, reverse=True)

    feature_names = list(scores.keys())
    features      = np.column_stack([scores[n] for n in feature_names])

    print(f"[scores] Computed {features.shape[1]} MIA scores: {feature_names}")
    return features, feature_names
