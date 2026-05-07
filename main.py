# main.py
# Orchestrates the full dataset inference pipeline for a given
# (subset, model) combination. Can also be used to run multiple
# combinations in sequence via run_all().
#
# Usage in Colab:
#   from main import run
#   results = run(subset="wikipedia_(en)", model_tag="410m")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import parameters as P
from data.loader      import load_mimir
from models.pythia    import load_model, get_token_losses
from mia.scores       import compute_all_scores
from train            import run_training
from test             import run_ttest, run_false_positive_test, print_summary
from utils.checkpoint import (save_loss_list, load_loss_list,
                               save_features,  load_features,
                               save_results)
from utils.visualization import plot_auc_per_mia, plot_pvalue_heatmap


def run(subset: str = P.SUBSET, model_tag: str = "410m") -> dict:
    """
    Run the full dataset inference pipeline for one (subset, model) pair.

    The pipeline follows the four stages described in Maini et al. (2024):
        Stage 0: Load member (train) and non-member (val) texts.
        Stage 1: Compute per-token losses using the target LLM.
        Stage 2: Extract MIA features; train linear regressor on A-split.
        Stage 3: Assign membership scores on B-split; run t-test.

    Loss lists and feature matrices are checkpointed to Drive after each
    expensive computation. If a checkpoint exists, it is loaded instead of
    recomputing, making session restarts safe.

    Args:
        subset:    Short dataset subset name defined in parameters.py
                   (e.g. 'wikipedia_(en)', 'arxiv', 'github').
        model_tag: Short identifier for the Pythia model size
                   (e.g. '410m', '1.4b', '6.9b', '12b').

    Returns:
        Dictionary containing p-value, t-statistic, mean scores, false
        positive p-value, detection verdict, and per-MIA AUC scores.
    """
    print(f"\n{'=' * 60}")
    print(f"PIPELINE  subset={subset}  model={model_tag}")
    print(f"{'=' * 60}\n")

    # ── Stage 0: Load data ───────────────────────────────────────
    print("[main] Stage 0: Loading data...")
    train_texts, val_texts = load_mimir(subset=subset)

    # ── Stage 1: Load model and compute losses ───────────────────
    print("\n[main] Stage 1: Loading model...")
    model, tokenizer = load_model(P.MODEL_NAME)

    train_loss = load_loss_list("train", subset, model_tag)
    if train_loss is None:
        print("[main] Computing train losses (this may take a while)...")
        train_loss = get_token_losses(model, tokenizer, train_texts)
        save_loss_list(train_loss, "train", subset, model_tag)

    val_loss = load_loss_list("val", subset, model_tag)
    if val_loss is None:
        print("[main] Computing val losses...")
        val_loss = get_token_losses(model, tokenizer, val_texts)
        save_loss_list(val_loss, "val", subset, model_tag)

    print(f"[main] Train: {len(train_loss)} texts | Val: {len(val_loss)} texts")

    # ── Stage 2: Compute MIA features and train regressor ────────
    print("\n[main] Stage 2: Computing MIA features...")

    train_features = load_features("train", subset, model_tag)
    val_features   = load_features("val",   subset, model_tag)

    if train_features is not None and val_features is not None:
        _, feature_names = compute_all_scores(train_loss[:1], train_texts[:1])
    else:
        train_features, feature_names = compute_all_scores(train_loss, train_texts)
        val_features,   _             = compute_all_scores(val_loss,   val_texts)
        save_features(train_features, "train", subset, model_tag)
        save_features(val_features,   "val",   subset, model_tag)

    print(f"[main] Feature matrix shape: {train_features.shape}")

    regressor, member_B, nonmember_B, baseline_aucs = run_training(
        train_features, val_features, feature_names
    )
    plot_auc_per_mia(baseline_aucs, subset, model_tag)

    # ── Stage 3: T-test and false positive check ─────────────────
    print("\n[main] Stage 3: Running t-test...")
    tp_result = run_ttest(regressor, member_B, nonmember_B, subset, model_tag)
    fp_result = run_false_positive_test(regressor, val_features, subset, model_tag)

    results = {**tp_result, **fp_result, "baseline_aucs": baseline_aucs}
    save_results(results, subset, model_tag)
    print_summary(tp_result, fp_result)

    return results


def run_all(subsets: list = ["wikipedia_(en)", "arxiv"],
            model_tags: list = ["410m", "1.4b"]) -> dict:
    """
    Run the pipeline for all combinations of subsets and model sizes.

    Results are collected and displayed as a p-value heatmap (domain x model).

    Args:
        subsets:    List of subset names to evaluate.
        model_tags: List of model size identifiers to evaluate.

    Returns:
        Dictionary mapping (subset, model_tag) tuples to p-values.
    """
    all_results = {}

    for subset in subsets:
        for model_tag in model_tags:
            try:
                result = run(subset=subset, model_tag=model_tag)
                all_results[(subset, model_tag)] = result["p_value"]
            except Exception as e:
                print(f"[main] ERROR: {subset} | {model_tag} — {e}")
                all_results[(subset, model_tag)] = 1.0

    plot_pvalue_heatmap(all_results)
    return all_results


if __name__ == "__main__":
    run(subset="wikipedia_(en)", model_tag="410m")
