# utils/visualization.py
# Plotting functions for dataset inference results.
# All plots are saved to the results directory defined in parameters.py.

import numpy as np
import matplotlib.pyplot as plt
import os
import parameters as P


def plot_auc_per_mia(auc_dict: dict, subset: str, model_tag: str) -> None:
    """
    Plot the AUC score of each individual MIA as a bar chart.

    Under IID conditions, all MIA methods are expected to produce AUC values
    close to 0.5 (random guessing). Bars colored red deviate by more than 0.1
    from 0.5, which may indicate residual distribution shift.

    Args:
        auc_dict:  Dictionary mapping MIA name to its AUC score.
        subset:    Dataset subset name, used for the plot title and filename.
        model_tag: Short model identifier, used for the plot title and filename.
    """
    names  = list(auc_dict.keys())
    values = list(auc_dict.values())

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#E63946" if abs(v - 0.5) > 0.1 else "#3A56A8" for v in values]
    ax.bar(names, values, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC per MIA — {subset} | {model_tag}")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(P.RESULTS_DIR, f"auc_per_mia_{subset}_{model_tag}.png")
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[viz] Saved: {path}")


def plot_score_distribution(member_scores: np.ndarray,
                             nonmember_scores: np.ndarray,
                             subset: str,
                             model_tag: str) -> None:
    """
    Plot the distribution of membership scores for member and non-member texts.

    For a successful dataset inference, the two distributions should be
    clearly separated: member scores should cluster near 0 and non-member
    scores should cluster near 1.

    Args:
        member_scores:    1D array of membership scores for member texts.
        nonmember_scores: 1D array of membership scores for non-member texts.
        subset:           Dataset subset name.
        model_tag:        Short model identifier.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(member_scores,    bins=40, alpha=0.6, color="#3A56A8", label="Member (train)")
    ax.hist(nonmember_scores, bins=40, alpha=0.6, color="#E63946", label="Non-member (val)")
    ax.set_xlabel("Membership Score  (0 = member,  1 = non-member)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {subset} | {model_tag}")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(P.RESULTS_DIR, f"score_dist_{subset}_{model_tag}.png")
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[viz] Saved: {path}")


def plot_pvalue_heatmap(results_table: dict) -> None:
    """
    Plot a heatmap of dataset inference p-values across domains and model sizes.

    Cells with p-value below P_VALUE_THRESHOLD (0.1) are shown in green,
    indicating that the dataset was successfully detected as a training set.
    Cells above the threshold are shown in red.

    Args:
        results_table: Dictionary mapping (subset, model_tag) tuples to p-values.
                       Example: {('wikipedia_(en)', '410m'): 0.003, ...}
    """
    subsets    = sorted(set(k[0] for k in results_table))
    model_tags = sorted(set(k[1] for k in results_table))

    matrix = np.ones((len(subsets), len(model_tags)))
    for i, subset in enumerate(subsets):
        for j, model_tag in enumerate(model_tags):
            if (subset, model_tag) in results_table:
                matrix[i, j] = results_table[(subset, model_tag)]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="RdYlGn_r")
    ax.set_xticks(range(len(model_tags)))
    ax.set_xticklabels(model_tags)
    ax.set_yticks(range(len(subsets)))
    ax.set_yticklabels(subsets)
    ax.set_title(f"Dataset Inference P-values\n(green < {P.P_VALUE_THRESHOLD} = detected)")

    for i in range(len(subsets)):
        for j in range(len(model_tags)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    path = os.path.join(P.RESULTS_DIR, "pvalue_heatmap.png")
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[viz] Saved: {path}")
