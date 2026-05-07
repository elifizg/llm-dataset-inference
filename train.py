# train.py
# Trains the linear regressor on the A-split of member and non-member features.
# Learns which MIA signals are informative for the given data distribution.

import numpy as np
from sklearn.metrics import roc_auc_score
import parameters as P
from models.regressor import MembershipRegressor


def make_ab_split(train_features: np.ndarray,
                  val_features: np.ndarray,
                  seed: int = P.RANDOM_SEED) -> tuple:
    """
    Split member and non-member feature matrices into A and B subsets.

    The A-split is used to train the linear regressor (Stage 2 of the
    dataset inference pipeline). The B-split is held out for the t-test
    (Stage 3), ensuring no data leakage between training and evaluation.

    Args:
        train_features: Feature matrix for member texts, shape (n, n_scores).
        val_features:   Feature matrix for non-member texts, shape (n, n_scores).
        seed:           Random seed for reproducibility.

    Returns:
        member_A:    A-split of member features.
        member_B:    B-split of member features (held out for t-test).
        nonmember_A: A-split of non-member features.
        nonmember_B: B-split of non-member features (held out for t-test).
    """
    n = len(train_features)
    np.random.seed(seed)
    idx   = np.random.permutation(n)
    split = int(n * P.A_SPLIT_RATIO)

    a_idx, b_idx = idx[:split], idx[split:]
    return (
        train_features[a_idx], train_features[b_idx],
        val_features[a_idx],   val_features[b_idx],
    )


def compute_baseline_aucs(train_features: np.ndarray,
                           val_features: np.ndarray,
                           feature_names: list) -> dict:
    """
    Compute AUC for each individual MIA score under IID conditions.

    This reproduces the key finding of Maini et al. (2024): when members and
    non-members come from the same distribution (IID), every individual MIA
    achieves AUC close to 0.5 (random guessing). Any AUC significantly above
    0.5 suggests residual distribution shift in the data.

    Args:
        train_features: Feature matrix for member texts, shape (n, n_scores).
        val_features:   Feature matrix for non-member texts, shape (n, n_scores).
        feature_names:  Names of the MIA scores, one per column.

    Returns:
        Dictionary mapping each MIA name to its AUC score (float).
    """
    y_true = [0] * len(train_features) + [1] * len(val_features)
    aucs   = {}

    for i, name in enumerate(feature_names):
        scores = list(train_features[:, i]) + list(val_features[:, i])
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = 0.5
        aucs[name] = round(auc, 4)

    print("\n[train] Baseline AUC per MIA (IID — all should be close to 0.5):")
    for name, auc in aucs.items():
        flag = "  <- WARNING" if abs(auc - 0.5) > 0.1 else ""
        print(f"  {name:25s}: {auc:.4f}{flag}")

    return aucs


def run_training(train_features: np.ndarray,
                 val_features: np.ndarray,
                 feature_names: list,
                 seed: int = P.RANDOM_SEED) -> tuple:
    """
    Run the full training stage of the dataset inference pipeline.

    Steps:
        1. Compute baseline AUC for each individual MIA (should be ~0.5).
        2. Split features into A and B subsets.
        3. Train the linear regressor on A-split features.
        4. Print the top features selected by the regressor.

    Args:
        train_features: Feature matrix for member texts, shape (n, n_scores).
        val_features:   Feature matrix for non-member texts, shape (n, n_scores).
        feature_names:  Names of the MIA scores, one per column.
        seed:           Random seed for the A/B split.

    Returns:
        regressor:     Fitted MembershipRegressor instance.
        member_B:      B-split member features (passed to test.py).
        nonmember_B:   B-split non-member features (passed to test.py).
        baseline_aucs: Dictionary of per-MIA AUC scores.
    """
    print(f"\n[train] === Training Stage ===")
    print(f"[train] Seed: {seed} | A-split ratio: {P.A_SPLIT_RATIO}")

    baseline_aucs = compute_baseline_aucs(train_features, val_features, feature_names)

    member_A, member_B, nonmember_A, nonmember_B = make_ab_split(
        train_features, val_features, seed=seed
    )
    print(f"\n[train] A-split: {len(member_A)} member + {len(nonmember_A)} non-member")
    print(f"[train] B-split: {len(member_B)} member + {len(nonmember_B)} non-member")

    regressor = MembershipRegressor()
    regressor.fit(member_A, nonmember_A)

    print(f"\n[train] Top 5 most informative MIAs for this domain:")
    for name, weight in regressor.top_features(feature_names, n=5):
        direction = "member->" if weight < 0 else "nonmember->"
        print(f"  {name:25s}: {weight:+.4f}  ({direction})")

    return regressor, member_B, nonmember_B, baseline_aucs
