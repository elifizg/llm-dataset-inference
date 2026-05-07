# test.py
# Evaluates the trained regressor on the B-split using a two-sample t-test.
# Also runs a false positive test by comparing two non-member groups.

import numpy as np
from scipy import stats
import parameters as P
from models.regressor import MembershipRegressor
from utils.visualization import plot_score_distribution


def run_ttest(regressor: MembershipRegressor,
              member_B: np.ndarray,
              nonmember_B: np.ndarray,
              subset: str,
              model_tag: str) -> dict:
    """
    Run a one-sided two-sample t-test on B-split membership scores.

    The null hypothesis H0 is that the mean membership score of member texts
    equals that of non-member texts. The alternative hypothesis H1 is that
    member scores are strictly lower (closer to 0) than non-member scores.

    A p-value below P_VALUE_THRESHOLD (0.1) leads to rejecting H0, providing
    statistical evidence that the suspect dataset was used for training.

    Args:
        regressor:    Fitted MembershipRegressor from train.py.
        member_B:     B-split feature matrix for member texts.
        nonmember_B:  B-split feature matrix for non-member texts.
        subset:       Dataset subset name, used for logging and visualization.
        model_tag:    Short model identifier, used for logging and visualization.

    Returns:
        Dictionary containing t-statistic, p-value, mean scores, detection
        verdict, and a human-readable summary string.
    """
    print(f"\n[test] === T-Test: {subset} | {model_tag} ===")

    member_scores    = regressor.predict(member_B)
    nonmember_scores = regressor.predict(nonmember_B)

    print(f"[test] Member scores     — mean: {member_scores.mean():.4f} ± {member_scores.std():.4f}")
    print(f"[test] Non-member scores — mean: {nonmember_scores.mean():.4f} ± {nonmember_scores.std():.4f}")

    t_stat, p_value = stats.ttest_ind(
        member_scores,
        nonmember_scores,
        alternative="less"
    )

    print(f"\n[test] T-statistic : {t_stat:.4f}")
    print(f"[test] P-value     : {p_value:.6f}")

    if p_value < P.P_VALUE_THRESHOLD:
        verdict = f"DETECTED — dataset was in training (p={p_value:.4f} < {P.P_VALUE_THRESHOLD})"
    else:
        verdict = f"NOT DETECTED — cannot confirm (p={p_value:.4f} >= {P.P_VALUE_THRESHOLD})"
    print(f"[test] Result: {verdict}")

    plot_score_distribution(member_scores, nonmember_scores, subset, model_tag)

    return {
        "subset":         subset,
        "model":          model_tag,
        "t_stat":         round(float(t_stat), 4),
        "p_value":        round(float(p_value), 6),
        "member_mean":    round(float(member_scores.mean()), 4),
        "nonmember_mean": round(float(nonmember_scores.mean()), 4),
        "detected":       bool(p_value < P.P_VALUE_THRESHOLD),
        "verdict":        verdict,
    }


def run_false_positive_test(regressor: MembershipRegressor,
                             val_features: np.ndarray,
                             subset: str,
                             model_tag: str) -> dict:
    """
    Run a false positive check by comparing two non-member groups.

    Both halves of the validation set are non-members, so the regressor
    should not be able to distinguish them. The resulting p-value should
    be large (above FP_P_VALUE_MIN = 0.5), confirming that the method
    does not produce false positives under this condition.

    Args:
        regressor:    Fitted MembershipRegressor from train.py.
        val_features: Full feature matrix for non-member texts.
        subset:       Dataset subset name.
        model_tag:    Short model identifier.

    Returns:
        Dictionary containing the false positive p-value and a pass/fail flag.
    """
    print(f"\n[test] === False Positive Test: {subset} | {model_tag} ===")

    n      = len(val_features)
    half   = n // 2
    val_A  = val_features[:half]
    val_B  = val_features[half: 2 * half]

    scores_A = regressor.predict(val_A)
    scores_B = regressor.predict(val_B)

    _, p_fp = stats.ttest_ind(scores_A, scores_B, alternative="less")

    print(f"[test] False positive p-value: {p_fp:.4f}")

    if p_fp > P.FP_P_VALUE_MIN:
        print(f"[test] OK — no false positives (p={p_fp:.4f} > {P.FP_P_VALUE_MIN})")
    else:
        print(f"[test] WARNING — false positive risk (p={p_fp:.4f} <= {P.FP_P_VALUE_MIN})")

    return {
        "fp_p_value": round(float(p_fp), 4),
        "fp_ok":      bool(p_fp > P.FP_P_VALUE_MIN),
    }


def print_summary(tp_result: dict, fp_result: dict) -> None:
    """
    Print a concise summary table of the dataset inference results.

    Args:
        tp_result: Output of run_ttest().
        fp_result: Output of run_false_positive_test().
    """
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Domain        : {tp_result['subset']}")
    print(f"  Model         : {tp_result['model']}")
    print(f"  P-value       : {tp_result['p_value']:.6f}  ->  {'DETECTED' if tp_result['detected'] else 'NOT DETECTED'}")
    print(f"  FP p-value    : {fp_result['fp_p_value']:.4f}   ->  {'OK (no FP)' if fp_result['fp_ok'] else 'WARNING (FP risk)'}")
    print("=" * 55)
