# models/regressor.py
# Meta-learner that combines 52 MIA scores into a single membership score.
# Currently uses a Linear Regressor. The MLP extension (Phase 2) will be
# added here with the same interface, so no other files need to change.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import parameters as P


def remove_outliers(features: np.ndarray,
                    percentile: float = P.OUTLIER_PERCENTILE) -> np.ndarray:
    """
    Clip feature values at the given percentile boundaries column-wise.

    For each feature column, values below the lower percentile or above the
    upper percentile are replaced with the corresponding boundary value.
    This prevents extreme outliers from distorting the linear regressor.

    Following Maini et al. (2024), the top and bottom 2.5% are clipped.

    Args:
        features:   Input array of shape (n_samples, n_features).
        percentile: The percentile used for clipping (e.g. 2.5 clips the
                    bottom 2.5% and the top 2.5% of each column).

    Returns:
        Clipped feature array of the same shape as the input.
    """
    result = features.copy().astype(float)
    for col in range(features.shape[1]):
        low  = np.percentile(features[:, col], percentile)
        high = np.percentile(features[:, col], 100 - percentile)
        result[:, col] = np.clip(features[:, col], low, high)
    return result


class MembershipRegressor:
    """
    Domain-adaptive meta-learner for dataset inference.

    Takes 52 MIA feature scores as input and produces a single membership
    score per text. The regressor is trained on the A-split to learn which
    MIA signals are informative for the given data distribution. It then
    assigns scores on the B-split, which are used by the t-test in test.py.

    Score interpretation:
        Score close to 0 -> text is likely a member (seen during training).
        Score close to 1 -> text is likely a non-member.

    Attributes:
        scaler:    StandardScaler fitted on the A-split feature matrix.
        regressor: Fitted LinearRegression model.
        is_fitted: Flag indicating whether fit() has been called.
    """

    def __init__(self):
        """Initialize the regressor with an unfitted scaler and linear model."""
        self.scaler    = StandardScaler()
        self.regressor = LinearRegression()
        self.is_fitted = False

    def fit(self,
            member_features: np.ndarray,
            nonmember_features: np.ndarray) -> None:
        """
        Train the regressor on A-split features.

        Member features are assigned label 0 and non-member features are
        assigned label 1. Outliers are removed before scaling and fitting.

        Args:
            member_features:    Feature matrix for member texts, shape (n, 52).
            nonmember_features: Feature matrix for non-member texts, shape (n, 52).
        """
        member_clean    = remove_outliers(member_features)
        nonmember_clean = remove_outliers(nonmember_features)

        X = np.vstack([member_clean, nonmember_clean])
        y = np.array([0] * len(member_clean) + [1] * len(nonmember_clean))

        X_scaled = self.scaler.fit_transform(X)
        self.regressor.fit(X_scaled, y)
        self.is_fitted = True

        print("[regressor] Training complete.")
        print(f"[regressor] First 5 feature weights: {self.regressor.coef_[:5].round(4)}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Produce membership scores for B-split features.

        Outliers are removed and features are normalized using the scaler
        fitted during training before prediction.

        Args:
            features: Feature matrix of shape (n_samples, n_features).

        Returns:
            Membership scores as a 1D array of shape (n_samples,).
            Values near 0 indicate members; values near 1 indicate non-members.

        Raises:
            AssertionError: If fit() has not been called yet.
        """
        assert self.is_fitted, "Call fit() before predict()."
        features_clean = remove_outliers(features)
        X_scaled = self.scaler.transform(features_clean)
        return self.regressor.predict(X_scaled)

    def top_features(self, feature_names: list, n: int = 10) -> list:
        """
        Return the n most important features ranked by absolute weight.

        Args:
            feature_names: List of feature names corresponding to model columns.
            n:             Number of top features to return.

        Returns:
            List of (feature_name, weight) tuples sorted by |weight| descending.

        Raises:
            AssertionError: If fit() has not been called yet.
        """
        assert self.is_fitted, "Call fit() before top_features()."
        pairs = sorted(
            zip(feature_names, self.regressor.coef_),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return pairs[:n]
