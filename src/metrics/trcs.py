"""
Token Reconstruction Confidence Score (TRCS)
TokenLeak Framework -- Novel Metric Implementation

TRCS(tau, O) = sum(w_i * phi_i(tau, O_i)) / sum(w_i)

phi_1 (timing): Gaussian likelihood  -- N(mu_tau, sigma_tau)
phi_2 (cache):  Cosine similarity    -- cos(c_j, c_tau)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class TRCS:
    """Token Reconstruction Confidence Score."""

    def __init__(self, w1: float = 1.0, w2: float = 1.0):
        self.w = np.array([w1, w2], dtype=np.float64)

    # ------------------------------------------------------------------ #
    def phi_timing(self, t_obs: float, mu: float, sigma: float) -> float:
        """Gaussian likelihood feature (Channel 1)."""
        if sigma <= 0:
            return 0.0
        return float(np.exp(-0.5 * ((t_obs - mu) / sigma) ** 2))

    def phi_cache(self, c_obs: np.ndarray, c_ref: np.ndarray) -> float:
        """Cosine similarity feature (Channel 2)."""
        norm_obs = np.linalg.norm(c_obs)
        norm_ref = np.linalg.norm(c_ref)
        if norm_obs < 1e-9 or norm_ref < 1e-9:
            return 0.0
        return float(np.dot(c_obs, c_ref) / (norm_obs * norm_ref))

    def score(self, t_obs: float, c_obs: np.ndarray,
              mu_tau: float, sigma_tau: float,
              c_tau: np.ndarray) -> float:
        """Compute TRCS for a single candidate token tau."""
        phi = np.array([
            self.phi_timing(t_obs, mu_tau, sigma_tau),
            self.phi_cache(c_obs, c_tau),
        ])
        return float(np.dot(self.w, phi) / self.w.sum())

    # ------------------------------------------------------------------ #
    def fit_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Learn weights w_1, w_2 via logistic regression on profiling data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            [phi_timing, phi_cache] features for each (position, token) pair.
        y : ndarray of shape (n_samples,)
            Binary label: 1 if this token is the true token at that position.
        """
        clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=500,
                                 random_state=42)
        clf.fit(X, y)
        raw_w = np.maximum(clf.coef_[0], 0.0)
        self.w = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(2) / 2

    def rank(self, t_obs: float, c_obs: np.ndarray,
             profile: dict) -> list:
        """
        Return vocabulary tokens ranked by TRCS (descending).

        Parameters
        ----------
        profile : dict mapping token_id -> {'mu': float, 'sigma': float,
                                             'cache': np.ndarray}
        """
        scores = {
            tok: self.score(t_obs, c_obs,
                            v['mu'], v['sigma'], v['cache'])
            for tok, v in profile.items()
        }
        return sorted(scores, key=scores.__getitem__, reverse=True)
