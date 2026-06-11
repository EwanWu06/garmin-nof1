"""Shared statistical helpers for the Layer-A recovery estimators.

Both the H-A1 cost estimator (:mod:`garmin_nof1.models.recovery_cost`) and the H-A2
per-sport recovery-τ estimator (:mod:`garmin_nof1.models.recovery_tau`) are within-person
linear models fit with the same weakly-informative conjugate Bayesian machinery — a
closed-form Normal-Inverse-Gamma posterior whose coefficient marginal is multivariate
Student-t. Keeping that machinery in one place keeps the two estimators consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SPORTS = ("triathlon", "soccer")


def deviation(df: pd.DataFrame, outcome: str, deviation_col: str | None, window: int) -> pd.Series:
    """Return the modeled deviation series.

    If ``deviation_col`` is given it is used verbatim (already-detrended, e.g. an oracle
    baseline). Otherwise ``outcome`` is detrended by subtracting a centered rolling mean —
    the pre-registered deviation definition.
    """
    if deviation_col is not None:
        if deviation_col not in df.columns:
            raise KeyError(f"deviation_col {deviation_col!r} not in DataFrame")
        return df[deviation_col].astype(float)
    if outcome not in df.columns:
        raise KeyError(f"outcome column {outcome!r} not in DataFrame")
    raw = df[outcome].astype(float)
    baseline = raw.rolling(window, center=True, min_periods=max(3, window // 2)).mean()
    return raw - baseline


def conjugate_posterior(X: np.ndarray, y: np.ndarray, prior_scale: float):
    """Normal-Inverse-Gamma conjugate posterior for ``y = X b + e``.

    Weakly-informative: zero-mean Gaussian prior on ``b`` with std ``prior_scale``
    (negligible against the data), and a near-flat Inverse-Gamma on the variance.
    Returns ``(mu_n, cov, dof, sigma2_mean)`` where the marginal posterior of ``b`` is
    multivariate Student-t: ``b ~ t_dof(mu_n, cov)`` (``cov`` is the t *scale* matrix).
    """
    n, p = X.shape
    a0 = b0 = 1e-3
    lambda0 = np.eye(p) / prior_scale**2
    lambda_n = X.T @ X + lambda0
    lambda_n_inv = np.linalg.inv(lambda_n)
    mu_n = lambda_n_inv @ (X.T @ y)  # prior mean is zero
    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * float(y @ y - mu_n @ lambda_n @ mu_n)
    dof = 2.0 * a_n
    cov = (b_n / a_n) * lambda_n_inv  # scale matrix of the multivariate-t
    sigma2_mean = b_n / (a_n - 1.0)
    return mu_n, cov, dof, sigma2_mean


def ci(mean: float, scale: float, dof: float, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed Student-t credible interval at the given level."""
    half = float(stats.t.ppf(0.5 + level / 2.0, dof)) * scale
    return (mean - half, mean + half)


def sample_mvt(loc: np.ndarray, scale_cov: np.ndarray, dof: float, n_draws: int, rng):
    """Draw ``n_draws`` samples from the multivariate Student-t ``t_dof(loc, scale_cov)``.

    Used to propagate the joint coefficient posterior through a nonlinear transform
    (here ``tau = -1/ln(phi)``), where a delta-method interval would hide the skew.
    """
    p = len(loc)
    chol = np.linalg.cholesky(scale_cov)
    z = rng.standard_normal((n_draws, p))
    g = rng.chisquare(dof, size=n_draws) / dof
    return loc + (z @ chol.T) / np.sqrt(g)[:, None]
