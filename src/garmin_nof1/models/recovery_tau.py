"""Layer A (headline, H-A2): the per-sport recovery time-constant estimator.

What this answers
-----------------
Within *this one person*, does the HRV deviation left by a session **decay back to
baseline at a different rate** after soccer than after triathlon? The headline quantity
is the difference in recovery time-constants ``τ_soccer − τ_triathlon`` (OSF
preregistration H-A2: supported iff its 95% credible interval excludes 0).

The model
---------
A session leaves a deviation ``dev[t] = ln_rmssd[t] − baseline[t]`` that then mean-reverts.
The *speed* of that reversion is the recovery rate. We let the autoregressive persistence
depend on the **recovery regime** — the sport of the most recent session — and fit one
within-person regression::

    dev[t] = c + phi_triathlon * 1[regime[t]==triathlon] * dev[t-1]
               + phi_soccer    * 1[regime[t]==soccer]    * dev[t-1]
               - cost_tri * tri_load[t] - cost_soc * soc_load[t] + eps[t]

where ``regime[t]`` is the sport of the most recent session *strictly before* day *t*
(forward-filled, lag-1). The two interacted lag coefficients are the sport-specific
persistences ``phi_s``; the recovery time-constant is ``τ_s = -1/ln(phi_s)``. This mirrors
the synthetic data-generating process (:mod:`garmin_nof1.data.synthetic`), where a
deviation decays at the most-recent-session's ``phi`` until the next session — so the
estimator can be validated against a planted ``τ_soccer ≠ τ_triathlon``.

Inference is the same weakly-informative conjugate Bayesian fit as the H-A1 cost model
(:mod:`garmin_nof1.models._common`). Because ``τ = -1/ln(phi)`` is nonlinear (and steep as
``phi → 1``), the posterior of the τ difference is propagated by **Monte-Carlo draws from
the joint coefficient posterior** (a multivariate Student-t) rather than a delta-method
interval, so the reported credible interval keeps the real skew. Draws are taken with a
fixed seed, so the result is deterministic. Draws falling outside ``phi ∈ (0, 1)`` (no
physical recovery time-constant) are discarded and counted in ``n_draws_used``.

Calibration (verified on the synthetic substrate)
-------------------------------------------------
With a planted gap (``phi_triathlon=0.70``, ``phi_soccer=0.88`` → τ ≈ 2.8 vs 7.8 days),
H-A2 is detected with the correct sign in 100% of 30 seeds. Under a single global φ (no
gap) the estimator is unbiased — mean ``tau_diff ≈ 0`` — and its H-A2 type-I error sits
near the nominal 5% (≈6% over 100 seeds): it does not manufacture a recovery-speed
difference. Because soccer regimes are sparse, ``τ_soccer`` is the less precise estimate.

Honest scope notes
------------------
* This is the H-A2 estimator. H-A1 (the per-unit-load *cost* and its sport interaction)
  is the separate :func:`garmin_nof1.models.fit_recovery_cost`.
* Soccer recovery regimes are sparser than triathlon (soccer is the rarer session), so
  ``τ_soccer`` is estimated from fewer days and carries a wider interval — reported via
  ``tau_ci`` and the regime counts ``n_regime``.
* The centered-rolling-mean detrend (when ``deviation_col`` is None) is the pre-registered
  in-sample deviation definition; it is not used by the leakage-controlled prediction layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from garmin_nof1.models._common import SPORTS, conjugate_posterior, deviation, sample_mvt


@dataclass(frozen=True)
class RecoveryTauFit:
    """Posterior summary of the per-sport recovery-τ model.

    Attributes
    ----------
    tau : dict[str, float]
        Posterior-mean recovery time-constant per sport (days).
    tau_ci : dict[str, tuple[float, float]]
        Credible interval for each ``tau`` at the reporting ``ci_level``.
    tau_diff : float
        Posterior-mean ``tau['soccer'] - tau['triathlon']`` — the headline (H-A2).
    tau_diff_ci : tuple[float, float]
        Credible interval for ``tau_diff`` at the reporting ``ci_level``.
    prob_tau_soccer_longer : float
        Posterior ``P(tau_soccer > tau_triathlon)``.
    h_a2_supported : bool
        Pre-registered H-A2 verdict: the **95%** credible interval of ``tau_diff``
        excludes 0 (OSF §6), regardless of the reporting ``ci_level``.
    phi : dict[str, float]
        Posterior-mean per-sport AR(1) recovery persistence.
    n_regime : dict[str, int]
        Day-rows fit under each sport's recovery regime.
    n_obs : int
        Total day-rows used in the regression.
    n_draws_used : int
        Posterior draws with ``phi`` in (0, 1) for both sports (used for the τ summaries).
    """

    tau: dict[str, float]
    tau_ci: dict[str, tuple[float, float]]
    tau_diff: float
    tau_diff_ci: tuple[float, float]
    prob_tau_soccer_longer: float
    h_a2_supported: bool
    phi: dict[str, float]
    n_regime: dict[str, int]
    n_obs: int
    n_draws_used: int


def fit_recovery_tau(
    df: pd.DataFrame,
    *,
    outcome: str = "ln_rmssd",
    deviation_col: str | None = None,
    detrend_window: int = 28,
    prior_scale: float = 10.0,
    ci_level: float = 0.95,
    n_draws: int = 20_000,
    seed: int = 0,
) -> RecoveryTauFit:
    """Estimate the per-sport recovery time-constant τ and the τ difference (H-A2).

    Parameters
    ----------
    df : pandas.DataFrame
        Daily panel with ``sport`` (incl. ``triathlon`` / ``soccer`` / ``rest``),
        ``trimp`` and the outcome column, on contiguous calendar days.
    outcome, deviation_col, detrend_window, prior_scale, ci_level :
        As in :func:`garmin_nof1.models.fit_recovery_cost`. The H-A2 decision is always
        evaluated at 95% per the preregistration.
    n_draws, seed :
        Monte-Carlo draws (and RNG seed for determinism) used to propagate the joint
        coefficient posterior to the nonlinear ``τ = -1/ln(phi)``.
    """
    work = df.copy()
    dev = deviation(work, outcome, deviation_col, detrend_window)
    sport = work["sport"].astype(object)
    sport_arr = np.asarray(sport)
    trimp = work["trimp"].astype(float).to_numpy()

    # Recovery regime = sport of the most recent session strictly before each day.
    session = sport.where(sport != "rest")
    regime = session.ffill().shift(1).to_numpy()
    dev_lag = dev.shift(1).to_numpy()

    work = work.assign(
        _dev=dev.to_numpy(),
        _dev_lag=dev_lag,
        _regime=regime,
        _ar_tri=np.where(regime == "triathlon", dev_lag, 0.0),
        _ar_soc=np.where(regime == "soccer", dev_lag, 0.0),
        _tri_load=np.where(sport_arr == "triathlon", trimp / 100.0, 0.0),
        _soc_load=np.where(sport_arr == "soccer", trimp / 100.0, 0.0),
    )

    cols = ["_dev", "_dev_lag", "_regime", "_tri_load", "_soc_load"]
    fit_df = work.dropna(subset=cols)
    if len(fit_df) < 30:
        raise ValueError("too few aligned observations to fit the per-sport recovery model")

    y = fit_df["_dev"].to_numpy(float)
    # columns: 0 intercept, 1 phi_triathlon, 2 phi_soccer, 3 triathlon load, 4 soccer load
    X = np.column_stack(
        [
            np.ones(len(fit_df)),
            fit_df["_ar_tri"].to_numpy(float),
            fit_df["_ar_soc"].to_numpy(float),
            fit_df["_tri_load"].to_numpy(float),
            fit_df["_soc_load"].to_numpy(float),
        ]
    )
    mu_n, cov_m, dof, _ = conjugate_posterior(X, y, prior_scale)
    phi_col = {"triathlon": 1, "soccer": 2}
    phi_mean = {s: float(mu_n[phi_col[s]]) for s in SPORTS}

    # Propagate the joint (phi_tri, phi_soc) posterior to tau via Monte-Carlo.
    rng = np.random.default_rng(seed)
    draws = sample_mvt(mu_n, cov_m, dof, n_draws, rng)
    phi_t = draws[:, phi_col["triathlon"]]
    phi_s = draws[:, phi_col["soccer"]]
    ok = (phi_t > 0.0) & (phi_t < 1.0) & (phi_s > 0.0) & (phi_s < 1.0)
    tau_t = -1.0 / np.log(phi_t[ok])
    tau_s = -1.0 / np.log(phi_s[ok])
    tau_diff_draws = tau_s - tau_t

    lo_q, hi_q = (1.0 - ci_level) / 2.0, (1.0 + ci_level) / 2.0
    tau = {"triathlon": float(np.mean(tau_t)), "soccer": float(np.mean(tau_s))}
    tau_ci = {
        "triathlon": (float(np.quantile(tau_t, lo_q)), float(np.quantile(tau_t, hi_q))),
        "soccer": (float(np.quantile(tau_s, lo_q)), float(np.quantile(tau_s, hi_q))),
    }
    tau_diff = float(np.mean(tau_diff_draws))
    tau_diff_ci = (
        float(np.quantile(tau_diff_draws, lo_q)),
        float(np.quantile(tau_diff_draws, hi_q)),
    )
    prob_longer = float(np.mean(tau_diff_draws > 0.0))
    # Pre-registered H-A2 decision (OSF §6): 95% CrI of the difference excludes 0.
    d_lo, d_hi = np.quantile(tau_diff_draws, [0.025, 0.975])
    h_a2_supported = bool(d_lo > 0.0 or d_hi < 0.0)

    regime_used = fit_df["_regime"]
    n_regime = {s: int((regime_used == s).sum()) for s in SPORTS}

    return RecoveryTauFit(
        tau=tau,
        tau_ci=tau_ci,
        tau_diff=tau_diff,
        tau_diff_ci=tau_diff_ci,
        prob_tau_soccer_longer=prob_longer,
        h_a2_supported=h_a2_supported,
        phi=phi_mean,
        n_regime=n_regime,
        n_obs=len(fit_df),
        n_draws_used=int(ok.sum()),
    )
