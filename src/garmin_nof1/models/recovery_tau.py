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

from garmin_nof1.models._common import (
    conjugate_posterior,
    deviation,
    modeled_sports,
    sample_mvt,
)


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
    load_lag: int = 0,
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
    load_lag : int
        Days by which a session is shifted forward onto the HRV night it first affects (as in
        :func:`garmin_nof1.models.fit_recovery_cost`). ``0`` (default) is the synthetic-DGP
        convention; real Garmin HRV is morning-timestamped, so use ``load_lag=1`` — the load
        terms shift by one day and the recovery regime by one further day (the deviation a
        session leaves appears one night later and only then begins to decay).
    """
    if load_lag < 0:
        raise ValueError("load_lag must be >= 0")
    work = df.copy()
    dev = deviation(work, outcome, deviation_col, detrend_window)
    sport = work["sport"].astype(object)
    sport_arr = np.asarray(sport)
    trimp = work["trimp"].astype(float).to_numpy()

    # Recovery regime = sport of the session whose deviation is currently decaying: the most
    # recent session strictly before each day, shifted a further ``load_lag`` days because a
    # session's deviation only appears (and starts decaying) ``load_lag`` nights later.
    session = sport.where(sport != "rest")
    regime = session.ffill().shift(1 + load_lag).to_numpy()
    dev_lag = dev.shift(1).to_numpy()

    # Each non-rest sport gets its own recovery regime (interacted lag) AND load column, so
    # post-strength recovery is no longer mis-attributed to the last triathlon/soccer regime.
    # With only triathlon/soccer present this is exactly the original two-regime design.
    sports = modeled_sports(sport_arr)
    ar_cols = {s: f"_ar_{s}" for s in sports}
    load_cols = {s: f"_load_{s}" for s in sports}

    assigned = {"_dev": dev.to_numpy(), "_dev_lag": dev_lag, "_regime": regime}
    for s in sports:
        assigned[ar_cols[s]] = np.where(regime == s, dev_lag, 0.0)
        load = np.where(sport_arr == s, trimp / 100.0, 0.0)
        assigned[load_cols[s]] = pd.Series(load).shift(load_lag).to_numpy()
    work = work.assign(**assigned)

    cols = ["_dev", "_dev_lag", "_regime", *ar_cols.values(), *load_cols.values()]
    fit_df = work.dropna(subset=cols)
    if len(fit_df) < 30:
        raise ValueError("too few aligned observations to fit the per-sport recovery model")

    y = fit_df["_dev"].to_numpy(float)
    # columns: 0 intercept, 1.. per-sport phi (in ``sports`` order), then per-sport load.
    design = [np.ones(len(fit_df))]
    design += [fit_df[ar_cols[s]].to_numpy(float) for s in sports]
    design += [fit_df[load_cols[s]].to_numpy(float) for s in sports]
    X = np.column_stack(design)

    mu_n, cov_m, dof, _ = conjugate_posterior(X, y, prior_scale)
    phi_col = {s: 1 + i for i, s in enumerate(sports)}
    phi_mean = {s: float(mu_n[phi_col[s]]) for s in sports}

    # Propagate the joint phi posterior to tau via Monte-Carlo.
    rng = np.random.default_rng(seed)
    draws = sample_mvt(mu_n, cov_m, dof, n_draws, rng)
    lo_q, hi_q = (1.0 - ci_level) / 2.0, (1.0 + ci_level) / 2.0

    # Per-sport tau from the draws where that sport's phi is a valid recovery rate in (0, 1).
    tau, tau_ci, phi_draws, ok_draws = {}, {}, {}, {}
    for s in sports:
        ph = draws[:, phi_col[s]]
        ok_s = (ph > 0.0) & (ph < 1.0)
        phi_draws[s], ok_draws[s] = ph, ok_s
        tau_s_draws = -1.0 / np.log(ph[ok_s])
        tau[s] = float(np.mean(tau_s_draws))
        tau_ci[s] = (float(np.quantile(tau_s_draws, lo_q)), float(np.quantile(tau_s_draws, hi_q)))

    # Headline tau difference (soccer - triathlon), on draws where BOTH phis are valid.
    if "triathlon" in phi_col and "soccer" in phi_col:
        both = ok_draws["triathlon"] & ok_draws["soccer"]
        tau_t = -1.0 / np.log(phi_draws["triathlon"][both])
        tau_s = -1.0 / np.log(phi_draws["soccer"][both])
        tau_diff_draws = tau_s - tau_t
        tau_diff = float(np.mean(tau_diff_draws))
        tau_diff_ci = (
            float(np.quantile(tau_diff_draws, lo_q)),
            float(np.quantile(tau_diff_draws, hi_q)),
        )
        prob_longer = float(np.mean(tau_diff_draws > 0.0))
        # Pre-registered H-A2 decision (OSF §6): 95% CrI of the difference excludes 0.
        d_lo, d_hi = np.quantile(tau_diff_draws, [0.025, 0.975])
        h_a2_supported = bool(d_lo > 0.0 or d_hi < 0.0)
        n_draws_used = int(both.sum())
    else:
        tau_diff = float("nan")
        tau_diff_ci = (float("nan"), float("nan"))
        prob_longer = float("nan")
        h_a2_supported = False
        n_draws_used = 0

    regime_used = fit_df["_regime"]
    n_regime = {s: int((regime_used == s).sum()) for s in sports}

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
        n_draws_used=n_draws_used,
    )
