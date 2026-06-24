"""Layer A (headline): the differential recovery-cost estimator.

What this answers
-----------------
Within *this one person*, does a unit of training load cost more next-night vagal
HRV after **soccer** (intermittent sprint) than after **triathlon** (endurance)?
The headline quantity is the *sport × load* interaction on the recovery-cost scale,
plus the per-sport cost slopes and the AR(1) persistence of the HRV deviation.

The model (faithful to ``preregistration/OSF_preregistration.md`` §5)
---------------------------------------------------------------------
Nightly ``ln rMSSD`` is trend-dominated, so it is modeled in **deviation form**:
``dev[t] = ln_rmssd[t] - baseline[t]``, where ``baseline`` is either supplied
(``deviation_col``, e.g. an oracle baseline on synthetic data) or estimated
internally as a centered rolling mean (the pre-registered 28-day window). On the
deviation we fit a within-person autoregressive load model::

    dev[t] = c + phi * dev[t-1]
             - cost_triathlon * tri_load[t]
             - cost_soccer    * soc_load[t]
             + eps[t]

where ``*_load[t] = 1[sport[t] == s] * trimp[t] / 100`` (load in units of 100
TRIMP, so a slope reads directly as "ln-rMSSD lost per 100 TRIMP"). The contemporaneous
load with a lag-1 autoregressive term mirrors the synthetic data-generating process
in :mod:`garmin_nof1.data.synthetic`: a load shock on day *t* suppresses that night's
HRV deviation, and the suppression then decays geometrically through ``phi`` —
giving a recovery time-constant ``tau = -1 / ln(phi)``.

Indexing convention: ``ln_rmssd[t]`` is the HRV measured during the sleep that
*follows* day-*t*'s training, so the "next-night cost of day-*t* load" lands on the
same row index *t*. Regressing the deviation level on its lag-1 with the
contemporaneous load is algebraically the AR(1)-residualized change the OSF plan
writes as ``Δlndev ~ sport*TRIMP + AR(1) residual``; the two notations describe the
one structure (day-*t* load → the following night's HRV, with carry-over via ``phi``).

Inference is **Bayesian** with weakly-informative priors via the conjugate
Normal-Inverse-Gamma posterior (closed form, deterministic, no MCMC dependency).
We report the **posterior** of every quantity as a credible interval — never a
p-value — exactly as pre-registered. The conjugate linear form is the analytic
realization of the pre-registered within-person model for the synthetic-recovery
test and the daily panel; a partially-pooled MCMC variant (PyMC) is reserved for
real-data robustness runs and adds nothing to recovering the planted effect here.

Honest scope notes
-------------------
* The centered rolling-mean detrend is the pre-registered *deviation definition*
  for this in-sample structural-estimation layer. It deliberately uses both past
  and future days and would leak for out-of-sample prediction; the demoted
  prediction layer (Phase 4) never uses it — it goes through the leakage-safe
  :mod:`garmin_nof1.eval.cv` scaffold instead.
* ``cost_slope`` is reported with the sign convention "positive = HRV suppressed"
  (i.e. a *cost*), which is the negative of the raw load regression coefficient.
* This estimator implements the H-A1 headline (the sport × load interaction) and
  exposes the pre-registered decision rule (``h_a1_supported``). It does **not**
  implement H-A2 (does recovery τ *differ by sport*): the pooled ``tau_days`` here is
  a single global constant. H-A2 needs a per-sport post-session decay fit on a DGP
  that plants sport-specific φ — a separate estimator and synthetic substrate, left
  as the next increment.
* On real data, part of any "soccer costs more per TRIMP" effect can be a TRIMP
  *measurement* artifact (TRIMP underestimates intermittent-sprint load). Here the
  synthetic substrate puts the sport effect entirely in the slope, so this is a
  clean test of the *estimator*, not a physiological claim — see the module
  docstring of :mod:`garmin_nof1.data.synthetic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from garmin_nof1.models._common import ci, conjugate_posterior, deviation, modeled_sports


@dataclass(frozen=True)
class RecoveryCostFit:
    """Posterior summary of the differential recovery-cost model.

    Attributes
    ----------
    cost_slope : dict[str, float]
        Posterior-mean next-night ln-rMSSD cost per 100 TRIMP, per sport
        (positive = HRV suppressed).
    cost_slope_ci : dict[str, tuple[float, float]]
        95% credible interval for each ``cost_slope``.
    interaction : float
        ``cost_slope['soccer'] - cost_slope['triathlon']`` — the headline (H-A1).
    interaction_ci : tuple[float, float]
        Credible interval for ``interaction`` at the reporting ``ci_level`` (default 95%).
    prob_interaction_positive : float
        Posterior ``P(interaction > 0)``.
    rope_excludes : bool
        Whether the **95%** credible interval excludes the pre-registered ROPE of
        ``±rope_margin`` around 0 (always evaluated at 95%, per OSF §6, regardless of
        the reporting ``ci_level``).
    h_a1_supported : bool
        The pre-registered H-A1 verdict: ``prob_interaction_positive >= 0.95`` **and**
        ``rope_excludes`` (OSF preregistration §6).
    rope_margin : float
        The ROPE half-width used for ``rope_excludes`` (ln-units per 100 TRIMP).
    n : dict[str, int]
        Number of sessions of each sport used in the fit (after alignment/NaN drop).
    phi : float
        Posterior-mean AR(1) persistence of the HRV deviation (pooled across sports).
    tau_days : float
        Global recovery time-constant ``-1 / ln(phi)`` implied by the pooled ``phi``
        (NaN if ``phi`` is not in (0, 1)). **This is not the H-A2 quantity:** the
        pre-registered headline H-A2 (τ *differs by sport*) needs a per-sport
        post-session decay fit and is a separate, not-yet-implemented estimator.
    sigma : float
        Posterior-mean residual scale (ln units).
    n_obs : int
        Total day-rows used in the regression.
    """

    cost_slope: dict[str, float]
    cost_slope_ci: dict[str, tuple[float, float]]
    interaction: float
    interaction_ci: tuple[float, float]
    prob_interaction_positive: float
    rope_excludes: bool
    h_a1_supported: bool
    rope_margin: float
    n: dict[str, int]
    phi: float
    tau_days: float
    sigma: float
    n_obs: int


def fit_recovery_cost(
    df: pd.DataFrame,
    *,
    outcome: str = "ln_rmssd",
    deviation_col: str | None = None,
    covariates: list[str] | None = None,
    detrend_window: int = 28,
    prior_scale: float = 10.0,
    ci_level: float = 0.95,
    rope_margin: float = 0.02,
    load_lag: int = 0,
) -> RecoveryCostFit:
    """Estimate the per-sport recovery cost and the sport × load interaction.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily panel with columns ``sport`` (categorical incl. ``triathlon`` /
        ``soccer`` / ``rest``), ``trimp``, and the outcome column. Rows are assumed
        contiguous calendar days (as produced by the panel builder), so the lag-1
        AR term is a true one-day lag — see the note where ``_dev_lag`` is built.
    outcome : str
        Outcome column to detrend when ``deviation_col`` is None (default the
        nightly ``ln_rmssd``; the negative control passes ``"sleep_hours"``). When
        ``deviation_col`` is None the outcome is detrended by a centered rolling mean.
    deviation_col : str, optional
        Pre-computed deviation column to use verbatim (skips internal detrending).
    covariates : list[str], optional
        Extra columns to enter the design matrix (mean-centered), e.g. ``sleep``,
        day-of-week dummies, a seasonal sinusoid, or pre-computed distributed-lag
        load terms. This is the bridge from the clean synthetic substrate (where the
        minimal AR + contemporaneous-load model matches the DGP exactly) to the full
        pre-registered real-data model (OSF §5). The sport × load contrast is unaffected
        by appended covariates.
    detrend_window : int
        Centered rolling-mean window for internal detrending (pre-registered: 28).
    prior_scale : float
        Std of the weakly-informative zero-mean Gaussian coefficient prior.
    ci_level : float
        Reporting credible level for ``interaction_ci`` / ``cost_slope_ci`` (default 0.95).
        The H-A1 ROPE decision is always evaluated at 95% per the preregistration.
    rope_margin : float
        ROPE half-width (ln-units per 100 TRIMP) for the H-A1 decision (OSF §6: ±0.02).
    load_lag : int
        Number of days by which a session's load is shifted forward onto the HRV night it
        first affects. ``0`` (default) aligns day-*t* load with the same-row deviation — the
        convention of the synthetic DGP. Real Garmin overnight HRV is timestamped to the
        morning, so a day's training first lands on the *next* night: use ``load_lag=1`` on
        real panels. (Mis-aligning load to the same night captures the "train-when-recovered"
        behavioural confound instead of the recovery cost — see ``preregistration``.)
    """
    covariates = list(covariates or [])
    work = df.copy()
    dev = deviation(work, outcome, deviation_col, detrend_window)
    trimp = work["trimp"].astype(float).to_numpy()
    sport = np.asarray(work["sport"].astype(object))

    # Each non-rest sport gets its own load column (data-driven): its nights are explained by
    # its own slope and so leave the "rest" baseline uncontaminated. With only triathlon/soccer
    # present this is exactly the original two-load design.
    # Known limitation: the panel stores one sport label + the whole day's summed TRIMP, so on a
    # mixed-session day (e.g. run + strength, ~3% of active days) the full load is attributed to
    # the dominant sport's column and the co-occurring sport's load reads zero. Per-session load
    # decomposition would remove this; the effect on the headline contrast is negligible.
    if load_lag < 0:
        raise ValueError("load_lag must be >= 0")
    sports = modeled_sports(sport)
    load_cols = {s: f"_load_{s}" for s in sports}

    # _dev_lag is a positional shift computed BEFORE dropna; because the panel is
    # contiguous daily rows, a row survives the dropna only if it and its immediately
    # preceding row are both observed -> the surviving AR pairs are true one-day lags
    # (a row whose previous night was non-wear has a NaN lag and is dropped, rather
    # than being silently paired with an earlier day). Each load column is shifted forward
    # by ``load_lag`` so a day's session is matched to the HRV night it first affects.
    assigned = {"_dev": dev.to_numpy(), "_dev_lag": dev.shift(1).to_numpy()}
    for s in sports:
        load = np.where(sport == s, trimp / 100.0, 0.0)
        assigned[load_cols[s]] = pd.Series(load).shift(load_lag).to_numpy()
    work = work.assign(**assigned)
    for cov in covariates:
        if cov not in work.columns:
            raise KeyError(f"covariate column {cov!r} not in DataFrame")

    cols = ["_dev", "_dev_lag", *load_cols.values(), *covariates]
    fit_df = work.dropna(subset=cols)
    if len(fit_df) < 10:
        raise ValueError("too few aligned observations to fit the recovery-cost model")

    y = fit_df["_dev"].to_numpy(float)
    # columns: 0 intercept, 1 AR(1), 2.. per-sport load (in ``sports`` order), then covariates
    design = [np.ones(len(fit_df)), fit_df["_dev_lag"].to_numpy(float)]
    design += [fit_df[load_cols[s]].to_numpy(float) for s in sports]
    for cov in covariates:
        v = fit_df[cov].to_numpy(float)
        design.append(v - v.mean())  # center so the intercept stays interpretable
    X = np.column_stack(design)

    mu_n, cov_m, dof, sigma2_mean = conjugate_posterior(X, y, prior_scale)

    # cost = -(load coefficient): positive cost == HRV suppressed.
    idx = {s: 2 + i for i, s in enumerate(sports)}
    cost_slope = {s: float(-mu_n[idx[s]]) for s in sports}
    cost_slope_ci = {
        s: tuple(sorted(ci(-mu_n[idx[s]], float(np.sqrt(cov_m[idx[s], idx[s]])), dof, ci_level)))
        for s in sports
    }

    # Headline interaction = cost_soccer - cost_triathlon = mu_triathlon - mu_soccer.
    # Defined only when both headline sports are present (always so on the real/synthetic panel).
    if "triathlon" in idx and "soccer" in idx:
        c = np.zeros(X.shape[1])
        c[idx["triathlon"]], c[idx["soccer"]] = 1.0, -1.0
        interaction = float(c @ mu_n)
        inter_scale = float(np.sqrt(c @ cov_m @ c))
        interaction_ci = ci(interaction, inter_scale, dof, ci_level)
        prob_pos = (
            float(stats.t.cdf(interaction / inter_scale, dof)) if inter_scale > 0 else float("nan")
        )
        # Pre-registered H-A1 decision (OSF §6): fixed at the 95% level, independent of ci_level.
        lo95, hi95 = ci(interaction, inter_scale, dof, 0.95)
        rope_excludes = bool(lo95 > rope_margin or hi95 < -rope_margin)
        h_a1_supported = bool(prob_pos >= 0.95 and rope_excludes)
    else:
        interaction = float("nan")
        interaction_ci = (float("nan"), float("nan"))
        prob_pos = float("nan")
        rope_excludes = False
        h_a1_supported = False

    phi = float(mu_n[1])
    tau_days = float(-1.0 / np.log(phi)) if 0.0 < phi < 1.0 else float("nan")

    # Count each sport by the rows whose lag-aligned load actually entered the regression
    # (nonzero load column), not the deviation row's own-day sport — these differ when
    # load_lag > 0, and it is the contributing session that the slope is estimated from.
    n = {s: int((fit_df[load_cols[s]].to_numpy() != 0).sum()) for s in sports}

    return RecoveryCostFit(
        cost_slope=cost_slope,
        cost_slope_ci=cost_slope_ci,
        interaction=interaction,
        interaction_ci=interaction_ci,
        prob_interaction_positive=prob_pos,
        rope_excludes=rope_excludes,
        h_a1_supported=h_a1_supported,
        rope_margin=rope_margin,
        n=n,
        phi=phi,
        tau_days=tau_days,
        sigma=float(np.sqrt(sigma2_mean)),
        n_obs=len(fit_df),
    )
