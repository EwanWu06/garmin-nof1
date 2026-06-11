"""Headline layer A: the differential-recovery-cost estimator must recover the
planted ground truth (soccer costs more vagal HRV per unit TRIMP than triathlon)
on synthetic data, and a negative control (sleep) must show no sport interaction.
"""

import numpy as np

from garmin_nof1.data.synthetic import GroundTruth, make_daily_panel
from garmin_nof1.models import fit_recovery_cost


def _panel_with_true_deviation(n_days=1460, seed=3, gt=None):
    gt = gt or GroundTruth()
    df = make_daily_panel(n_days=n_days, seed=seed, gt=gt)
    t = np.arange(len(df))
    baseline = gt.baseline + gt.seasonal_amp * np.sin(2 * np.pi * t / 365.0)
    df["dev_true"] = df["ln_rmssd"] - baseline  # oracle deviation for a clean recovery test
    return df, gt


def test_recovers_planted_cost_ordering_and_sign():
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df, deviation_col="dev_true")
    assert res.cost_slope["soccer"] > res.cost_slope["triathlon"] > 0
    assert res.interaction > 0
    assert res.interaction_ci[0] > 0  # 95% CI excludes zero


def test_recovers_planted_cost_magnitude():
    df, gt = _panel_with_true_deviation()
    res = fit_recovery_cost(df, deviation_col="dev_true")
    assert abs(res.cost_slope["triathlon"] - gt.beta_triathlon) < 0.03
    assert abs(res.cost_slope["soccer"] - gt.beta_soccer) < 0.04


def test_detrend_path_recovers_ordering():
    # realistic path: internal 28-day rolling detrend (attenuated, so ordering only)
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df)  # deviation_col=None -> detrend internally
    assert res.cost_slope["soccer"] > res.cost_slope["triathlon"] > 0


def test_negative_control_sleep_has_no_sport_interaction():
    # sleep carries a main load effect but NO sport*load interaction in the DGP
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df, outcome="sleep_hours")
    lo, hi = res.interaction_ci
    assert lo < 0 < hi


def test_reports_session_counts():
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df, deviation_col="dev_true")
    assert res.n["soccer"] > 0 and res.n["triathlon"] > 0


def test_exposes_preregistered_h_a1_decision_rule():
    # OSF preregistration §6: H-A1 supported iff P(interaction>0) >= 0.95 AND the
    # 95% CrI excludes a ROPE of +-0.02 ln-units/100-TRIMP. The fit must expose the
    # verdict, not just the ingredients.
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df, deviation_col="dev_true")
    assert res.prob_interaction_positive >= 0.95
    assert res.rope_excludes is True
    assert res.h_a1_supported is True


def test_negative_control_does_not_support_h_a1():
    # The sleep negative control has no sport interaction, so the pre-registered
    # decision rule must NOT declare H-A1 supported.
    df, _ = _panel_with_true_deviation()
    res = fit_recovery_cost(df, outcome="sleep_hours")
    assert res.h_a1_supported is False


def test_higher_ci_level_widens_interval():
    df, _ = _panel_with_true_deviation()
    res95 = fit_recovery_cost(df, deviation_col="dev_true", ci_level=0.95)
    res99 = fit_recovery_cost(df, deviation_col="dev_true", ci_level=0.99)
    w95 = res95.interaction_ci[1] - res95.interaction_ci[0]
    w99 = res99.interaction_ci[1] - res99.interaction_ci[0]
    assert w99 > w95


def test_irrelevant_covariate_leaves_interaction_conclusion():
    # The registered model carries sleep/dow/season covariates; the API must accept
    # extra covariates without disturbing the headline when they are uninformative.
    df, _ = _panel_with_true_deviation()
    base = fit_recovery_cost(df, deviation_col="dev_true")
    rng = np.random.default_rng(0)
    df = df.copy()
    df["noise_cov"] = rng.normal(size=len(df))
    with_cov = fit_recovery_cost(df, deviation_col="dev_true", covariates=["noise_cov"])
    assert with_cov.h_a1_supported == base.h_a1_supported is True
    assert abs(with_cov.interaction - base.interaction) < 0.01
