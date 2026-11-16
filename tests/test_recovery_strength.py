"""Multi-sport (strength) validation of the Layer-A estimators.

Strength training is real autonomic load. The design decision (and the methodological point)
is that it must be modeled as its OWN category, not folded into the ``rest`` baseline. These
tests check, on the synthetic substrate, that:

* the estimators discover and estimate a third sport ("strength") without disturbing the
  triathlon-vs-soccer headline (H-A1 / H-A2), and
* modeling strength explains variance that lumping it into ``rest`` cannot (lower residual
  scale) — the concrete payoff of not contaminating the baseline.
"""

import numpy as np

from garmin_nof1.data.synthetic import GroundTruth, make_daily_panel
from garmin_nof1.models import fit_recovery_cost, fit_recovery_tau
from garmin_nof1.models._common import modeled_sports


def _with_oracle_dev(df, gt):
    t = np.arange(len(df))
    baseline = gt.baseline + gt.seasonal_amp * np.sin(2 * np.pi * t / 365.0)
    df = df.copy()
    df["dev_true"] = df["ln_rmssd"] - baseline
    return df


def test_modeled_sports_orders_headline_first():
    assert modeled_sports(np.array(["rest", "soccer", "triathlon"])) == ["triathlon", "soccer"]
    assert modeled_sports(np.array(["rest", "strength", "soccer", "triathlon"])) == [
        "triathlon",
        "soccer",
        "strength",
    ]
    assert modeled_sports(np.array(["rest", "rest"])) == []


def test_p_strength_generates_strength_days_else_none():
    none = make_daily_panel(n_days=400, seed=0)
    assert (none["sport"].astype(str) == "strength").sum() == 0
    gt = GroundTruth(beta_strength=0.1)
    with_str = make_daily_panel(n_days=400, seed=0, gt=gt, p_strength=0.2)
    mask = with_str["sport"].astype(str) == "strength"
    assert mask.sum() > 0
    assert (with_str.loc[mask, "trimp"] > 0).all()


def test_cost_model_estimates_strength_and_keeps_headline():
    gt = GroundTruth(beta_strength=0.10)
    df = _with_oracle_dev(make_daily_panel(n_days=1460, seed=11, gt=gt, p_strength=0.20), gt)
    res = fit_recovery_cost(df, deviation_col="dev_true")
    # third sport is estimated explicitly...
    assert "strength" in res.cost_slope and res.n["strength"] > 0
    assert abs(res.cost_slope["strength"] - gt.beta_strength) < 0.05
    # ...without disturbing the triathlon-vs-soccer headline
    assert res.cost_slope["soccer"] > res.cost_slope["triathlon"] > 0
    assert res.h_a1_supported is True


def test_modeling_strength_explains_variance_lumping_into_rest_cannot():
    gt = GroundTruth(beta_strength=0.12)
    df = _with_oracle_dev(make_daily_panel(n_days=1460, seed=4, gt=gt, p_strength=0.25), gt)
    modeled = fit_recovery_cost(df, deviation_col="dev_true")
    naive = df.copy()
    naive["sport"] = naive["sport"].astype(object).replace("strength", "rest")
    naive_fit = fit_recovery_cost(naive, deviation_col="dev_true")
    # the naive fit cannot even name strength; modeling it leaves a smaller residual scale
    assert "strength" not in naive_fit.cost_slope
    assert modeled.sigma < naive_fit.sigma


def test_tau_model_estimates_strength_regime_and_keeps_headline():
    gt = GroundTruth(phi_triathlon=0.70, phi_soccer=0.88, phi_strength=0.80, beta_strength=0.08)
    df = _with_oracle_dev(make_daily_panel(n_days=1460, seed=5, gt=gt, p_strength=0.20), gt)
    res = fit_recovery_tau(df, deviation_col="dev_true")
    assert "strength" in res.tau and res.tau["strength"] > 0
    assert res.n_regime["strength"] > 0
    # headline recovery-speed ordering survives the added regime
    assert res.tau["soccer"] > res.tau["triathlon"] > 0
    assert res.h_a2_supported is True
