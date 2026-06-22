"""Demoted prediction layer (H-P1): does cross-sport load add next-day-lnRMSSD skill
over an AR(1) / random-walk baseline, evaluated leakage-safely (PurgedWalkForward + CPCV)?

These tests lock the falsification machinery: it must (a) detect incremental skill when a
next-day load effect is genuinely planted, and (b) NOT manufacture skill when load is noise
(the pre-registered prior is null). Real-data interpretation is done in scripts/, not here.
"""

import numpy as np
import pandas as pd

from garmin_nof1.models.prediction import (
    PredictionResult,
    build_supervised,
    evaluate_prediction,
    holdout_skill,
    holdout_split,
)


def _panel(n, *, planted_next_day_effect, seed):
    """Daily panel where, optionally, day-t load drives day-(t+1) lnRMSSD beyond AR(1)."""
    rng = np.random.default_rng(seed)
    sport = np.where(rng.random(n) < 0.5, "triathlon", "rest")
    trimp = np.where(sport == "triathlon", rng.uniform(40, 120, n), 0.0)
    eps = rng.normal(0, 0.05, n)
    y = np.empty(n)
    y[0] = 4.0
    beta = 0.6 if planted_next_day_effect else 0.0
    for t in range(1, n):
        y[t] = 4.0 + 0.3 * (y[t - 1] - 4.0) - beta * trimp[t - 1] / 100.0 + eps[t]
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n),
            "sport": pd.Categorical(sport, categories=["rest", "triathlon", "soccer", "strength"]),
            "trimp": trimp,
            "ln_rmssd": y,
            "sleep_hours": 7.0,
            "rhr": 50.0,
            "hrv_observed": True,
        }
    )


def test_build_supervised_pairs_today_features_with_next_day_target():
    df = _panel(20, planted_next_day_effect=True, seed=0)
    sup = build_supervised(df)
    # one fewer row than the panel (last day has no next-day target)
    assert len(sup) == len(df) - 1
    assert {"y_next", "y_t"}.issubset(sup.columns)
    assert any(c.startswith("load_") for c in sup.columns)
    # y_next on row 0 is the panel's ln_rmssd on day 1
    assert abs(sup.iloc[0]["y_next"] - df.iloc[1]["ln_rmssd"]) < 1e-12
    assert abs(sup.iloc[0]["y_t"] - df.iloc[0]["ln_rmssd"]) < 1e-12


def test_build_supervised_drops_pairs_with_missing_nights():
    df = _panel(20, planted_next_day_effect=False, seed=1)
    df.loc[5, "ln_rmssd"] = np.nan  # non-wear night
    sup = build_supervised(df)
    # day 4->5 (target NaN) and day 5->6 (feature NaN) are both dropped
    assert len(sup) == (len(df) - 1) - 2


def test_detects_planted_next_day_skill():
    df = _panel(700, planted_next_day_effect=True, seed=2)
    res = evaluate_prediction(df, n_groups=6, n_test_groups=2, embargo=2)
    assert isinstance(res, PredictionResult)
    assert res.skill_improvement > 0  # candidate beats AR(1) on average
    assert res.skill_improvement_p05 > 0  # ... even at the 5th percentile of CPCV paths
    assert res.beats_baseline is True
    assert res.rmse["candidate"] < res.rmse["ar1"] < res.rmse["random_walk"]


def test_does_not_manufacture_skill_from_noise_load():
    df = _panel(700, planted_next_day_effect=False, seed=3)
    res = evaluate_prediction(df, n_groups=6, n_test_groups=2, embargo=2)
    # load is uninformative -> the pre-registered decision must NOT fire
    assert res.beats_baseline is False
    assert res.skill_improvement_p05 <= 0


def test_ar1_beats_random_walk_on_mean_reverting_series():
    df = _panel(700, planted_next_day_effect=False, seed=4)
    res = evaluate_prediction(df, n_groups=6, n_test_groups=2, embargo=2)
    # mean-reverting (phi=0.3) -> fitted AR(1) predicts better than persistence
    assert res.rmse["ar1"] < res.rmse["random_walk"]


def test_holdout_split_is_the_temporal_tail():
    df = _panel(100, planted_next_day_effect=False, seed=5)
    dev, hold = holdout_split(df, frac=0.2)
    assert len(hold) == 20 and len(dev) == 80
    assert dev["date"].max() < hold["date"].min()  # holdout is strictly later
    assert len(dev) + len(hold) == len(df)


def test_holdout_skill_confirms_planted_effect_on_unseen_tail():
    df = _panel(700, planted_next_day_effect=True, seed=7)
    dev, hold = holdout_split(df, frac=0.2)
    out = holdout_skill(dev, hold)
    assert out["n_holdout"] > 0
    assert out["skill_improvement"] > 0  # candidate beats AR(1) on the unseen holdout
    assert out["rmse"]["candidate"] < out["rmse"]["ar1"]


def test_holdout_skill_no_effect_when_load_is_noise():
    df = _panel(700, planted_next_day_effect=False, seed=8)
    dev, hold = holdout_split(df, frac=0.2)
    out = holdout_skill(dev, hold)
    # adding noise load features should not help on the unseen tail
    assert out["skill_improvement"] <= 0.005  # ~0 or slightly negative


def test_evaluate_reports_effective_sample_size():
    df = _panel(400, planted_next_day_effect=True, seed=6)
    res = evaluate_prediction(df, n_groups=6, n_test_groups=2, embargo=2)
    assert 0 < res.ess <= len(df)
    assert res.n_splits == 15  # C(6,2) combinatorial train/test combinations
