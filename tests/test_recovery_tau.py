"""Headline layer A, second half (H-A2): the per-sport recovery-time-constant
estimator must recover a planted tau difference (soccer recovers slower than
triathlon) on synthetic data, and must NOT manufacture a difference when the
data-generating process has a single global recovery rate.
"""

import numpy as np

from garmin_nof1.data.synthetic import GroundTruth, make_daily_panel
from garmin_nof1.models import fit_recovery_tau


def _oracle_dev(df, gt):
    t = np.arange(len(df))
    baseline = gt.baseline + gt.seasonal_amp * np.sin(2 * np.pi * t / 365.0)
    df = df.copy()
    df["dev_true"] = df["ln_rmssd"] - baseline
    return df


def _panel_per_sport_tau(n_days=1460, seed=5):
    # soccer recovers slower (phi closer to 1 -> larger tau) than triathlon
    gt = GroundTruth(phi_triathlon=0.70, phi_soccer=0.88)
    df = _oracle_dev(make_daily_panel(n_days=n_days, seed=seed, gt=gt), gt)
    return df, gt


def test_recovers_planted_tau_ordering_and_sign():
    df, _ = _panel_per_sport_tau()
    res = fit_recovery_tau(df, deviation_col="dev_true")
    assert res.tau["soccer"] > res.tau["triathlon"] > 0
    assert res.tau_diff > 0
    assert res.tau_diff_ci[0] > 0  # 95% credible interval excludes 0


def test_h_a2_supported_when_planted():
    df, _ = _panel_per_sport_tau()
    res = fit_recovery_tau(df, deviation_col="dev_true")
    assert res.h_a2_supported is True
    assert res.prob_tau_soccer_longer > 0.95


def test_recovers_planted_tau_magnitude():
    df, gt = _panel_per_sport_tau()
    res = fit_recovery_tau(df, deviation_col="dev_true")
    assert abs(res.tau["triathlon"] - gt.tau_triathlon) < 1.0
    assert abs(res.tau["soccer"] - gt.tau_soccer) < 3.0


def test_global_phi_does_not_manufacture_tau_gap():
    # Under one global phi, tau is identical for both sports. Averaged over seeds the
    # estimated gap must sit at ~0 — an order of magnitude below the ~5-day gap recovered
    # when a difference is truly planted — i.e. the estimator does not INVENT a
    # recovery-speed difference. (A single 95% CrI legitimately excludes 0 ~5% of the time
    # under the null, so we assert calibration across seeds, not one straddle.)
    gt = GroundTruth()
    diffs = np.array(
        [
            fit_recovery_tau(
                _oracle_dev(make_daily_panel(n_days=1460, seed=s, gt=gt), gt),
                deviation_col="dev_true",
                n_draws=5000,
            ).tau_diff
            for s in range(16)
        ]
    )
    assert abs(diffs.mean()) < 0.25  # no systematic gap (unbiased)
    assert np.median(np.abs(diffs)) < 0.5  # individual gaps tiny vs the planted ~5 days


def test_reports_regime_counts():
    df, _ = _panel_per_sport_tau()
    res = fit_recovery_tau(df, deviation_col="dev_true")
    assert res.n_regime["soccer"] > 0 and res.n_regime["triathlon"] > 0
