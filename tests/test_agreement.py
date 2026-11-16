"""Agreement statistics for the D-layer measurement validation.

Targets are hand-computed on a tiny fixture (ref=[10,20,30], test=[11,22,33]) so each
statistic is checked against a known number, not just a property.
"""

import numpy as np
import pytest

from garmin_nof1.eval.agreement import agreement, bland_altman, ccc, icc_2_1, mape

REF = np.array([10.0, 20.0, 30.0])
TEST = np.array([11.0, 22.0, 33.0])


def test_bland_altman_bias_and_limits():
    ba = bland_altman(REF, TEST)  # diff = test - ref = [1, 2, 3]
    assert abs(ba.bias - 2.0) < 1e-9
    assert abs(ba.sd_diff - 1.0) < 1e-9  # sample SD (ddof=1) of [1,2,3]
    assert abs(ba.loa_lower - (2.0 - 1.96)) < 1e-9
    assert abs(ba.loa_upper - (2.0 + 1.96)) < 1e-9


def test_mape_is_ten_percent():
    assert abs(mape(REF, TEST) - 10.0) < 1e-9


def test_icc_2_1_matches_hand_computed_anova():
    # Hand-computed ICC(2,1) absolute-agreement for the fixture = 0.97922.
    assert abs(icc_2_1(REF, TEST) - 0.97922) < 1e-3


def test_ccc_matches_hand_computed():
    # Lin's CCC for the fixture = 0.96916 (population cov/var).
    assert abs(ccc(REF, TEST) - 0.96916) < 1e-3


def test_identical_series_are_perfect_agreement():
    x = np.array([50.0, 55.0, 60.0, 48.0])
    assert abs(icc_2_1(x, x) - 1.0) < 1e-9
    assert abs(ccc(x, x) - 1.0) < 1e-9
    ba = bland_altman(x, x)
    assert ba.bias == 0.0 and ba.sd_diff == 0.0
    assert mape(x, x) == 0.0


def test_icc_penalizes_a_constant_shift_more_than_ccc_consistency():
    # A pure additive shift hurts absolute-agreement ICC(2,1) (it sees the bias) but the
    # ranks are perfect — a basic sanity check that ICC reacts to systematic offset.
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    shifted = x + 5.0
    assert icc_2_1(x, shifted) < 1.0
    assert bland_altman(x, shifted).bias == 5.0


def test_agreement_bundles_all_metrics_with_n():
    res = agreement(REF, TEST)
    assert res.n == 3
    assert abs(res.bias - 2.0) < 1e-9
    assert abs(res.mape - 10.0) < 1e-9
    assert abs(res.icc - 0.97922) < 1e-3
    assert abs(res.ccc - 0.96916) < 1e-3


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        bland_altman(np.array([1.0, 2.0]), np.array([1.0]))


def test_too_few_pairs_raises():
    with pytest.raises(ValueError, match="at least 2"):
        icc_2_1(np.array([1.0]), np.array([2.0]))
