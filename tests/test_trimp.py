import math

import pytest

from garmin_nof1.pipeline.trimp import banister_trimp, edwards_trimp, trimp_variants


def test_banister_zero_when_hr_at_rest():
    assert banister_trimp(60.0, hr_avg=50.0, hr_rest=50.0, hr_max=200.0, sex="M") == 0.0


def test_banister_known_value_male():
    # HRr = (150-50)/(200-50) = 0.6667; TRIMP = 60 * HRr * 0.64 * e^(1.92*HRr)
    hrr = (150 - 50) / (200 - 50)
    expected = 60.0 * hrr * 0.64 * math.exp(1.92 * hrr)
    assert abs(banister_trimp(60.0, 150.0, 50.0, 200.0, "M") - expected) < 1e-6


def test_banister_monotone_in_avg_hr():
    low = banister_trimp(60.0, 120.0, 50.0, 200.0)
    high = banister_trimp(60.0, 170.0, 50.0, 200.0)
    assert high > low > 0


def test_banister_rejects_bad_hr_bounds():
    with pytest.raises(ValueError):
        banister_trimp(60.0, 150.0, hr_rest=200.0, hr_max=200.0)


def test_edwards_weighted_zone_sum():
    # zones 1..5 minutes weighted by zone number
    assert edwards_trimp([10, 20, 5, 2, 1]) == 1 * 10 + 2 * 20 + 3 * 5 + 4 * 2 + 5 * 1


def test_trimp_variants_includes_requested():
    out = trimp_variants(
        60.0, 150.0, 50.0, 200.0, time_in_zones_min=[10, 20, 5, 2, 1], garmin_load=88.0
    )
    assert set(out) == {"banister", "edwards", "garmin"}
    assert out["garmin"] == 88.0
