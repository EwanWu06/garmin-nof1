import numpy as np
import pytest

from garmin_nof1.data.synthetic import make_rr_series
from garmin_nof1.pipeline.parse_rr import (
    HrvMetrics,
    filter_artifacts,
    mean_hr,
    metrics_from_rr,
    reconstruct_rr_ms,
    rmssd,
    sdnn,
)


def test_reconstruct_flattens_drops_none_and_scales_to_ms():
    # fitparse yields each hrv message's `time` as a list in SECONDS, padding = None
    arrays = [[0.800, 0.810, None], None, [0.795], [None, 0.805]]
    rr = reconstruct_rr_ms(arrays)
    assert np.allclose(rr, [800.0, 810.0, 795.0, 805.0])


def test_reconstruct_handles_scalar_and_empty():
    assert reconstruct_rr_ms([]).size == 0
    # arr=0.9 (bare float) hits the scalar branch
    assert np.allclose(reconstruct_rr_ms([0.9]), [900.0])


def test_rmssd_matches_known_definition():
    rr = np.array([800.0, 810.0, 790.0, 805.0])
    # diffs [10, -20, 15]; mean([100, 400, 225]) = 241.6667; sqrt = 15.5456...
    assert abs(rmssd(rr) - 15.5456317551) < 1e-6


def test_rmssd_recovers_synthetic_target():
    rr = make_rr_series(n_beats=4000, rmssd_target=45.0, seed=0)
    assert 35.0 < rmssd(rr) < 55.0


def test_sdnn_and_mean_hr():
    rr = np.array([1000.0, 1000.0, 1000.0])  # 60 bpm, zero variability
    assert abs(mean_hr(rr) - 60.0) < 1e-9
    assert sdnn(rr) == 0.0


def test_metrics_require_enough_beats():
    with pytest.raises(ValueError):
        rmssd([800.0])  # needs >= 2
    with pytest.raises(ValueError):
        sdnn([800.0])  # needs >= 2
    with pytest.raises(ValueError):
        mean_hr([])  # needs >= 1


def test_filter_artifacts_drops_outliers_vs_previous_accepted():
    rr = np.array([800.0, 810.0, 1500.0, 805.0])  # 1500 is an ectopic spike
    kept = filter_artifacts(rr, threshold=0.2)
    assert np.allclose(kept, [800.0, 810.0, 805.0])


def test_filter_artifacts_empty():
    assert filter_artifacts(np.array([])).size == 0


def test_metrics_from_rr_reports_removed_count():
    rr = np.array([800.0, 810.0, 1500.0, 805.0])
    m = metrics_from_rr(rr, correct_artifacts=True)
    assert isinstance(m, HrvMetrics)
    assert m.n_beats == 3 and m.n_artifacts_removed == 1
    assert m.rmssd > 0 and m.sdnn >= 0 and 40.0 < m.mean_hr < 90.0
