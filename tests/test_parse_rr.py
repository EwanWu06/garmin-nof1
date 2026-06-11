import numpy as np
import pytest

from garmin_nof1.data.synthetic import make_rr_series
from garmin_nof1.pipeline.parse_rr import mean_hr, reconstruct_rr_ms, rmssd, sdnn


def test_reconstruct_flattens_drops_none_and_scales_to_ms():
    # fitparse yields each hrv message's `time` as a list in SECONDS, padding = None
    arrays = [[0.800, 0.810, None], None, [0.795], [None, 0.805]]
    rr = reconstruct_rr_ms(arrays)
    assert np.allclose(rr, [800.0, 810.0, 795.0, 805.0])


def test_reconstruct_handles_scalar_and_empty():
    assert reconstruct_rr_ms([]).size == 0
    assert np.allclose(reconstruct_rr_ms([0.9]), [900.0])  # scalar tolerated


def test_rmssd_matches_known_definition():
    rr = np.array([800.0, 810.0, 790.0, 805.0])
    expected = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    assert abs(rmssd(rr) - expected) < 1e-9


def test_rmssd_recovers_synthetic_target():
    rr = make_rr_series(n_beats=4000, rmssd_target=45.0, seed=0)
    assert 35.0 < rmssd(rr) < 55.0


def test_sdnn_and_mean_hr():
    rr = np.array([1000.0, 1000.0, 1000.0])  # 60 bpm, zero variability
    assert abs(mean_hr(rr) - 60.0) < 1e-9
    assert sdnn(rr) == 0.0


def test_metrics_require_enough_beats():
    with pytest.raises(ValueError):
        rmssd([800.0])
