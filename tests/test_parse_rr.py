import numpy as np
import pytest

from garmin_nof1.data.synthetic import make_rr_series
from garmin_nof1.pipeline.parse_rr import (
    ActivityRecord,
    HrvMetrics,
    activity_record_from_fitfile,
    filter_artifacts,
    mean_hr,
    metrics_from_rr,
    reconstruct_rr_ms,
    rmssd,
    rr_quality,
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


def test_metrics_from_rr_raises_when_too_few_beats_after_filter():
    with pytest.raises(ValueError, match="remain after artifact filtering"):
        metrics_from_rr(np.array([800.0, 2000.0]))


class _FakeMsg:
    def __init__(self, name, values):
        self.name = name
        self._values = values

    def get_value(self, field, fallback=None):
        return self._values.get(field, fallback)


class _FakeFitFile:
    """Mimics the slice of fitparse.FitFile that the adapter uses."""

    def __init__(self, messages):
        self._messages = messages

    def get_messages(self, name):
        return [m for m in self._messages if m.name == name]


def test_activity_record_from_fitfile_extracts_rr_and_session():
    fit = _FakeFitFile(
        [
            _FakeMsg("hrv", {"time": [0.800, 0.810, None]}),
            _FakeMsg("hrv", {"time": [0.795, 0.805]}),
            _FakeMsg(
                "session",
                {
                    "sport": "soccer",
                    "start_time": "2024-05-01T18:00:00",
                    "avg_heart_rate": 150,
                    "total_timer_time": 3600.0,
                },
            ),
        ]
    )
    rec = activity_record_from_fitfile(fit, correct_artifacts=False)
    assert isinstance(rec, ActivityRecord)
    assert rec.sport == "soccer" and rec.hr_avg == 150
    assert abs(rec.duration_min - 60.0) < 1e-9
    assert np.allclose(rec.rr_ms, [800.0, 810.0, 795.0, 805.0])
    assert rec.metrics is not None and rec.metrics.n_beats == 4


def test_activity_record_without_session_or_hrv():
    fit = _FakeFitFile([_FakeMsg("record", {"heart_rate": 120})])
    rec = activity_record_from_fitfile(fit)
    assert rec.sport is None and rec.rr_ms.size == 0 and rec.metrics is None


def test_adapter_sets_metrics_none_when_filter_exhausts_beats():
    # raw has 2 beats but the second is an artifact -> filter leaves 1 -> metrics None
    fit = _FakeFitFile([_FakeMsg("hrv", {"time": [0.800, 2.000]})])
    rec = activity_record_from_fitfile(fit, correct_artifacts=True)
    assert rec.rr_ms.size == 2 and rec.metrics is None


def test_rr_quality_counts_artifacts_and_rmssd_shift():
    # one big jump that the artifact filter should drop
    rr = np.array([800.0, 810.0, 805.0, 2000.0, 808.0, 802.0])  # 2000 ms is an artifact
    q = rr_quality(rr, session_seconds=None)
    assert q.n_beats_raw == 6
    assert q.n_artifacts == 1
    assert abs(q.artifact_rate - 1 / 6) < 1e-9
    # raw RMSSD is inflated by the 2000 ms spike; corrected is much smaller
    assert q.rmssd_raw > q.rmssd_corrected
    assert np.isnan(q.coverage)


def test_rr_quality_coverage_uses_corrected_duration():
    # 100 clean beats of ~800 ms -> ~80 s of beats; over a 100 s session -> coverage ~0.8
    rr = np.full(100, 800.0)
    q = rr_quality(rr, session_seconds=100.0)
    assert q.n_artifacts == 0
    assert abs(q.coverage - 0.8) < 1e-9
    assert abs(q.mean_hr - 75.0) < 1e-9  # 60000/800


def test_rr_quality_rejects_too_few_beats():
    with pytest.raises(ValueError, match=">= 2 raw"):
        rr_quality(np.array([800.0]))
