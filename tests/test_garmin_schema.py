import numpy as np

from garmin_nof1.pipeline.garmin_schema import (
    activity_to_session,
    extract_hrv,
    extract_rhr,
    extract_sleep,
)


def test_extract_hrv_pulls_date_and_nightly_rmssd():
    resp = {"hrvSummary": {"calendarDate": "2024-05-01", "lastNightAvg": 55}}
    date, fields = extract_hrv(resp)
    assert date == "2024-05-01" and fields["rmssd"] == 55


def test_extract_hrv_missing_summary_returns_none():
    assert extract_hrv({"foo": 1}) is None


def test_extract_hrv_present_date_missing_value_keeps_date():
    resp = {"hrvSummary": {"calendarDate": "2024-05-02"}}  # no lastNightAvg
    date, fields = extract_hrv(resp)
    assert date == "2024-05-02" and fields["rmssd"] is None


def test_extract_sleep_converts_seconds_to_hours():
    resp = {"dailySleepDTO": {"calendarDate": "2024-05-01", "sleepTimeSeconds": 27000}}
    date, fields = extract_sleep(resp)
    assert date == "2024-05-01" and abs(fields["sleep_hours"] - 7.5) < 1e-9


def test_extract_sleep_missing_seconds_is_nan():
    resp = {"dailySleepDTO": {"calendarDate": "2024-05-03"}}
    date, fields = extract_sleep(resp)
    assert date == "2024-05-03" and np.isnan(fields["sleep_hours"])


def test_extract_rhr_pulls_first_metric_value():
    resp = {
        "allMetrics": {
            "metricsMap": {
                "WELLNESS_RESTING_HEART_RATE": [{"value": 50, "calendarDate": "2024-05-01"}]
            }
        }
    }
    date, fields = extract_rhr(resp)
    assert date == "2024-05-01" and fields["rhr"] == 50


def test_extract_rhr_empty_series_returns_none():
    resp = {"allMetrics": {"metricsMap": {"WELLNESS_RESTING_HEART_RATE": []}}}
    assert extract_rhr(resp) is None


def test_extract_sleep_missing_dto_returns_none():
    assert extract_sleep({}) is None


def test_extract_rhr_tolerates_drifted_series_shapes():
    # series as a dict, or first item not a dict, or first item missing date -> None (no crash)
    mm = "WELLNESS_RESTING_HEART_RATE"
    assert extract_rhr({"allMetrics": {"metricsMap": {mm: {"x": 1}}}}) is None
    assert extract_rhr({"allMetrics": {"metricsMap": {mm: [42]}}}) is None
    assert extract_rhr({"allMetrics": {"metricsMap": {mm: [{"value": 50}]}}}) is None


def test_activity_to_session_maps_fields_and_date():
    act = {
        "startTimeLocal": "2024-05-01 18:00:00",
        "activityType": {"typeKey": "soccer"},
        "averageHR": 160,
        "duration": 5400,
    }
    s = activity_to_session(act)
    assert s == {"date": "2024-05-01", "sport_key": "soccer", "hr_avg": 160, "duration_min": 90.0}


def test_activity_to_session_tolerates_iso_start():
    act = {
        "startTimeLocal": "2024-05-02T07:30:00",
        "activityType": {"typeKey": "running"},
        "averageHR": 140,
        "duration": 1800,
    }
    assert activity_to_session(act)["date"] == "2024-05-02"


def test_activity_to_session_none_when_hr_or_duration_missing():
    base = {"startTimeLocal": "2024-05-01 18:00:00", "activityType": {"typeKey": "soccer"}}
    assert activity_to_session({**base, "duration": 5400}) is None  # no averageHR
    assert activity_to_session({**base, "averageHR": 160}) is None  # no duration
