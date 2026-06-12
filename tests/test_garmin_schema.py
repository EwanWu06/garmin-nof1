import numpy as np

from garmin_nof1.pipeline.garmin_schema import extract_hrv, extract_rhr, extract_sleep


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
