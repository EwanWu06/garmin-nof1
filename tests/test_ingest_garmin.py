import json
from unittest.mock import Mock

import pytest

from garmin_nof1.pipeline.ingest_garmin import (
    GarminClient,
    GarminConfig,
    archive_raw,
    with_backoff,
)


def test_with_backoff_retries_then_succeeds():
    calls = {"n": 0}
    sleep = Mock()

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limited")
        return 42

    result = with_backoff(flaky, retries=5, sleep=sleep)
    assert result == 42 and calls["n"] == 3
    assert sleep.call_count == 2  # two failures → two sleeps before the third success


def test_with_backoff_reraises_after_exhausting():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        with_backoff(always_fail, retries=3, sleep=lambda *_: None)
    assert calls["n"] == 3  # tried all 3 attempts


def test_archive_raw_writes_json(tmp_path):
    payload = {"calendarDate": "2024-01-01", "lastNightAvg": 45}
    path = archive_raw(payload, "hrv", tmp_path, "2024-01-01")
    assert path.exists()
    assert json.loads(path.read_text()) == payload


def test_client_fetch_uses_injected_api_and_archives(tmp_path):
    class FakeApi:
        def get_hrv_data(self, date_str):
            return {"date": date_str, "lastNightAvg": 50}

    cfg = GarminConfig(email="x@y.z", password="pw", raw_dir=tmp_path)
    client = GarminClient(cfg, api=FakeApi())
    payload = client.fetch_daily_hrv("2024-01-02")
    assert payload["lastNightAvg"] == 50
    assert (tmp_path / "hrv-2024-01-02.json").exists()


def test_from_env_reads_raw_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GARMIN_EMAIL", "x@y.z")
    monkeypatch.setenv("GARMIN_PASSWORD", "pw")
    monkeypatch.setenv("GARMIN_RAW_DIR", str(tmp_path))
    cfg = GarminConfig.from_env()
    assert str(cfg.raw_dir) == str(tmp_path)


def test_fetch_sleep_and_rhr_and_activities_archive(tmp_path):
    class FakeApi:
        def get_sleep_data(self, d):
            return {"dailySleepDTO": {"calendarDate": d, "sleepTimeSeconds": 27000}}

        def get_rhr_day(self, d):
            return {"date": d, "restingHeartRate": 50}

        def get_activities_by_date(self, start, end):
            return [{"startTimeLocal": f"{start} 18:00:00"}]

    cfg = GarminConfig(email="x@y.z", password="pw", raw_dir=tmp_path)
    client = GarminClient(cfg, api=FakeApi())

    assert client.fetch_sleep("2024-05-01")["dailySleepDTO"]["sleepTimeSeconds"] == 27000
    assert (tmp_path / "sleep-2024-05-01.json").exists()
    client.fetch_rhr("2024-05-01")
    assert (tmp_path / "rhr-2024-05-01.json").exists()
    acts = client.fetch_activities("2024-05-01", "2024-05-31")
    assert len(acts) == 1 and (tmp_path / "activities-2024-05-01_2024-05-31.json").exists()


def test_download_activity_fits_writes_files(tmp_path):
    class FakeApi:
        def download_activity(self, activity_id, **kwargs):
            return b"FITDATA-" + str(activity_id).encode()

    cfg = GarminConfig(email="x@y.z", password="pw", raw_dir=tmp_path)
    client = GarminClient(cfg, api=FakeApi())
    paths = client.download_activity_fits([101, 202])
    assert len(paths) == 2
    assert (tmp_path / "activities" / "101.fit").read_bytes() == b"FITDATA-101"
    assert (tmp_path / "activities" / "202.fit").exists()


def test_ingest_range_pulls_each_day_and_activities(tmp_path):
    class FakeApi:
        def get_hrv_data(self, d):
            return {"hrvSummary": {"calendarDate": d, "lastNightAvg": 50}}

        def get_sleep_data(self, d):
            return {"dailySleepDTO": {"calendarDate": d, "sleepTimeSeconds": 27000}}

        def get_rhr_day(self, d):
            return {"date": d, "restingHeartRate": 50}

        def get_activities_by_date(self, start, end):
            return [{"startTimeLocal": f"{start} 18:00:00"}]

    cfg = GarminConfig(email="x@y.z", password="pw", raw_dir=tmp_path)
    client = GarminClient(cfg, api=FakeApi())
    counts = client.ingest_range("2024-05-01", "2024-05-03")
    assert counts == {"hrv": 3, "sleep": 3, "rhr": 3, "activities": 1}
    assert len(list(tmp_path.glob("hrv-*.json"))) == 3
    assert (tmp_path / "activities-2024-05-01_2024-05-03.json").exists()
