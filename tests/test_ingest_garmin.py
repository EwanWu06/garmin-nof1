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
