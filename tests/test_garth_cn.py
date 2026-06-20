"""Tests for the garth-backed Garmin China adapter.

The adapter cannot be exercised against the live China API in CI, but its *contract*
— the exact connectapi endpoint paths, query params, pagination, and download path it
builds — is pure and is tested here with a fake garth client. These are the paths
garminconnect itself uses (reconciled from its source), so a drop-in for our ingest.
"""

from garmin_nof1.pipeline.garth_cn import GarthCnApi

ACTIVITIES_PATH = "/activitylist-service/activities/search/activities"


class FakeGarth:
    """Records connectapi/download calls and returns canned responses by path."""

    def __init__(self, responses=None, download_bytes=b"FIT"):
        self.calls = []  # list of (path, params)
        self.downloads = []  # list of paths
        self._responses = responses or {}
        self._download_bytes = download_bytes

    def connectapi(self, path, **kwargs):
        params = kwargs.get("params")
        self.calls.append((path, params))
        resp = self._responses.get(path)
        return resp(params) if callable(resp) else resp

    def download(self, path, **kwargs):
        self.downloads.append(path)
        return self._download_bytes


def _api(fake, display_name="ewan"):
    return GarthCnApi(connectapi=fake.connectapi, download=fake.download, display_name=display_name)


def test_get_hrv_data_builds_dated_path():
    fake = FakeGarth(responses={"/hrv-service/hrv/2024-05-01": {"hrvSummary": {"avg": 55}}})
    out = _api(fake).get_hrv_data("2024-05-01")
    assert out == {"hrvSummary": {"avg": 55}}
    assert fake.calls == [("/hrv-service/hrv/2024-05-01", None)]


def test_get_sleep_data_uses_display_name_and_date_param():
    path = "/wellness-service/wellness/dailySleepData/ewan"
    fake = FakeGarth(responses={path: {"dailySleepDTO": {}}})
    out = _api(fake).get_sleep_data("2024-05-01")
    assert out == {"dailySleepDTO": {}}
    assert fake.calls[0][0] == path
    assert fake.calls[0][1] == {"date": "2024-05-01", "nonSleepBufferMinutes": 60}


def test_get_rhr_day_uses_display_name_and_range_params():
    path = "/userstats-service/wellness/daily/ewan"
    fake = FakeGarth(responses={path: {"allMetrics": {}}})
    out = _api(fake).get_rhr_day("2024-05-01")
    assert out == {"allMetrics": {}}
    assert fake.calls[0][0] == path
    assert fake.calls[0][1] == {"fromDate": "2024-05-01", "untilDate": "2024-05-01", "metricId": 60}


def test_get_activities_by_date_paginates_until_empty():
    def acts(params):
        start = int(params["start"])
        if start == 0:
            return [{"id": i} for i in range(20)]
        if start == 20:
            return [{"id": i} for i in range(20, 25)]
        return []

    fake = FakeGarth(responses={ACTIVITIES_PATH: acts})
    out = _api(fake).get_activities_by_date("2024-01-01", "2024-12-31")
    assert len(out) == 25
    # Three calls: start 0 (full page) -> 20 (partial) -> 40 (empty, stop).
    starts = [c[1]["start"] for c in fake.calls]
    assert starts == ["0", "20", "40"]
    # The date window is forwarded as params.
    assert fake.calls[0][1]["startDate"] == "2024-01-01"
    assert fake.calls[0][1]["endDate"] == "2024-12-31"


def test_get_activities_by_date_omits_enddate_when_not_given():
    fake = FakeGarth(responses={ACTIVITIES_PATH: lambda p: []})
    _api(fake).get_activities_by_date("2024-01-01")
    assert "endDate" not in fake.calls[0][1]


def test_download_activity_builds_fit_path_and_returns_bytes():
    fake = FakeGarth(download_bytes=b"ZIPDATA")
    # dl_fmt is accepted positionally (mirrors garminconnect) but ignored — CN always FIT.
    out = _api(fake).download_activity(12345, "ORIGINAL")
    assert out == b"ZIPDATA"
    assert fake.downloads == ["/download-service/files/activity/12345"]


def test_activity_download_format_enum_has_original():
    # download_activity_fits resolves ActivityDownloadFormat.ORIGINAL; the adapter must expose it.
    assert GarthCnApi.ActivityDownloadFormat.ORIGINAL is not None
