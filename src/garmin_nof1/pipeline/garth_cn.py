"""garth-backed adapter for Garmin **China** (connect.garmin.cn).

Why this exists: the installed ``garminconnect`` (0.3.2) cannot authenticate data calls
for Garmin China — its DI-token exchange is hardcoded to global ``.com`` hosts, so China
logins fall back to a JWT_WEB cookie that ``connectapi.garmin.cn`` rejects with 403.
``garth`` *does* support China (``garth.configure(domain="garmin.cn")`` then OAuth1/OAuth2
Bearer auth, which the China backend accepts) and auto-refreshes/persists tokens.

This adapter exposes exactly the method surface ``GarminClient`` (in
:mod:`garmin_nof1.pipeline.ingest_garmin`) calls, delegating to garth's ``connectapi`` /
``download``. The endpoint paths and query params are reconciled from garminconnect's own
source so the two backends are interchangeable — ``GarminClient`` neither knows nor cares
which one it was handed.

The HTTP plumbing is injected (``connectapi`` / ``download`` callables), so the path /
param / pagination contract is unit-tested with a fake; only the live wiring (in
``scripts/pull_garmin.py``) touches the real garth client.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

# Endpoint paths — identical to garminconnect's, so this is a drop-in for our ingest.
_HRV_URL = "/hrv-service/hrv"
_SLEEP_URL = "/wellness-service/wellness/dailySleepData"
_RHR_URL = "/userstats-service/wellness/daily"
_ACTIVITIES_URL = "/activitylist-service/activities/search/activities"
_FIT_DOWNLOAD_URL = "/download-service/files/activity"
_ACTIVITIES_PAGE = 20  # Garmin's web UI fetches 20 at a time; we mimic it and paginate.


class GarthCnApi:
    """garminconnect-compatible facade over a garth China client.

    ``connectapi(path, params=...)`` returns parsed JSON; ``download(path)`` returns raw
    bytes. ``display_name`` is the account's Garmin display name (needed by the sleep and
    resting-HR endpoints), fetched once at wiring time.
    """

    class ActivityDownloadFormat(Enum):
        """Mirrors garminconnect's enum so ``download_activity_fits`` can resolve ORIGINAL.

        Only ORIGINAL (the ``.fit``-bearing zip) is meaningful here — it is the sole format
        carrying the beat-to-beat RR the D-layer needs.
        """

        ORIGINAL = 1

    def __init__(
        self,
        *,
        connectapi: Callable[..., Any],
        download: Callable[..., bytes],
        display_name: str,
    ) -> None:
        self._connectapi = connectapi
        self._download = download
        self.display_name = display_name

    def get_hrv_data(self, cdate: str) -> Any:
        """Nightly HRV-status summary for ``cdate`` (YYYY-MM-DD)."""
        return self._connectapi(f"{_HRV_URL}/{cdate}")

    def get_sleep_data(self, cdate: str) -> Any:
        """Daily sleep summary for ``cdate`` (keyed by display name, like garminconnect)."""
        return self._connectapi(
            f"{_SLEEP_URL}/{self.display_name}",
            params={"date": cdate, "nonSleepBufferMinutes": 60},
        )

    def get_rhr_day(self, cdate: str) -> Any:
        """Resting-heart-rate summary for the single day ``cdate``."""
        return self._connectapi(
            f"{_RHR_URL}/{self.display_name}",
            params={"fromDate": cdate, "untilDate": cdate, "metricId": 60},
        )

    def get_activities_by_date(self, startdate: str, enddate: str | None = None) -> list[dict]:
        """All activities in ``[startdate, enddate]``, paginating until a short/empty page."""
        activities: list[dict] = []
        start = 0
        while True:
            params: dict[str, str] = {
                "startDate": startdate,
                "start": str(start),
                "limit": str(_ACTIVITIES_PAGE),
            }
            if enddate:
                params["endDate"] = enddate
            page = self._connectapi(_ACTIVITIES_URL, params=params)
            if not page:
                break
            activities.extend(page)
            start += _ACTIVITIES_PAGE
        return activities

    def download_activity(self, activity_id: Any, dl_fmt: Any = None) -> bytes:
        """Download an activity's ORIGINAL export (the ``.fit`` zip) as raw bytes.

        ``dl_fmt`` is accepted to match garminconnect's signature but ignored: the China
        FIT-download endpoint always yields the ORIGINAL zip, the only format we use.
        """
        return self._download(f"{_FIT_DOWNLOAD_URL}/{activity_id}")
