"""Adapters from archived Garmin Connect JSON into tidy per-day records.

This is the ONE file that encodes the *assumed* shape of Garmin Connect API responses
(as returned by the ``garminconnect`` client). The exact keys are a best-effort guess,
documented inline; they are **reconciled against real responses the first time you run
ingest**. Everything here is a pure function over dicts — tested with hand-built fixtures,
never real data — and imports nothing from garminconnect or the rest of the pipeline.
"""

from __future__ import annotations


def _get(d, *path, default=None):
    """Safe nested lookup: returns ``default`` if any key is missing or a level is not a
    dict / is None. Keeps the adapters tolerant of the schema drift we expect on real data."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or cur.get(key) is None:
            return default
        cur = cur[key]
    return cur


def extract_hrv(resp: dict):
    """``get_hrv_data`` response -> ``(date, {"rmssd": nightly_avg})`` or None.

    Assumed schema: ``resp["hrvSummary"]["calendarDate"]`` and ``["lastNightAvg"]`` (the
    nightly Garmin-derived RMSSD summary, ms). Returns None only when the date is absent;
    a present date with a missing value yields ``rmssd=None`` (a non-wear night)."""
    date = _get(resp, "hrvSummary", "calendarDate")
    if date is None:
        return None
    return (str(date), {"rmssd": _get(resp, "hrvSummary", "lastNightAvg")})


def extract_sleep(resp: dict):
    """``get_sleep_data`` response -> ``(date, {"sleep_hours": hours})`` or None.

    Assumed schema: ``resp["dailySleepDTO"]["calendarDate"]`` and ``["sleepTimeSeconds"]``."""
    date = _get(resp, "dailySleepDTO", "calendarDate")
    if date is None:
        return None
    secs = _get(resp, "dailySleepDTO", "sleepTimeSeconds")
    hours = float(secs) / 3600.0 if secs is not None else float("nan")
    return (str(date), {"sleep_hours": hours})


def extract_rhr(resp: dict):
    """``get_rhr_day`` response -> ``(date, {"rhr": value})`` or None.

    Assumed schema: ``resp["allMetrics"]["metricsMap"]["WELLNESS_RESTING_HEART_RATE"]`` is a
    list whose first item has ``{"value", "calendarDate"}``."""
    series = _get(resp, "allMetrics", "metricsMap", "WELLNESS_RESTING_HEART_RATE")
    if not series:
        return None
    first = series[0]
    date = first.get("calendarDate")
    if date is None:
        return None
    return (str(date), {"rhr": first.get("value")})
