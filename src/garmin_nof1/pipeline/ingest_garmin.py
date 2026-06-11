"""Credential-gated Garmin Connect ingest (you run this; it needs your login).

Pulls 2 years of daily summaries + 3 months of activity FITs and **archives every raw
response locally** (`data/raw/`, gitignored) so the panel can be rebuilt offline and
the pipeline is reproducible without re-hitting the API. Since 2026-03 Garmin tightened
auth (Cloudflare TLS fingerprinting + 429s), so requests go through a curl_cffi session
with exponential backoff.

This module is the bridge to your *real* data and cannot be unit-tested against the live
API. The testable parts — backoff and archiving — are pure and injected; ``GarminClient``
takes an injectable ``api`` so its wiring is tested with a fake.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def with_backoff(
    fn,
    *,
    retries=5,
    base_delay=1.0,
    max_delay=60.0,
    sleep=time.sleep,
    exceptions=(Exception,),
):
    """Call ``fn()`` with exponential backoff on ``exceptions``; re-raise after
    ``retries`` attempts. ``sleep`` is injected so tests run instantly."""
    delay = base_delay
    for attempt in range(retries):
        try:
            return fn()
        except exceptions:
            if attempt == retries - 1:
                raise
            sleep(min(delay, max_delay))
            delay *= 2.0
    raise RuntimeError("unreachable")  # pragma: no cover


def archive_raw(payload, name: str, raw_dir, date_str: str) -> Path:
    """Write ``payload`` (JSON-serializable) to ``<raw_dir>/<name>-<date_str>.json``.

    Archiving every raw API response means the processing pipeline can be re-run
    entirely offline and the exact data returned by Garmin Connect is preserved for
    audit and reproducibility — matching the rationale in the module docstring.

    The filename is built directly from ``name`` and ``date_str``.  This is
    intentional: the module is a single-user personal tool with no untrusted-input
    surface, so there is no path-traversal risk worth guarding against here.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}-{date_str}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


@dataclass
class GarminConfig:
    """Credentials + archive location. Load secrets from the environment (.env)."""

    email: str
    password: str
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))

    @classmethod
    def from_env(cls) -> GarminConfig:
        return cls(email=os.environ["GARMIN_EMAIL"], password=os.environ["GARMIN_PASSWORD"])


class GarminClient:
    """Thin wrapper: each fetch goes through backoff and archives its raw response.

    ``api`` is injected in tests; in real use it is lazily built from garminconnect
    (which the user installs and logs in with their own credentials).
    """

    def __init__(self, config: GarminConfig, api=None):
        self.config = config
        self._api = api

    def _ensure_api(self):
        if self._api is None:  # pragma: no cover - requires real credentials
            from garminconnect import Garmin

            self._api = Garmin(self.config.email, self.config.password)
            self._api.login()
        return self._api

    def fetch_daily_hrv(self, date_str: str) -> dict:
        api = self._ensure_api()
        payload = with_backoff(lambda: api.get_hrv_data(date_str))
        archive_raw(payload, "hrv", self.config.raw_dir, date_str)
        return payload
