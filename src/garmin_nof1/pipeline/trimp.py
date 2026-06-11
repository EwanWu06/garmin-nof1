"""Training-load (TRIMP) variants from per-session summaries.

The pre-registered primary exposure (OSF §4) is HR-based **Banister TRIMP**, computed
from a session's average HR and duration (no need to integrate the full HR stream, so
2 years of activity FITs need not be downloaded). Robustness variants: **Edwards**
5-zone load (from time-in-zones) and Garmin's own training-load (passthrough). All pure
functions, validated on known values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Banister weighting constants (sex-specific): TRIMP = dur * HRr * a * exp(b * HRr)
_BANISTER = {"M": (0.64, 1.92), "F": (0.86, 1.67)}


def banister_trimp(
    duration_min: float, hr_avg: float, hr_rest: float, hr_max: float, sex: str = "M"
) -> float:
    """Banister TRIMP for one session. ``HRr`` is the heart-rate reserve fraction,
    floored at 0 (a session at/below resting HR contributes no load)."""
    if hr_max <= hr_rest:
        raise ValueError("hr_max must exceed hr_rest")
    hrr = max(0.0, (hr_avg - hr_rest) / (hr_max - hr_rest))
    a, b = _BANISTER[sex.upper()]
    return float(duration_min * hrr * a * math.exp(b * hrr))


def edwards_trimp(time_in_zones_min: Sequence[float]) -> float:
    """Edwards summated heart-rate-zone load: minutes in zone *i* weighted by *i*
    (zones numbered 1..n in order)."""
    return float(sum((i + 1) * t for i, t in enumerate(time_in_zones_min)))


def trimp_variants(
    duration_min: float,
    hr_avg: float,
    hr_rest: float,
    hr_max: float,
    *,
    time_in_zones_min: Sequence[float] | None = None,
    garmin_load: float | None = None,
    sex: str = "M",
) -> dict[str, float]:
    """All available TRIMP variants for one session (Banister always; Edwards / Garmin
    when their inputs are supplied)."""
    out = {"banister": banister_trimp(duration_min, hr_avg, hr_rest, hr_max, sex)}
    if time_in_zones_min is not None:
        out["edwards"] = edwards_trimp(time_in_zones_min)
    if garmin_load is not None:
        out["garmin"] = float(garmin_load)
    return out
