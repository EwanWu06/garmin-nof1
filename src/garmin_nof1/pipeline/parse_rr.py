"""Activity-FIT RR parser and HRV-metric reconstruction (D-layer core).

The chest-strap activity FITs (3-month window) carry beat-to-beat R-R intervals in
`hrv` messages. fitparse applies the FIT scale (1000) and returns each interval in
*seconds*, with invalid/padding entries (raw 0xFFFF) as ``None``. We concatenate the
per-message `time` arrays, drop ``None``, convert to milliseconds, and compute the
standard short-term HRV metrics — the "true" RMSSD that the D layer compares against
Garmin's derived value.

Pure logic here imports no fitparse, so it is fully testable on synthetic RR; the file
I/O lives in the thin adapter at the bottom (lazy fitparse import).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def reconstruct_rr_ms(
    hrv_time_arrays: Iterable[Sequence[float | None] | float | None],
) -> np.ndarray:
    """Flatten fitparse `hrv`-message `time` values into a flat RR array in milliseconds.

    Each element is one message's `time` value: a list of up to 5 intervals in seconds
    (``None`` for padding), or a scalar, or ``None``. Output preserves beat order.
    """
    rr_s: list[float] = []
    for arr in hrv_time_arrays:
        if arr is None:
            continue
        values = arr if isinstance(arr, (list, tuple, np.ndarray)) else [arr]
        rr_s.extend(float(v) for v in values if v is not None)
    return np.asarray(rr_s, dtype=float) * 1000.0


def rmssd(rr_ms) -> float:
    """Root mean square of successive RR differences (ms)."""
    rr = np.asarray(rr_ms, dtype=float)
    # np.diff of a single value is empty; np.mean([]) would return nan rather than raising.
    if rr.size < 2:
        raise ValueError("need >= 2 RR intervals for RMSSD")
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def sdnn(rr_ms) -> float:
    """Standard deviation of RR intervals (ms, sample SD)."""
    rr = np.asarray(rr_ms, dtype=float)
    # np.diff of a single value is empty; np.mean([]) would return nan rather than raising.
    if rr.size < 2:
        raise ValueError("need >= 2 RR intervals for SDNN")
    return float(np.std(rr, ddof=1))


def mean_hr(rr_ms) -> float:
    """Mean heart rate (bpm) implied by the RR intervals."""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 1:
        raise ValueError("need >= 1 RR interval for mean HR")
    return float(60_000.0 / np.mean(rr))


def filter_artifacts(rr_ms, threshold: float = 0.2) -> np.ndarray:
    """Drop beats whose interval differs from the previous *accepted* beat by more than
    ``threshold`` (fractional) — a simple ectopic/artifact filter. Documented as a crude
    cleaner; the D layer reports how many beats it removed (``n_artifacts_removed``)."""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return rr
    kept = [float(rr[0])]
    for x in rr[1:]:
        if abs(x - kept[-1]) <= threshold * kept[-1]:
            kept.append(float(x))
    return np.asarray(kept, dtype=float)


@dataclass(frozen=True)
class HrvMetrics:
    """Short-term HRV metrics reconstructed from RR intervals (ms)."""

    rmssd: float
    sdnn: float
    mean_hr: float
    n_beats: int
    n_artifacts_removed: int


def metrics_from_rr(rr_ms, *, correct_artifacts: bool = True) -> HrvMetrics:
    """Compute HRV metrics, optionally after the artifact filter."""
    rr = np.asarray(rr_ms, dtype=float)
    n_raw = int(rr.size)
    rr_use = filter_artifacts(rr) if correct_artifacts else rr
    return HrvMetrics(
        rmssd=rmssd(rr_use),
        sdnn=sdnn(rr_use),
        mean_hr=mean_hr(rr_use),
        n_beats=int(rr_use.size),
        n_artifacts_removed=n_raw - int(rr_use.size),
    )
