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
from dataclasses import dataclass, field

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
    cleaner; the D layer reports how many beats it removed (``n_artifacts_removed``).

    The threshold is *fractional/relative* (not absolute) because a safe window must scale
    with the RR baseline: an 800 ms beat and a 400 ms beat need different absolute
    tolerances for the same physiological deviation to be flagged consistently.
    """
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
    """Short-term HRV metrics reconstructed from RR intervals.

    Fields
    ------
    rmssd : float
        Root mean square of successive RR differences, in milliseconds.
    sdnn : float
        Standard deviation of RR intervals (sample SD), in milliseconds.
    mean_hr : float
        Mean heart rate implied by the RR intervals, in beats per minute (bpm).
    n_beats : int
        Number of beats used to compute the metrics (after artifact removal).
    n_artifacts_removed : int
        Count of beats removed by the artifact filter (0 when
        ``correct_artifacts=False``).
    """

    rmssd: float
    sdnn: float
    mean_hr: float
    n_beats: int
    n_artifacts_removed: int


def metrics_from_rr(rr_ms, *, correct_artifacts: bool = True) -> HrvMetrics:
    """Compute HRV metrics from a sequence of RR intervals, optionally after artifact filtering.

    Parameters
    ----------
    rr_ms:
        RR intervals in **milliseconds**.
    correct_artifacts:
        When ``True`` (default), run :func:`filter_artifacts` before computing
        metrics.  Pass ``False`` to skip filtering, which is useful for auditing
        raw data or comparing corrected vs. uncorrected values.

    Returns
    -------
    HrvMetrics
        Dataclass with ``rmssd``, ``sdnn``, ``mean_hr`` (see field docs for units),
        ``n_beats`` (beats used after filtering), and ``n_artifacts_removed``
        (beats dropped by the filter; 0 when ``correct_artifacts=False``).

    Raises
    ------
    ValueError
        If fewer than 2 beats remain after artifact filtering.
    """
    rr = np.asarray(rr_ms, dtype=float)
    n_raw = int(rr.size)
    rr_use = filter_artifacts(rr) if correct_artifacts else rr
    if rr_use.size < 2:
        raise ValueError(
            f"metrics_from_rr: only {rr_use.size} beat(s) remain after artifact "
            f"filtering ({n_raw - int(rr_use.size)} removed); need >= 2"
        )
    return HrvMetrics(
        rmssd=rmssd(rr_use),
        sdnn=sdnn(rr_use),
        mean_hr=mean_hr(rr_use),
        n_beats=int(rr_use.size),
        n_artifacts_removed=n_raw - int(rr_use.size),
    )


@dataclass(frozen=True)
class RrQuality:
    """Per-session RR data-quality summary (D-layer measurement audit).

    Fields
    ------
    n_beats_raw : int
        RR intervals before artifact filtering.
    n_artifacts : int
        Intervals dropped by :func:`filter_artifacts`.
    artifact_rate : float
        ``n_artifacts / n_beats_raw`` (0.0 when there are no beats).
    rmssd_raw, rmssd_corrected : float
        RMSSD (ms) before and after artifact correction.
    sdnn_corrected : float
        SDNN (ms) after artifact correction.
    mean_hr : float
        Mean heart rate (bpm) from the artifact-corrected RR.
    coverage : float
        Fraction of the session covered by accepted beats — the corrected RR total
        duration divided by ``session_seconds`` (NaN if ``session_seconds`` is None/0).
    """

    n_beats_raw: int
    n_artifacts: int
    artifact_rate: float
    rmssd_raw: float
    rmssd_corrected: float
    sdnn_corrected: float
    mean_hr: float
    coverage: float


def rr_quality(rr_ms, *, session_seconds: float | None = None, threshold: float = 0.2) -> RrQuality:
    """Summarize one session's RR quality: artifact rate and the RMSSD shift it causes.

    Quantifies how much the artifact filter changes the headline metric (``rmssd_raw`` vs
    ``rmssd_corrected``) and how completely the beats cover the session — the measurement
    audit the D layer reports per session. Requires >= 2 raw beats (and >= 2 after
    correction); raises :class:`ValueError` otherwise.
    """
    rr = np.asarray(rr_ms, dtype=float)
    n_raw = int(rr.size)
    if n_raw < 2:
        raise ValueError("rr_quality needs >= 2 raw RR intervals")
    corrected = filter_artifacts(rr, threshold)
    if corrected.size < 2:
        raise ValueError("rr_quality: < 2 beats remain after artifact filtering")
    n_art = n_raw - int(corrected.size)
    if session_seconds:
        coverage = float(np.sum(corrected) / 1000.0 / session_seconds)
    else:
        coverage = float("nan")
    return RrQuality(
        n_beats_raw=n_raw,
        n_artifacts=n_art,
        artifact_rate=float(n_art / n_raw),
        rmssd_raw=rmssd(rr),
        rmssd_corrected=rmssd(corrected),
        sdnn_corrected=sdnn(corrected),
        mean_hr=mean_hr(corrected),
        coverage=coverage,
    )


@dataclass(frozen=True)
class ActivityRecord:
    """Parsed activity: session metadata + reconstructed RR and HRV metrics.

    Equality and hashing are over the scalar metadata fields only (``start_time``,
    ``sport``, ``hr_avg``, ``duration_min``, ``metrics``); the ``rr_ms`` ndarray
    is excluded because ndarray ``__eq__`` returns an array rather than a bool.

    Fields
    ------
    start_time : object | None
        Session start timestamp, as fitparse yields it.
    sport : str | None
        Raw FIT sport key (e.g. ``"running"``, ``"soccer"``).
    hr_avg : int | float | None
        Average heart rate for the session, in bpm. FIT ``avg_heart_rate`` is
        typically an int, but floats are accepted.
    duration_min : float | None
        Session duration in minutes, derived from ``total_timer_time``.
    rr_ms : np.ndarray
        Beat-to-beat R-R intervals in milliseconds (beat order preserved).
        Excluded from equality/hash comparisons.
    metrics : HrvMetrics | None
        HRV metrics computed from ``rr_ms``, or ``None`` when fewer than 2
        usable beats remain after artifact filtering.
    """

    start_time: object | None
    sport: str | None
    hr_avg: int | float | None
    duration_min: float | None
    rr_ms: np.ndarray = field(compare=False, hash=False, repr=False)
    metrics: HrvMetrics | None


def activity_record_from_fitfile(fitfile, *, correct_artifacts: bool = True) -> ActivityRecord:
    """Build an :class:`ActivityRecord` from an opened fitparse-like file.

    Kept separate from disk I/O so it is testable with a fake fitfile — no real
    .fit fixtures needed.

    Parameters
    ----------
    fitfile:
        Opened fitparse-like object exposing ``get_messages(name)`` (returns an
        iterable of message objects) where each message exposes
        ``get_value(field)``.
    correct_artifacts:
        Passed through to :func:`metrics_from_rr`. When ``True`` (default),
        the artifact filter is applied before computing HRV metrics.

    Returns
    -------
    ActivityRecord
        Parsed session metadata, raw RR intervals, and (when available) HRV
        metrics. ``metrics`` is ``None`` when fewer than 2 usable beats remain
        after artifact filtering.
    """
    hrv_arrays = [m.get_value("time") for m in fitfile.get_messages("hrv")]
    rr_ms = reconstruct_rr_ms(hrv_arrays)

    session = next(iter(fitfile.get_messages("session")), None)

    def _sess(f):
        return session.get_value(f) if session is not None else None

    total_s = _sess("total_timer_time")
    duration_min = float(total_s) / 60.0 if total_s is not None else None
    metrics = None
    if rr_ms.size >= 2:
        try:
            metrics = metrics_from_rr(rr_ms, correct_artifacts=correct_artifacts)
        except ValueError:
            # artifact filtering can leave <2 usable beats on very arrhythmic data
            metrics = None
    return ActivityRecord(
        start_time=_sess("start_time"),
        sport=_sess("sport"),
        hr_avg=_sess("avg_heart_rate"),
        duration_min=duration_min,
        rr_ms=rr_ms,
        metrics=metrics,
    )


def parse_activity_fit(path, *, correct_artifacts: bool = True) -> ActivityRecord:
    """Open an activity FIT and parse it. Lazy fitparse import keeps the pure logic
    (and its tests) free of the fitparse dependency.

    Unit pitfalls when reading other FIT fields later: latitude/longitude are
    *semicircles* (value * 180 / 2**31 -> degrees); altitude uses scale 5 / offset 500.
    """
    from fitparse import FitFile  # lazy

    return activity_record_from_fitfile(FitFile(str(path)), correct_artifacts=correct_artifacts)
