"""Assemble archived Garmin exports into a tidy, missingness-aware daily panel.

The output schema is **identical to the synthetic panel** (`garmin_nof1.data.synthetic`)
— `date, sport, trimp, sleep_hours, rhr, ln_rmssd, hrv_observed` — so real data is a
drop-in for the already-validated Layer-A models. Days with no record are filled as
contiguous rest days with NaN HRV (non-wear), so the calendar has no holes.

The raw-dict -> record extractors carry an *assumed* Garmin Connect schema (documented
inline); they are reconciled when the user first runs ingest on real data. Everything
here is pure (operates on dicts/DataFrames) and TDD'd on synthetic fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from garmin_nof1.pipeline.trimp import banister_trimp

SCHEMA_COLUMNS = ["date", "sport", "trimp", "sleep_hours", "rhr", "ln_rmssd", "hrv_observed"]
_SPORT_CATEGORIES = ["rest", "triathlon", "soccer"]


def ln_rmssd_from_rmssd(rmssd_value) -> float:
    """ln(RMSSD), or NaN for missing/non-positive input (non-wear nights)."""
    if rmssd_value is None or rmssd_value <= 0:
        return float("nan")
    return float(np.log(rmssd_value))


def assemble_panel(records: Iterable[dict], *, start=None, end=None) -> pd.DataFrame:
    """Build the tidy daily panel from per-day records on a CONTIGUOUS date range.

    Each record is a dict with keys ``date`` and any of ``sport, trimp, sleep_hours,
    rhr, ln_rmssd`` (missing -> NaN). Missing days become rest days with zero load and
    NaN HRV; ``hrv_observed`` flags ln_rmssd presence.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("assemble_panel received no records; provide at least one day.")
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        first_dup = df.loc[df["date"].duplicated(), "date"].iloc[0].date()
        raise ValueError(
            f"assemble_panel received duplicate dates (first: {first_dup}); "
            "deduplicate before calling."
        )
    lo = df["date"].min() if start is None else pd.Timestamp(start)
    hi = df["date"].max() if end is None else pd.Timestamp(end)
    full = pd.date_range(lo, hi, freq="D")

    df = df.set_index("date").reindex(full)
    df.index.name = "date"
    df = df.reset_index()

    for col in ["trimp", "sleep_hours", "rhr", "ln_rmssd"]:
        if col not in df.columns:
            df[col] = np.nan
    df["sport"] = df["sport"].fillna("rest") if "sport" in df.columns else "rest"
    df["trimp"] = df["trimp"].fillna(0.0)
    # Any sport value not in _SPORT_CATEGORIES is coerced to NaN by Categorical;
    # in normal flow only rest/triathlon/soccer reach here (Task 6's sport mapping guarantees it).
    df["sport"] = pd.Categorical(df["sport"], categories=_SPORT_CATEGORIES)
    df["hrv_observed"] = df["ln_rmssd"].notna()
    return df[SCHEMA_COLUMNS]


# Garmin activityType.typeKey -> modeled sport. Endurance triathlon disciplines collapse
# to "triathlon"; soccer/football to "soccer"; everything else is an unmodeled session
# that contributes no triathlon/soccer load (treated as "rest" for the panel label).
_SPORT_MAP = {
    "running": "triathlon",
    "track_running": "triathlon",
    "trail_running": "triathlon",
    "treadmill_running": "triathlon",
    "cycling": "triathlon",
    "road_biking": "triathlon",
    "indoor_cycling": "triathlon",
    "mountain_biking": "triathlon",
    "virtual_ride": "triathlon",
    "swimming": "triathlon",
    "lap_swimming": "triathlon",
    "open_water_swimming": "triathlon",
    "soccer": "soccer",
    "football": "soccer",
}


def map_sport(activity_type_key) -> str:
    """Map a Garmin activityType key to {triathlon, soccer, rest}.

    Keys are lowercased before lookup so mixed-case export values (e.g. "Running",
    "SOCCER") are tolerated. Anything not found in the map is treated as an unmodeled
    session ("rest") that contributes no label priority or TRIMP load.
    """
    return _SPORT_MAP.get(str(activity_type_key).lower(), "rest")


def daily_sport_and_trimp(
    activities: Iterable[dict], hr_rest: float, hr_max: float, sex: str = "M"
) -> dict[str, dict]:
    """Fold per-activity summaries into per-day ``{date: {"sport", "trimp"}}``.

    The day's label is the highest-priority modeled sport present (soccer > triathlon >
    rest); ``trimp`` sums Banister TRIMP over the day's modeled (triathlon/soccer)
    sessions. Each activity dict needs ``date, sport_key, hr_avg, duration_min``.

    TRIMP is summed across ALL modeled sessions regardless of which sport wins the label
    (a "bricks" day — e.g. run + soccer — contributes load from both).
    """
    priority = {"rest": 0, "triathlon": 1, "soccer": 2}
    out: dict[str, dict] = {}
    for act in activities:
        date = str(act["date"])
        sport = map_sport(act["sport_key"])
        day = out.setdefault(date, {"sport": "rest", "trimp": 0.0})
        if priority[sport] > priority[day["sport"]]:
            day["sport"] = sport
        if sport != "rest":
            day["trimp"] += banister_trimp(act["duration_min"], act["hr_avg"], hr_rest, hr_max, sex)
    return out


def classify_missingness(df: pd.DataFrame, *, columns=("ln_rmssd", "rhr")) -> pd.DataFrame:
    """Annotate non-wear / missing days: add ``<col>_missing`` booleans and a
    ``missing_any`` flag. Labeling only — the MCAR-vs-MAR probe is separate."""
    out = df.copy()
    for c in columns:
        out[f"{c}_missing"] = out[c].isna()
    flags = [f"{c}_missing" for c in columns]
    out["missing_any"] = out[flags].any(axis=1)
    return out


def missingness_diagnostic(df: pd.DataFrame, *, predictor="trimp", target="ln_rmssd") -> dict:
    """Crude MCAR probe: does ``target`` missingness depend on ``predictor``? Compares the
    predictor's mean on missing vs present days; flags ``suspect_mar`` when the gap
    exceeds half a standard deviation. A signal that missingness may be MAR (not
    completely at random) — not a formal test."""
    miss = df[target].isna()
    if miss.sum() == 0 or (~miss).sum() == 0:
        return {
            "mean_when_missing": float("nan"),
            "mean_when_present": float("nan"),
            "suspect_mar": False,
        }
    a = float(df.loc[miss, predictor].mean())
    b = float(df.loc[~miss, predictor].mean())
    return {
        "mean_when_missing": a,
        "mean_when_present": b,
        "suspect_mar": bool(abs(a - b) > 0.5 * df[predictor].std()),
    }
