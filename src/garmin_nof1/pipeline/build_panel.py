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
