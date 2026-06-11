import numpy as np
import pandas as pd
import pytest

from garmin_nof1.models import fit_recovery_cost
from garmin_nof1.pipeline.build_panel import SCHEMA_COLUMNS, assemble_panel, ln_rmssd_from_rmssd


def test_ln_rmssd_handles_missing():
    assert abs(ln_rmssd_from_rmssd(55.0) - np.log(55.0)) < 1e-9
    assert np.isnan(ln_rmssd_from_rmssd(None))
    assert np.isnan(ln_rmssd_from_rmssd(0.0))


def test_assemble_panel_fills_contiguous_days_and_schema():
    records = [
        {
            "date": "2024-01-01",
            "sport": "triathlon",
            "trimp": 90.0,
            "sleep_hours": 7.5,
            "rhr": 50.0,
            "ln_rmssd": np.log(55.0),
        },
        # gap on 2024-01-02 (no record -> rest, NaN HRV)
        {
            "date": "2024-01-03",
            "sport": "soccer",
            "trimp": 70.0,
            "sleep_hours": 6.9,
            "rhr": 53.0,
            "ln_rmssd": np.log(48.0),
        },
    ]
    df = assemble_panel(records)
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) == 3  # contiguous Jan 1..3
    mid = df.iloc[1]
    assert mid["sport"] == "rest" and mid["trimp"] == 0.0
    assert not mid["hrv_observed"] and np.isnan(mid["ln_rmssd"])
    assert df.iloc[0]["hrv_observed"] and df.iloc[2]["sport"] == "soccer"


def test_assembled_panel_is_drop_in_for_layer_a_model():
    # The whole point of schema alignment: a built panel must flow through the model.
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    records = []
    for i, d in enumerate(dates):
        sport = "soccer" if i % 7 == 0 else ("triathlon" if i % 2 == 0 else "rest")
        trimp = 0.0 if sport == "rest" else float(rng.uniform(40, 120))
        records.append(
            {
                "date": d,
                "sport": sport,
                "trimp": trimp,
                "sleep_hours": float(rng.normal(7.5, 0.4)),
                "rhr": float(rng.normal(50, 2)),
                "ln_rmssd": float(rng.normal(4.0, 0.1)),
            }
        )
    df = assemble_panel(records)
    res = fit_recovery_cost(df)  # must not raise; schema is compatible
    assert "soccer" in res.cost_slope and "triathlon" in res.cost_slope


def test_assemble_panel_rejects_empty_records():
    with pytest.raises(ValueError, match="no records"):
        assemble_panel([])


def test_assemble_panel_rejects_duplicate_dates():
    records = [
        {
            "date": "2024-01-01",
            "sport": "rest",
            "trimp": 0.0,
            "sleep_hours": 7.0,
            "rhr": 50.0,
            "ln_rmssd": np.log(50.0),
        },
        {
            "date": "2024-01-01",
            "sport": "soccer",
            "trimp": 70.0,
            "sleep_hours": 6.5,
            "rhr": 55.0,
            "ln_rmssd": np.log(45.0),
        },
    ]
    with pytest.raises(ValueError, match="duplicate dates"):
        assemble_panel(records)
