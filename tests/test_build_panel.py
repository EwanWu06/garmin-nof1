import numpy as np
import pandas as pd
import pytest

from garmin_nof1.models import fit_recovery_cost, fit_recovery_tau
from garmin_nof1.pipeline.build_panel import (
    SCHEMA_COLUMNS,
    assemble_panel,
    classify_missingness,
    daily_sport_and_trimp,
    ln_rmssd_from_rmssd,
    map_sport,
    missingness_diagnostic,
)
from garmin_nof1.pipeline.trimp import banister_trimp


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
    # "drop-in for the Layer-A models" is plural: the tau estimator must also accept it.
    tau = fit_recovery_tau(df)
    assert "soccer" in tau.tau and "triathlon" in tau.tau


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


def test_map_sport_groups_endurance_and_soccer():
    assert map_sport("running") == "triathlon"
    assert map_sport("lap_swimming") == "triathlon"
    assert map_sport("road_biking") == "triathlon"
    assert map_sport("soccer") == "soccer"
    assert map_sport("strength_training") == "rest"  # unmodeled -> no session


def test_daily_fold_priority_soccer_and_sums_trimp():
    activities = [
        {"date": "2024-05-01", "sport_key": "running", "hr_avg": 150, "duration_min": 60},
        {"date": "2024-05-01", "sport_key": "soccer", "hr_avg": 160, "duration_min": 90},
        {"date": "2024-05-02", "sport_key": "strength_training", "hr_avg": 110, "duration_min": 40},
    ]
    folded = daily_sport_and_trimp(activities, hr_rest=50, hr_max=200, sex="M")
    d1 = folded["2024-05-01"]
    assert d1["sport"] == "soccer"  # soccer wins the day's label
    expected = banister_trimp(60, 150, 50, 200, "M") + banister_trimp(90, 160, 50, 200, "M")
    assert abs(d1["trimp"] - expected) < 1e-9
    d2 = folded["2024-05-02"]
    assert d2["sport"] == "rest" and d2["trimp"] == 0.0  # only unmodeled activity


def test_classify_missingness_flags_nonwear():
    df = assemble_panel(
        [
            {
                "date": "2024-01-01",
                "sport": "rest",
                "trimp": 0.0,
                "sleep_hours": 7.5,
                "rhr": 50.0,
                "ln_rmssd": np.log(55.0),
            },
            {
                "date": "2024-01-02",
                "sport": "rest",
                "trimp": 0.0,
                "sleep_hours": np.nan,
                "rhr": np.nan,
                "ln_rmssd": np.nan,
            },
        ]
    )
    out = classify_missingness(df)
    assert out["ln_rmssd_missing"].tolist() == [False, True]
    assert out["rhr_missing"].tolist() == [False, True]
    assert out["missing_any"].tolist() == [False, True]


def test_missingness_diagnostic_mcar_does_not_cry_wolf():
    # missingness independent of trimp (MCAR) -> the diagnostic must NOT flag MAR
    rng = np.random.default_rng(2)
    recs = []
    for i in range(300):
        trimp = float(rng.uniform(0, 200))
        ln = np.nan if rng.random() < 0.1 else float(rng.normal(4.0, 0.1))
        recs.append(
            {
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                "sport": "triathlon",
                "trimp": trimp,
                "sleep_hours": 7.5,
                "rhr": 50.0,
                "ln_rmssd": ln,
            }
        )
    df = assemble_panel(recs)
    diag = missingness_diagnostic(df, predictor="trimp", target="ln_rmssd")
    assert diag["suspect_mar"] is False


def test_missingness_diagnostic_detects_load_dependence():
    # Construct missingness that depends on trimp (MAR, not MCAR): hard days drop HRV.
    rng = np.random.default_rng(1)
    recs = []
    for i in range(200):
        trimp = float(rng.uniform(0, 200))
        ln = np.nan if trimp > 150 else float(rng.normal(4.0, 0.1))  # missing on hard days
        recs.append(
            {
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                "sport": "triathlon",
                "trimp": trimp,
                "sleep_hours": 7.5,
                "rhr": 50.0,
                "ln_rmssd": ln,
            }
        )
    df = assemble_panel(recs)
    diag = missingness_diagnostic(df, predictor="trimp", target="ln_rmssd")
    assert diag["suspect_mar"] is True
    assert diag["mean_when_missing"] > diag["mean_when_present"]
