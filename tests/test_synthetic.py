"""Lock the synthetic substrate: deterministic, well-formed, carries the effect."""

import numpy as np

from garmin_nof1.data.synthetic import GroundTruth, make_daily_panel, make_rr_series


def test_panel_is_deterministic_given_seed():
    a = make_daily_panel(n_days=200, seed=42)
    b = make_daily_panel(n_days=200, seed=42)
    # ln_rmssd has NaNs; compare with equal_nan semantics via fillna sentinel.
    assert a.drop(columns="sport").fillna(-999).equals(b.drop(columns="sport").fillna(-999))
    assert (a["sport"].astype(str).values == b["sport"].astype(str).values).all()


def test_panel_shape_and_columns():
    df = make_daily_panel(n_days=365, seed=0)
    assert len(df) == 365
    for col in ["date", "sport", "trimp", "sleep_hours", "rhr", "ln_rmssd", "hrv_observed"]:
        assert col in df.columns
    # rest days carry zero load; sport-labelled days carry positive load.
    assert (df.loc[df["sport"] == "rest", "trimp"] == 0).all()
    assert (df.loc[df["sport"] == "triathlon", "trimp"] > 0).all()


def test_missingness_flag_matches_nan():
    df = make_daily_panel(n_days=400, seed=1, missing_rate=0.1)
    assert (df.loc[~df["hrv_observed"], "ln_rmssd"].isna()).all()
    assert (df.loc[df["hrv_observed"], "ln_rmssd"].notna()).all()
    assert 0 < (~df["hrv_observed"]).sum() < len(df)


def test_ground_truth_tau_matches_phi():
    gt = GroundTruth(phi=np.exp(-1 / 3.0))  # tau should be exactly 3 days
    assert abs(gt.tau_days - 3.0) < 1e-9
    assert gt.beta("soccer") > gt.beta("triathlon") > gt.beta("rest") == 0.0


def test_rr_series_has_target_rmssd():
    rr = make_rr_series(n_beats=2000, rmssd_target=45.0, seed=0)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    assert 35.0 < rmssd < 55.0  # within ~20% of target
    assert rr.min() >= 300.0 and rr.max() <= 2000.0
