"""Behaviour spec for the leakage-safe CV scaffold (garmin_nof1.eval.cv).

This is the module every later conclusion is gated on, so it is specified tightly:
no future in train, embargo gap honoured, purge only for multi-day labels, CPCV
path counting, effective-sample-size, and — the headline — a behavioural test that
the scaffold actually *blocks* the adjacency leakage that shuffled k-fold admits.
"""

import math

import numpy as np
import pytest
from sklearn.model_selection import KFold, cross_val_score

from garmin_nof1.eval.cv import (
    PurgedWalkForwardSplit,
    combinatorial_purged_splits,
    effective_sample_size,
    n_backtest_paths,
)


def _ar1(n, phi, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


# ── PurgedWalkForwardSplit ────────────────────────────────────────────────────


def test_walkforward_never_trains_on_the_future():
    cv = PurgedWalkForwardSplit(n_splits=5, embargo=0)
    X = np.zeros((200, 1))
    for train, test in cv.split(X):
        assert train.max() < test.min()  # every train index strictly precedes test


def test_walkforward_embargo_gap_is_respected():
    embargo = 7
    cv = PurgedWalkForwardSplit(n_splits=5, embargo=embargo)
    X = np.zeros((300, 1))
    for train, test in cv.split(X):
        # gap between train end and test start must exceed the embargo
        assert test.min() - train.max() - 1 >= embargo


def test_walkforward_test_blocks_are_disjoint_and_increasing():
    cv = PurgedWalkForwardSplit(n_splits=5, embargo=0)
    X = np.zeros((250, 1))
    tests = [test for _, test in cv.split(X)]
    assert len(tests) == 5
    prev_max = -1
    seen = set()
    for test in tests:
        assert np.all(np.diff(test) == 1)  # contiguous block
        assert test.min() > prev_max  # strictly later than previous fold
        assert seen.isdisjoint(test.tolist())  # no overlap
        seen.update(test.tolist())
        prev_max = test.max()


def test_walkforward_rolling_window_bounds_train_size():
    cv = PurgedWalkForwardSplit(n_splits=4, embargo=0, expanding=False, max_train_size=50)
    X = np.zeros((300, 1))
    for train, _ in cv.split(X):
        assert len(train) <= 50


def test_walkforward_expanding_starts_at_zero():
    cv = PurgedWalkForwardSplit(n_splits=4, embargo=0, expanding=True)
    X = np.zeros((300, 1))
    for train, _ in cv.split(X):
        assert train.min() == 0


def test_purge_removes_extra_band_only_for_multiday_labels():
    X = np.zeros((300, 1))
    embargo, purge = 5, 10
    cv = PurgedWalkForwardSplit(n_splits=4, embargo=embargo, purge=purge)
    for train, test in cv.split(X):
        # train must end at least embargo+purge before the test block
        assert test.min() - train.max() - 1 >= embargo + purge


def test_walkforward_is_sklearn_compatible():
    cv = PurgedWalkForwardSplit(n_splits=5, embargo=3)
    assert cv.get_n_splits() == 5
    X = np.zeros((200, 1))
    y = _ar1(200, 0.5)
    from sklearn.linear_model import LinearRegression

    scores = cross_val_score(LinearRegression(), X + y.reshape(-1, 1), y, cv=cv)
    assert len(scores) == 5


def test_embargo_blocks_adjacency_leakage():
    """Headline: shuffled k-fold lets a 1-NN-in-time model interpolate its
    autocorrelated neighbours and look great; embargoed walk-forward does not.
    The whole point of the scaffold is that the leaked score is the optimistic one.
    """
    from sklearn.neighbors import KNeighborsRegressor

    n = 400
    y = _ar1(n, phi=0.9, seed=1)
    X = np.arange(n).reshape(-1, 1).astype(float)  # feature = time index
    knn = KNeighborsRegressor(n_neighbors=1)

    def rmse(cv):
        s = cross_val_score(knn, X, y, cv=cv, scoring="neg_mean_squared_error")
        return math.sqrt(-s.mean())

    rmse_shuffled = rmse(KFold(n_splits=5, shuffle=True, random_state=0))
    rmse_embargoed = rmse(PurgedWalkForwardSplit(n_splits=5, embargo=10))

    # Leakage makes the shuffled error far smaller than the honest one.
    assert rmse_shuffled < 0.5 * rmse_embargoed


# ── Combinatorial Purged CV (CPCV) ────────────────────────────────────────────


def test_cpcv_yields_all_group_combinations():
    splits = combinatorial_purged_splits(n_samples=600, n_groups=6, n_test_groups=2)
    assert len(splits) == math.comb(6, 2) == 15
    for train, test in splits:
        assert set(train).isdisjoint(set(test))


def test_cpcv_backtest_path_count():
    # de Prado: number of backtest paths phi = C(N-1, k-1)
    assert n_backtest_paths(6, 2) == 5  # C(5,1)
    assert n_backtest_paths(6, 3) == 10  # C(5,2)
    assert n_backtest_paths(5, 1) == 1  # C(4,0): k=1 stitches one path


def test_cpcv_purge_embargo_clears_band_around_test_groups():
    embargo, purge = 4, 6
    splits = combinatorial_purged_splits(
        n_samples=600, n_groups=6, n_test_groups=2, embargo=embargo, purge=purge
    )
    for train, test in splits:
        test_set = set(test.tolist())
        for ti in train:
            # no training index may sit inside [a-purge, b+embargo] of a test block
            assert ti not in test_set
            near = any((ti >= t - purge) and (ti <= t + embargo) for t in test_set)
            assert not near


# ── effective_sample_size ─────────────────────────────────────────────────────


def test_ess_iid_is_close_to_n():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert 0.8 * len(x) <= effective_sample_size(x) <= len(x) + 1


def test_ess_autocorrelated_is_far_below_n():
    x = _ar1(2000, phi=0.9, seed=0)
    ess = effective_sample_size(x)
    assert ess < 0.25 * len(x)  # strong autocorrelation shrinks ESS
    # ballpark AR(1) target: N*(1-phi)/(1+phi)
    target = len(x) * (1 - 0.9) / (1 + 0.9)
    assert 0.4 * target < ess < 3.0 * target


def test_ess_handles_nan():
    x = _ar1(1000, phi=0.5, seed=2)
    x[::13] = np.nan  # mimic non-wear nights
    ess = effective_sample_size(x)
    assert np.isfinite(ess)
    assert 0 < ess <= np.isfinite(x).sum()


def test_ess_rejects_too_short():
    with pytest.raises(ValueError):
        effective_sample_size(np.array([1.0, 2.0]))
