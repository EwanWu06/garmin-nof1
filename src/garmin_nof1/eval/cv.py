"""Leakage-safe cross-validation for autocorrelated single-subject daily series.

This module is the methodological spine of the project: on ~700 daily points with
strong serial correlation, ordinary k-fold leaks the future and inflates skill.
What is provided here, and *why* (see the plan's verification of the CV claims):

* ``PurgedWalkForwardSplit`` — forward-chaining (never trains on the future) with
  an **embargo** gap, plus an optional **purge** band. For a one-step-ahead label
  (predict next-day HRV) the embargo is the load-bearing tool; purge only matters
  when the label spans multiple days (e.g. a 7-day rolling recovery outcome), so it
  defaults to 0 and is documented as conditional rather than reflexive.
* ``combinatorial_purged_splits`` / ``n_backtest_paths`` — Combinatorial Purged CV
  (de Prado 2018), used as a **secondary** variance diagnostic. Its many paths share
  the same scarce data, so the spread it reports understates true uncertainty.
* ``effective_sample_size`` — because the real ceiling on power is the number of
  *independent* blocks, not the nominal day count. Report it next to any result.

Symmetric blocked CV (training on data after the test block) is intentionally NOT
provided: for an honest single-subject design you must never train on the future.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def _num_samples(x) -> int:
    if hasattr(x, "shape"):
        return int(x.shape[0])
    return len(x)


class PurgedWalkForwardSplit:
    """Forward-chaining time-series splits with an embargo (and optional purge).

    Parameters
    ----------
    n_splits : int
        Number of test folds (contiguous blocks at the tail of the series).
    test_size : int, optional
        Size of each test block. Defaults to ``n // (n_splits + 1)``.
    embargo : int
        Days of gap between the end of train and the start of test — the
        serial-correlation cooldown. Size it from the residual decorrelation time
        (inspect ACF/PACF), not from a financial default.
    purge : int
        Label horizon in days. Only needed for multi-day-horizon labels, where a
        training sample's label window would overlap the test block; removes that
        many extra days immediately before the test block. Defaults to 0.
    expanding : bool
        Expanding train window (anchored at 0) if True, else a rolling window.
    max_train_size : int, optional
        Cap on train length for the rolling window.

    Notes
    -----
    sklearn-compatible: implements ``split`` and ``get_n_splits`` so it drops into
    ``cross_val_score`` / ``GridSearchCV`` directly.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        embargo: int = 0,
        purge: int = 0,
        expanding: bool = True,
        max_train_size: int | None = None,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if embargo < 0 or purge < 0:
            raise ValueError("embargo and purge must be >= 0")
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo = embargo
        self.purge = purge
        self.expanding = expanding
        self.max_train_size = max_train_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = _num_samples(X)
        indices = np.arange(n)
        test_size = self.test_size or (n // (self.n_splits + 1))
        if test_size < 1:
            raise ValueError("series too short for the requested n_splits")
        gap = self.embargo + self.purge

        for i in range(self.n_splits):
            test_start = n - (self.n_splits - i) * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n
            train_end = test_start - gap
            if train_end <= 0:
                raise ValueError(
                    "not enough history before the first test fold; reduce "
                    "n_splits/test_size/embargo or supply more data"
                )
            if self.expanding or self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)
            yield indices[train_start:train_end], indices[test_start:test_end]


def n_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct backtest paths produced by CPCV (de Prado 2018):
    ``phi = C(N-1, k-1)`` where N = n_groups, k = n_test_groups.

    Bounds match :func:`combinatorial_purged_splits` (strict ``k < N``): with ``k == N`` the
    test set is every group and there is no training data, so it is rejected, not counted."""
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("require 1 <= n_test_groups < n_groups")
    return math.comb(n_groups - 1, n_test_groups - 1)


def combinatorial_purged_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
    purge: int = 0,
):
    """Combinatorial Purged Cross-Validation splits.

    Partition ``0..n_samples-1`` into ``n_groups`` contiguous groups and, for every
    choice of ``n_test_groups`` groups as the test set, build the train set from the
    remaining groups with a purge band of ``purge`` days *before* and an embargo band
    of ``embargo`` days *after* each test group removed.

    Returns a list of ``(train_idx, test_idx)`` arrays of length ``C(n_groups,
    n_test_groups)``. Use ``n_backtest_paths`` for the number of stitched paths.

    Caveat (report it): the paths reuse the same scarce data, so the resulting metric
    distribution is optimistically narrow — treat it as a variance probe, not a
    population of independent backtests.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("require 1 <= n_test_groups < n_groups")
    indices = np.arange(n_samples)
    groups = np.array_split(indices, n_groups)
    bounds = [(int(g[0]), int(g[-1])) for g in groups]

    splits = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[c] for c in combo])
        test_set = set(test_idx.tolist())
        # remove a [a-purge, b+embargo] band around each chosen test group
        blocked = set(test_set)
        for c in combo:
            a, b = bounds[c]
            blocked.update(range(a - purge, b + embargo + 1))
        train_idx = np.array([i for i in indices if i not in blocked], dtype=int)
        splits.append((train_idx, np.sort(test_idx)))
    return splits


def _autocorr(x: np.ndarray, maxlag: int) -> np.ndarray:
    """Biased sample autocorrelation rho_0..rho_maxlag."""
    x = x - x.mean()
    n = x.size
    var = float(np.dot(x, x) / n)
    if var == 0.0:
        return np.zeros(maxlag + 1)
    ac = np.empty(maxlag + 1)
    for k in range(maxlag + 1):
        ac[k] = float(np.dot(x[: n - k], x[k:]) / (n * var))
    return ac


def effective_sample_size(x) -> float:
    """Effective sample size of a serially correlated series.

    ``ESS = m / (1 + 2 * sum_k rho_k)`` with the sum truncated at the first
    **individually** non-positive autocorrelation (initial-positive-lag truncation). This is
    the simpler cousin of Geyer's initial-positive-*sequence*, which truncates on the first
    non-positive *pairwise* sum ``rho_2m + rho_2m+1``; the two coincide for a monotone-positive
    ACF (the case here) but the per-lag rule can stop earlier and slightly over-estimate ESS on
    an oscillating ACF. For an AR(1) series with autocorrelation ``rho`` this approaches
    ``m*(1-rho)/(1+rho)``; for i.i.d. data it approaches ``m``.

    NaNs (non-wear nights) are dropped before estimation — a simplification that
    treats the remaining points as contiguous, adequate for reporting the order of
    magnitude of independent information, not for exact inference.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x.size
    if m < 3:
        raise ValueError("need at least 3 finite observations")
    maxlag = min(m - 1, 400)
    ac = _autocorr(x, maxlag)
    s = 0.0
    for k in range(1, maxlag + 1):
        if ac[k] <= 0.0:
            break
        s += ac[k]
    ess = m / (1.0 + 2.0 * s)
    return float(min(ess, float(m)))
