"""Agreement statistics for the D-layer measurement validation.

Standard method-comparison statistics — Bland-Altman bias and 95% limits of agreement,
ICC(2,1) two-way-random absolute-agreement, MAPE, and Lin's concordance correlation
coefficient (CCC) — used to quantify how closely a *test* series reproduces a *reference*
series. Pure NumPy; every statistic is verified against a hand-computed fixture in the
tests (``tests/test_agreement.py``).

Convention: throughout, ``difference = test - reference`` (so a positive Bland-Altman bias
means the test method reads high).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_pair(reference, test) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=float)
    tst = np.asarray(test, dtype=float)
    if ref.shape != tst.shape:
        raise ValueError("reference and test must have the same length")
    if ref.size < 2:
        raise ValueError("need at least 2 paired observations")
    return ref, tst


@dataclass(frozen=True)
class BlandAltman:
    """Bland-Altman summary (differences are ``test - reference``).

    Attributes
    ----------
    bias : float
        Mean difference (systematic offset of test vs reference).
    sd_diff : float
        Sample standard deviation (ddof=1) of the differences.
    loa_lower, loa_upper : float
        95% limits of agreement, ``bias ± 1.96 · sd_diff``.
    n : int
        Number of paired observations.
    """

    bias: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    n: int


def bland_altman(reference, test) -> BlandAltman:
    """Bland-Altman bias and 95% limits of agreement for ``test`` vs ``reference``."""
    ref, tst = _as_pair(reference, test)
    diff = tst - ref
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return BlandAltman(
        bias=bias, sd_diff=sd, loa_lower=bias - 1.96 * sd, loa_upper=bias + 1.96 * sd, n=ref.size
    )


def mape(reference, test) -> float:
    """Mean absolute percentage error of ``test`` against ``reference`` (percent).

    The reference is the denominator; entries where ``reference == 0`` are excluded
    (an undefined percentage error)."""
    ref, tst = _as_pair(reference, test)
    nz = ref != 0.0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((tst[nz] - ref[nz]) / ref[nz])) * 100.0)


def icc_2_1(reference, test) -> float:
    """ICC(2,1): two-way random-effects, single-rater, **absolute-agreement** ICC.

    Treats reference and test as two raters scoring the same ``n`` subjects, and (unlike a
    consistency ICC or Pearson r) penalizes systematic offset between them. Uses the
    Shrout-Fleiss two-way ANOVA decomposition.
    """
    ref, tst = _as_pair(reference, test)
    data = np.column_stack([ref, tst])  # n subjects x k=2 raters
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((data - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)
    if denom == 0:
        return 1.0  # zero total variance -> identical constant series, perfect agreement
    return float((ms_rows - ms_err) / denom)


def ccc(reference, test) -> float:
    """Lin's concordance correlation coefficient between ``reference`` and ``test``.

    Combines precision (correlation) and accuracy (closeness to the 45° line) into one
    [-1, 1] agreement index. Population (ddof=0) variances/covariance, per Lin (1989).
    """
    ref, tst = _as_pair(reference, test)
    mx, my = ref.mean(), tst.mean()
    vx = np.mean((ref - mx) ** 2)
    vy = np.mean((tst - my) ** 2)
    cov = np.mean((ref - mx) * (tst - my))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return 1.0
    return float(2.0 * cov / denom)


@dataclass(frozen=True)
class AgreementResult:
    """Bundle of all D-layer agreement statistics for one test-vs-reference comparison."""

    n: int
    bias: float
    sd_diff: float
    loa_lower: float
    loa_upper: float
    mape: float
    icc: float
    ccc: float


def agreement(reference, test) -> AgreementResult:
    """Compute the full agreement panel (Bland-Altman + MAPE + ICC(2,1) + CCC) at once."""
    ba = bland_altman(reference, test)
    return AgreementResult(
        n=ba.n,
        bias=ba.bias,
        sd_diff=ba.sd_diff,
        loa_lower=ba.loa_lower,
        loa_upper=ba.loa_upper,
        mape=mape(reference, test),
        icc=icc_2_1(reference, test),
        ccc=ccc(reference, test),
    )
