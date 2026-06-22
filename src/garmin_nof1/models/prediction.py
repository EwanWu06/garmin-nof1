"""Demoted prediction layer (H-P1): leakage-safe next-day-lnRMSSD skill test.

What this answers
-----------------
Do **cross-sport load features** add *incremental* one-step-ahead lnRMSSD skill **over an
AR(1) / random-walk baseline**, for this one person? The pre-registered prior (OSF §3) is
that the increment is small or null. This is the project's falsification layer: it is built
to *fail to reject* the baseline unless real predictive signal exists.

The honest reason it might NOT be null here: the A-layer alignment finding (load on day *t*
suppresses the *next* night's HRV, which Garmin timestamps to day *t+1*) means today's load
genuinely carries next-morning information — so a small positive increment is plausible and,
if present, is reported.

Design
------
For each day *t* with both its night and the next night observed, predict ``lnRMSSD[t+1]``
from information available at the end of day *t*:

* **random_walk** — ``ŷ = lnRMSSD[t]`` (no parameters).
* **ar1** — ``ŷ = c + φ·lnRMSSD[t]`` (OLS on the training fold).
* **candidate** — AR(1) plus today's per-sport TRIMP loads (OLS on the training fold).

Skill is out-of-sample RMSE on held-out folds. The pre-registered decision (OSF §6) is read
off **Combinatorial Purged CV**: the per-path *skill improvement* ``RMSE_ar1 − RMSE_candidate``
must have its **5th percentile > 0** for H-P1 to "beat baseline"; otherwise it is reported as
null. All splits use :mod:`garmin_nof1.eval.cv` (purged, embargoed) so no fold trains on data
adjacent to its test block. The most-recent 20% is reserved via :func:`holdout_split` and
must be evaluated at most once, after the model is locked (discipline enforced by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from garmin_nof1.eval.cv import combinatorial_purged_splits, effective_sample_size
from garmin_nof1.models._common import modeled_sports

_MODELS = ("random_walk", "ar1", "candidate")


def holdout_split(df: pd.DataFrame, *, frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the daily panel into a development set and the temporally-final holdout.

    The holdout is the last ``frac`` of rows (most recent days); development is everything
    before. The two are disjoint and ordered (every dev day precedes every holdout day), so
    a model developed on ``dev`` can be evaluated once on ``hold`` without lookahead.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    n_hold = int(round(len(df) * frac))
    if n_hold < 1 or n_hold >= len(df):
        raise ValueError("holdout fraction leaves an empty dev or holdout set")
    return df.iloc[: len(df) - n_hold].copy(), df.iloc[len(df) - n_hold :].copy()


def build_supervised(df: pd.DataFrame, *, outcome: str = "ln_rmssd") -> pd.DataFrame:
    """Build the one-step-ahead supervised table from a contiguous daily panel.

    Row *t* carries features known at the end of day *t* — the persistence term ``y_t`` and
    per-sport loads ``load_<sport>`` (today's TRIMP, in units of 100) — and the target
    ``y_next`` = the outcome on day *t+1*. Rows whose night or whose next night is missing
    (non-wear) are dropped, so every kept row is a valid observed consecutive-day pair.
    """
    work = df.reset_index(drop=True)
    y = work[outcome].astype(float)
    trimp = work["trimp"].astype(float).to_numpy()
    sport = np.asarray(work["sport"].astype(object))
    cols = {"y_next": y.shift(-1).to_numpy(), "y_t": y.to_numpy()}
    for s in modeled_sports(sport):
        cols[f"load_{s}"] = np.where(sport == s, trimp / 100.0, 0.0)
    sup = pd.DataFrame(cols)
    return sup.dropna(subset=["y_next", "y_t"]).reset_index(drop=True)


def _ols_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Least-squares fit on train, predict test (design matrices include the intercept)."""
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return x_test @ beta


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _fold_rmse(sup: pd.DataFrame, load_cols: list[str], tr: np.ndarray, te: np.ndarray) -> dict:
    """RMSE of each model on one (train, test) split of the supervised table."""
    y_tr, y_te = sup["y_next"].to_numpy()[tr], sup["y_next"].to_numpy()[te]
    yt_tr, yt_te = sup["y_t"].to_numpy()[tr], sup["y_t"].to_numpy()[te]
    ar_tr = np.column_stack([np.ones(len(tr)), yt_tr])
    ar_te = np.column_stack([np.ones(len(te)), yt_te])
    load_tr = sup[load_cols].to_numpy()[tr]
    load_te = sup[load_cols].to_numpy()[te]
    cand_tr = np.column_stack([ar_tr, load_tr])
    cand_te = np.column_stack([ar_te, load_te])
    return {
        "random_walk": _rmse(yt_te, y_te),
        "ar1": _rmse(_ols_predict(ar_tr, y_tr, ar_te), y_te),
        "candidate": _rmse(_ols_predict(cand_tr, y_tr, cand_te), y_te),
    }


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of the leakage-safe H-P1 evaluation.

    Attributes
    ----------
    rmse : dict[str, float]
        Mean out-of-sample RMSE per model across CPCV splits (``random_walk``/``ar1``/
        ``candidate``).
    skill_improvement : float
        Mean of ``RMSE_ar1 − RMSE_candidate`` across splits (positive = candidate better).
    skill_improvement_p05 : float
        5th percentile of that per-split improvement (the pre-registered decision statistic).
    beats_baseline : bool
        Pre-registered H-P1 verdict: ``skill_improvement_p05 > 0`` (OSF §6); else null.
    n_splits : int
        Number of CPCV train/test combinations scored (= the size of the skill-improvement
        distribution, ``C(n_groups, n_test_groups)``).
    ess : float
        Effective sample size of the target series (independent-information ceiling on power).
    n_obs : int
        Supervised consecutive-day pairs available.
    """

    rmse: dict[str, float]
    skill_improvement: float
    skill_improvement_p05: float
    beats_baseline: bool
    n_splits: int
    ess: float
    n_obs: int


def holdout_skill(
    dev_df: pd.DataFrame, hold_df: pd.DataFrame, *, outcome: str = "ln_rmssd"
) -> dict:
    """Single confirmatory evaluation: fit on the development panel, predict the holdout.

    Models are fit on **all** of ``dev_df``'s supervised pairs and scored once on
    ``hold_df``'s pairs. The straddling pair (last dev day → first holdout day) is never
    formed because each panel's supervised table is built independently, so there is no
    lookahead across the boundary. The candidate's load columns are taken from the
    development fit; any sport absent in the holdout is filled with zero load.

    Returns ``{"rmse": {model: rmse}, "skill_improvement": rmse_ar1 - rmse_candidate,
    "n_holdout": int}``.
    """
    sup_dev = build_supervised(dev_df, outcome=outcome)
    sup_hold = build_supervised(hold_df, outcome=outcome)
    load_cols = [c for c in sup_dev.columns if c.startswith("load_")]
    # align holdout to the development load columns (missing sport -> zero load)
    for c in load_cols:
        if c not in sup_hold.columns:
            sup_hold[c] = 0.0
    tr = np.arange(len(sup_dev))
    # _fold_rmse expects one table; concatenate dev+hold and index into it
    both = pd.concat([sup_dev, sup_hold[sup_dev.columns]], ignore_index=True)
    te = np.arange(len(sup_dev), len(both))
    fold = _fold_rmse(both, load_cols, tr, te)
    return {
        "rmse": fold,
        "skill_improvement": fold["ar1"] - fold["candidate"],
        "n_holdout": len(sup_hold),
    }


def evaluate_prediction(
    df: pd.DataFrame,
    *,
    outcome: str = "ln_rmssd",
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 2,
) -> PredictionResult:
    """Evaluate H-P1 with Combinatorial Purged CV on the supervised consecutive-day pairs.

    Fits the three models on each CPCV training set and scores RMSE on the held-out groups,
    then summarizes the candidate-vs-AR(1) skill-improvement distribution and the
    pre-registered ``5th-percentile > 0`` decision. Pass only the **development** panel here;
    keep the holdout (see :func:`holdout_split`) for a single final evaluation.
    """
    sup = build_supervised(df, outcome=outcome)
    load_cols = [c for c in sup.columns if c.startswith("load_")]
    if len(sup) < n_groups * 3:
        raise ValueError("too few supervised pairs for the requested CPCV grouping")

    splits = combinatorial_purged_splits(
        len(sup), n_groups=n_groups, n_test_groups=n_test_groups, embargo=embargo
    )
    per_model: dict[str, list[float]] = {m: [] for m in _MODELS}
    improvements: list[float] = []
    for tr, te in splits:
        fold = _fold_rmse(sup, load_cols, tr, te)
        for m in _MODELS:
            per_model[m].append(fold[m])
        improvements.append(fold["ar1"] - fold["candidate"])

    imp = np.asarray(improvements)
    p05 = float(np.quantile(imp, 0.05))
    return PredictionResult(
        rmse={m: float(np.mean(per_model[m])) for m in _MODELS},
        skill_improvement=float(np.mean(imp)),
        skill_improvement_p05=p05,
        beats_baseline=bool(p05 > 0.0),
        n_splits=len(splits),
        ess=effective_sample_size(df[outcome].to_numpy()),
        n_obs=len(sup),
    )
