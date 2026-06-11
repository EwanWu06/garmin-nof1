"""Why the CV scaffold matters — run on the synthetic 2-year panel.

It makes two of the project's honesty points concrete:
  1. ~730 nominal days carry only a few dozen *independent* blocks of information.
  2. Shuffled k-fold reports an optimistic error because it interpolates
     autocorrelated neighbours; honest embargoed walk-forward does not.

    python examples/demo_cv_leakage.py
"""

import math

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from garmin_nof1.data.synthetic import make_daily_panel
from garmin_nof1.eval import (
    PurgedWalkForwardSplit,
    combinatorial_purged_splits,
    effective_sample_size,
    n_backtest_paths,
)


def main() -> None:
    df = make_daily_panel(n_days=730, seed=0)
    obs = df[df["hrv_observed"]].reset_index(drop=True)
    y = obs["ln_rmssd"].to_numpy()
    day = obs.index.to_numpy().reshape(-1, 1).astype(float)  # time as the only feature

    nominal = len(y)
    ess_raw = effective_sample_size(y)
    # Honest reporting: detrend first. The slow seasonal/fitness baseline is itself
    # highly autocorrelated, so ESS of the *raw* series is dominated by trend, not by
    # the day-to-day recovery dynamics you actually model. Report ESS on the residual.
    deviation = obs["ln_rmssd"] - obs["ln_rmssd"].rolling(28, center=True, min_periods=7).mean()
    ess_dev = effective_sample_size(deviation.to_numpy())

    print("── Effective sample size of nightly ln rMSSD ──────────────────")
    print(f"  nights with HRV (nominal)  : {nominal}")
    print(f"  ESS, raw (trend-inflated)  : {ess_raw:5.0f}  ({ess_raw / nominal:.0%} of nominal)")
    print(f"  ESS, 28-day residual       : {ess_dev:5.0f}  ({ess_dev / nominal:.0%} of nominal)")
    print("  → detrend before reporting; power is set by independent blocks.\n")

    knn = KNeighborsRegressor(n_neighbors=1)

    def rmse(cv):
        s = cross_val_score(knn, day, y, cv=cv, scoring="neg_mean_squared_error")
        return math.sqrt(-s.mean())

    rmse_shuffled = rmse(KFold(n_splits=5, shuffle=True, random_state=0))
    rmse_embargoed = rmse(PurgedWalkForwardSplit(n_splits=5, embargo=14))

    print("── Leakage gap (1-NN-in-time on an autocorrelated target) ─────")
    print(f"  shuffled 5-fold RMSE      : {rmse_shuffled:.3f}   (leaks neighbours)")
    print(f"  embargoed walk-forward    : {rmse_embargoed:.3f}   (honest)")
    print(
        f"  optimism from leakage     : {rmse_embargoed / rmse_shuffled:.1f}x worse "
        "once leakage is removed.\n"
    )

    splits = combinatorial_purged_splits(nominal, n_groups=6, n_test_groups=2, embargo=14)
    print("── CPCV (secondary variance diagnostic) ──────────────────────")
    print(f"  splits (C(6,2))           : {len(splits)}")
    print(f"  backtest paths            : {n_backtest_paths(6, 2)}")
    print("  caveat: paths share data → reported spread understates uncertainty.")


if __name__ == "__main__":
    main()
