#!/usr/bin/env python
"""H-P1 prediction report: does cross-sport load beat an AR(1)/random-walk baseline?

Runs the demoted prediction layer (OSF §5/§6) with strict holdout discipline:

  1. split the daily panel into development (first 80%) and a never-touched holdout
     (most-recent 20%);
  2. size the embargo from the AR(1)-residual autocorrelation on the development set;
  3. report the pre-registered decision from Combinatorial Purged CV on development
     (candidate beats baseline iff the 5th percentile of skill improvement > 0); then
  4. evaluate the locked model **once** on the holdout.

The pre-registered prior is that the increment is null. Output is privacy-safe
(RMSE / skill numbers only). Usage:
    python scripts/prediction_report.py --hr-rest <resting HR> --hr-max <max HR>
"""

from __future__ import annotations

import argparse

from garmin_nof1.models.prediction import (
    embargo_from_residual_acf,
    evaluate_prediction,
    holdout_skill,
    holdout_split,
)
from garmin_nof1.pipeline.build_panel import build_daily_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="H-P1 prediction skill report (holdout-safe).")
    parser.add_argument("--raw-dir", nargs="+", default=["data/raw", "data/raw_cn"])
    parser.add_argument("--hr-rest", type=float, required=True, help="resting HR for TRIMP")
    parser.add_argument("--hr-max", type=float, required=True, help="maximum HR for TRIMP")
    args = parser.parse_args()

    panel = build_daily_panel(args.raw_dir, hr_rest=args.hr_rest, hr_max=args.hr_max)
    dev, hold = holdout_split(panel, frac=0.2)
    embargo = embargo_from_residual_acf(dev)

    print("=" * 70)
    print("H-P1 prediction report (RMSE / skill only; no physiological values)")
    print("=" * 70)
    print(f"\nPanel {len(panel)} days -> dev {len(dev)} + holdout {len(hold)} (most recent 20%)")
    print(f"Embargo (from AR(1) residual ACF): {embargo} days")

    res = evaluate_prediction(dev, n_groups=6, n_test_groups=2, embargo=embargo)
    print("\n[dev · Combinatorial Purged CV] one-step next-day lnRMSSD RMSE:")
    for m in ("random_walk", "ar1", "candidate"):
        print(f"  {m:12}: {res.rmse[m]:.4f}")
    print(f"  skill improvement, candidate vs ar1: mean {res.skill_improvement:+.4f}, "
          f"5th pct {res.skill_improvement_p05:+.4f}")
    print(f"  CPCV paths {res.n_splits}   effective sample size ESS≈{res.ess:.0f}/{len(dev)}")
    verdict = "beats baseline (P05 > 0)" if res.beats_baseline else "does not beat baseline (null)"
    print(f"  >>> pre-registered H-P1 decision: {verdict}")

    print("\n[holdout, evaluated once] fit on dev, predict the most recent 20%:")
    out = holdout_skill(dev, hold)
    for m in ("random_walk", "ar1", "candidate"):
        print(f"  {m:12}: {out['rmse'][m]:.4f}")
    print(
        f"  skill improvement, candidate vs ar1: {out['skill_improvement']:+.4f}"
        f"  (n={out['n_holdout']})"
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
