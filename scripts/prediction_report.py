#!/usr/bin/env python
"""H-P1 prediction report: does cross-sport load beat an AR(1)/random-walk baseline?

Runs the demoted prediction layer (OSF §5/§6) with strict holdout discipline:

  1. split the daily panel into development (first 80%) and a never-touched holdout
     (most-recent 20%);
  2. size the embargo from the AR(1)-residual autocorrelation on the development set;
  3. report the pre-registered decision from Combinatorial Purged CV on development
     (candidate beats baseline iff the 5th percentile of skill improvement > 0); then
  4. evaluate the locked model **once** on the holdout.

The pre-registered prior is that the increment is null; a null result is the expected,
honestly-reported outcome. Output is privacy-safe (RMSE / skill numbers only). Usage:
    python scripts/prediction_report.py
    python scripts/prediction_report.py --hr-rest 49 --hr-max 211
"""

from __future__ import annotations

import argparse

import numpy as np

from garmin_nof1.models._common import deviation
from garmin_nof1.models.prediction import evaluate_prediction, holdout_skill, holdout_split
from garmin_nof1.pipeline.build_panel import build_daily_panel


def _embargo_from_residual_acf(dev_df, *, detrend_window=28, max_embargo=14) -> int:
    """First lag where the AR(1)-residual autocorrelation falls inside the ~95% white-noise
    band (|rho| < 2/sqrt(n)); the serial-correlation cooldown to embargo. Floored at 1."""
    dev = deviation(dev_df, "ln_rmssd", None, detrend_window).to_numpy()
    dev = dev[np.isfinite(dev)]
    # AR(1) residual: regress dev[t] on dev[t-1]
    x, y = dev[:-1], dev[1:]
    phi = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
    resid = y - (y.mean() + phi * (x - x.mean()))
    n = resid.size
    band = 2.0 / np.sqrt(n)
    rc = resid - resid.mean()
    var = float(np.dot(rc, rc))
    for k in range(1, min(max_embargo, n - 1) + 1):
        ac = float(np.dot(rc[:-k], rc[k:]) / var)
        if abs(ac) < band:
            return max(1, k)
    return max_embargo


def main() -> None:
    parser = argparse.ArgumentParser(description="H-P1 prediction skill report (holdout-safe).")
    parser.add_argument("--raw-dir", nargs="+", default=["data/raw", "data/raw_cn"])
    parser.add_argument("--hr-rest", type=float, default=49.0)
    parser.add_argument("--hr-max", type=float, default=211.0)
    args = parser.parse_args()

    panel = build_daily_panel(args.raw_dir, hr_rest=args.hr_rest, hr_max=args.hr_max)
    dev, hold = holdout_split(panel, frac=0.2)
    embargo = _embargo_from_residual_acf(dev)

    print("=" * 70)
    print("H-P1 预测层报告(只输出 RMSE/技能数值,不含具体生理读数)")
    print("=" * 70)
    print(f"\n面板 {len(panel)} 天 -> 开发集 {len(dev)} 天 + holdout {len(hold)} 天(最近 20%)")
    print(f"embargo(由 AR(1) 残差 ACF 定):{embargo} 天")

    res = evaluate_prediction(dev, n_groups=6, n_test_groups=2, embargo=embargo)
    print("\n[开发集 · Combinatorial Purged CV] 1 步预测 next-day lnRMSSD 的 RMSE:")
    for m in ("random_walk", "ar1", "candidate"):
        print(f"  {m:12}: {res.rmse[m]:.4f}")
    print(f"  技能提升 candidate vs ar1(RMSE 降低):均值 {res.skill_improvement:+.4f}, "
          f"5百分位 {res.skill_improvement_p05:+.4f}")
    print(f"  CPCV 组合数 {res.n_splits}   有效样本量 ESS≈{res.ess:.0f}/{len(dev)}")
    verdict = "超过基线(P05>0)" if res.beats_baseline else "未超过基线 -> 报 null(预注册先验)"
    print(f"  >>> 预注册 H-P1 判定:{verdict}")

    print("\n[holdout · 仅评一次] 在最近 20% 未见数据上拟合-预测:")
    out = holdout_skill(dev, hold)
    for m in ("random_walk", "ar1", "candidate"):
        print(f"  {m:12}: {out['rmse'][m]:.4f}")
    print(f"  技能提升 candidate vs ar1:{out['skill_improvement']:+.4f}  (n={out['n_holdout']})")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
