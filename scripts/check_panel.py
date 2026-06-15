#!/usr/bin/env python
"""Verify the assembled panel from your archived Garmin data (run after pull_garmin.py).

It builds the daily panel from ``data/raw`` and prints two privacy-safe summaries you can
paste back for the final schema reconciliation:
  1. the distinct activity ``typeKey``s found (+ how each maps under the current rules), so
     the sport mapping can be confirmed/extended, and
  2. how many real (non-missing) values each panel column received.

Neither summary contains your actual numbers. Add ``--show-panel`` to print the full panel
locally (that DOES include your values — for your eyes, don't paste it).

Usage:
    python scripts/check_panel.py
    python scripts/check_panel.py --hr-rest 48 --hr-max 190 --show-panel
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from garmin_nof1.pipeline.build_panel import build_daily_panel, map_sport


def distinct_sport_keys(raw_dirs: list[str]) -> dict[str, int]:
    """Count each ``activityType.typeKey`` across the archived activities files in all dirs."""
    counts: dict[str, int] = {}
    for raw_dir in raw_dirs:
        for path in sorted(glob.glob(os.path.join(raw_dir, "activities-*.json"))):
            for act in json.loads(open(path).read()):
                key = (act.get("activityType") or {}).get("typeKey")
                if key:
                    counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Build + sanity-check the daily panel.")
    parser.add_argument(
        "--raw-dir",
        nargs="+",
        default=["data/raw", "data/raw_cn"],
        help="一个或多个存档目录(默认两个账号都读:data/raw 与 data/raw_cn)",
    )
    parser.add_argument("--hr-rest", type=float, default=50.0, help="占位用于 TRIMP,先随便填")
    parser.add_argument("--hr-max", type=float, default=190.0, help="占位用于 TRIMP,先随便填")
    parser.add_argument("--show-panel", action="store_true", help="本地查看完整面板(含你的数值)")
    args = parser.parse_args()

    raw_dirs = [d for d in args.raw_dir if os.path.isdir(d)]

    print("=" * 70)
    print("把下面这一整段(到结尾的 ==== 为止)发回给 Claude 做最后核对。")
    print("只含运动类型名 + 计数,不含你的任何具体数值。")
    print("=" * 70)
    print(f"\n读取目录: {raw_dirs or '(没有任何存档目录存在)'}")

    print("\n[1] 活动里出现的 sport typeKey(次数)-> 当前映射:")
    keys = distinct_sport_keys(raw_dirs)
    if not keys:
        print("  (没找到活动文件)")
    for key, n in sorted(keys.items(), key=lambda kv: -kv[1]):
        print(f"  {key!r:28}  x{n:<4} -> {map_sport(key)}")

    panel = build_daily_panel(raw_dirs, hr_rest=args.hr_rest, hr_max=args.hr_max)
    total = len(panel)
    print(f"\n[2] 日面板组装成功:{total} 天。每列拿到多少真实(非缺失)值:")
    nonrest = int((panel["sport"].astype(str) != "rest").sum())
    print(f"  sport       : {nonrest}/{total} 天为非 rest(有训练)")
    for col in ["trimp", "sleep_hours", "rhr", "ln_rmssd"]:
        print(f"  {col:12}: {int(panel[col].notna().sum())}/{total} 非缺失")
    print("\n" + "=" * 70)
    print("(理想情况:sleep_hours / rhr / ln_rmssd 基本都接近总天数 = 各适配器都对上了)")

    if args.show_panel:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print("\n--- 完整面板(本地查看,含你的数值,勿外发)---")
        print(panel.to_string(index=False))


if __name__ == "__main__":
    main()
