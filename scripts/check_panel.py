#!/usr/bin/env python
"""Verify the assembled panel from your archived Garmin data (run after pull_garmin.py).

It builds the daily panel from ``data/raw`` and prints two privacy-safe summaries:
  1. the distinct activity ``typeKey``s found (+ how each maps under the current rules), so
     the sport mapping can be confirmed/extended, and
  2. how many real (non-missing) values each panel column received.

Neither summary contains physiological values. ``--show-panel`` prints the full panel
locally, which does include them.

Usage:
    python scripts/check_panel.py
    python scripts/check_panel.py --hr-rest 55 --hr-max 185 --show-panel
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
        help="one or more archive dirs (default: data/raw and data/raw_cn)",
    )
    parser.add_argument("--hr-rest", type=float, default=50.0, help="resting HR for TRIMP")
    parser.add_argument("--hr-max", type=float, default=190.0, help="maximum HR for TRIMP")
    parser.add_argument(
        "--show-panel", action="store_true", help="print the full panel (personal values)"
    )
    args = parser.parse_args()

    raw_dirs = [d for d in args.raw_dir if os.path.isdir(d)]

    print("=" * 70)
    print("Panel check (activity type names and counts only; no physiological values)")
    print("=" * 70)
    print(f"\nArchive dirs: {raw_dirs or '(none found)'}")

    print("\n[1] Activity typeKeys (count) -> mapped sport:")
    keys = distinct_sport_keys(raw_dirs)
    if not keys:
        print("  (no activity files found)")
    for key, n in sorted(keys.items(), key=lambda kv: -kv[1]):
        print(f"  {key!r:28}  x{n:<4} -> {map_sport(key)}")

    panel = build_daily_panel(raw_dirs, hr_rest=args.hr_rest, hr_max=args.hr_max)
    total = len(panel)
    print(f"\n[2] Daily panel built: {total} days. Non-missing values per column:")
    nonrest = int((panel["sport"].astype(str) != "rest").sum())
    print(f"  sport       : {nonrest}/{total} training (non-rest) days")
    for col in ["trimp", "sleep_hours", "rhr", "ln_rmssd"]:
        print(f"  {col:12}: {int(panel[col].notna().sum())}/{total} non-missing")
    print("\n" + "=" * 70)
    print("(sleep_hours / rhr / ln_rmssd near the day count means the schema adapters match)")

    if args.show_panel:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print("\n--- Full panel (personal values; keep local) ---")
        print(panel.to_string(index=False))


if __name__ == "__main__":
    main()
