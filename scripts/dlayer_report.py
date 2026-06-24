#!/usr/bin/env python
"""D-layer measurement report: reconstruct HRV from chest-strap RR and audit it.

Runs on the archived activity FITs that carry beat-to-beat RR (chest strap). For each such
session it:
  1. reconstructs mean HR / RMSSD / SDNN from the raw RR with the D-layer parser, and
  2. runs an RR-parsing **self-consistency check**: our RR-derived mean HR (= beats / duration)
     vs Garmin's firmware ``avg_heart_rate``. NOTE this is *not* independent device agreement —
     both come from the **same single chest-strap beat stream**, so ``60000/mean(rr)`` and the
     firmware average are near-arithmetic identities; agreement to ~1 bpm only confirms the RR
     parse/concatenation reproduces the firmware's beats/duration (it catches gross parser bugs,
     it does not validate a sensor), and
  3. audits RR data quality (artifact rate, the RMSSD shift artifact-correction causes,
     and beat coverage of the session), split by sport — the non-circular part of the report.

Honest scope (see preregistration §13/A5): when a chest strap is paired the watch logs the
strap as the *sole* HR source, so these files do **not** contain an independent wrist-optical
series — a simultaneous wrist-vs-chest device comparison is not supported and is not claimed
here.

Output is privacy-safe (aggregate statistics only; no raw HR/RR values). Usage:
    python scripts/dlayer_report.py
    python scripts/dlayer_report.py --raw-dir data/raw data/raw_cn
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from garmin_nof1.eval.agreement import agreement
from garmin_nof1.pipeline.parse_rr import reconstruct_rr_ms, rr_quality


def _iter_rr_sessions(raw_dirs: list[Path]):
    """Yield (path, sport, garmin_avg_hr, rr_ms, session_seconds) for FITs that carry RR."""
    from fitparse import FitFile

    for raw_dir in raw_dirs:
        for path in sorted((raw_dir / "activities").glob("*.fit")):
            ff = FitFile(str(path))
            rr = reconstruct_rr_ms([m.get_value("time") for m in ff.get_messages("hrv")])
            if rr.size < 2:
                continue
            sess = next(iter(ff.get_messages("session")), None)
            sport = sess.get_value("sport") if sess else None
            avg_hr = sess.get_value("avg_heart_rate") if sess else None
            secs = sess.get_value("total_timer_time") if sess else None
            yield path, sport, avg_hr, rr, secs


def main() -> None:
    parser = argparse.ArgumentParser(description="D-layer HRV reconstruction + quality report.")
    parser.add_argument("--raw-dir", nargs="+", default=["data/raw", "data/raw_cn"])
    parser.add_argument("--threshold", type=float, default=0.2, help="artifact filter threshold")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    raw_dirs = [Path(d) for d in args.raw_dir if (Path(d) / "activities").is_dir()]

    our_hr, garmin_hr, by_sport = [], [], {}
    rows = []
    for _path, sport, avg_hr, rr, secs in _iter_rr_sessions(raw_dirs):
        q = rr_quality(rr, session_seconds=secs, threshold=args.threshold)
        sport = sport or "?"
        rows.append((sport, q))
        by_sport.setdefault(sport, 0)
        by_sport[sport] += 1
        if avg_hr:  # Garmin reference present -> include in the HR agreement
            our_hr.append(60_000.0 / np.mean(rr))
            garmin_hr.append(float(avg_hr))

    print("=" * 70)
    print("D 层测量报告(仅聚合统计,不含任何具体心率/RR 数值)")
    print("=" * 70)
    if not rows:
        print("没有发现带 RR 的活动 FIT。先用 `pull_garmin.py --fits` 下载胸带那段的活动。")
        return

    print(f"\n带逐拍 RR 的 session:{len(rows)} 个  (按运动:{by_sport})")

    print("\n[1] RR 解析自洽性检查(非独立比对):我们的 60000/mean(RR) vs Garmin 固件 avg HR")
    print("    注意:两者出自同一条胸带拍流,近似恒等式;吻合只证明 RR 解析正确,非传感器验证。")
    if len(our_hr) >= 2:
        a = agreement(np.array(garmin_hr), np.array(our_hr))  # reference=Garmin, test=ours
        print(f"  n={a.n}  bias(ours-Garmin)={a.bias:+.2f} bpm  95% LoA=({a.loa_lower:+.2f},"
              f" {a.loa_upper:+.2f})")
        print(f"  MAPE={a.mape:.2f}%   ICC(2,1)={a.icc:.4f}   CCC={a.ccc:.4f}")
        print("  (≈恒等,只表示 RR 解析/拼接复现了固件的 beats/duration —— 能抓住粗解析 bug)")
    else:
        print("  (可用于比对的 session 不足 2 个)")

    print("\n[2] RR 数据质量审计(按运动)")
    print(f"  {'sport':10}{'n':>4}{'artifact%':>11}{'RMSSD_raw':>11}{'RMSSD_corr':>12}"
          f"{'coverage':>10}")
    for sport in sorted(by_sport):
        qs = [q for s, q in rows if s == sport]
        art = np.mean([q.artifact_rate for q in qs]) * 100
        rraw = np.mean([q.rmssd_raw for q in qs])
        rcorr = np.mean([q.rmssd_corrected for q in qs])
        cov = np.nanmean([q.coverage for q in qs])
        print(f"  {sport:10}{len(qs):>4}{art:>10.2f}%{rraw:>11.1f}{rcorr:>12.1f}{cov:>9.2f}")

    allq = [q for _, q in rows]
    dr = np.mean([q.rmssd_raw - q.rmssd_corrected for q in allq])
    print(f"\n  artifact 校正对 RMSSD 的平均影响:{dr:+.2f} ms"
          f"(raw 比 corrected 高这么多,说明伪影会虚增 RMSSD)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
