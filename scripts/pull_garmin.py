#!/usr/bin/env python
"""Beginner-friendly runner: log into Garmin Connect and archive raw data locally.

This is the ONE script you run to pull your own data. It:
  1. logs you in (handles two-factor codes; saves the session so re-runs need no password),
  2. pulls daily summaries (HRV / sleep / resting-HR) + the activity list for a date range,
  3. archives every raw response under ``data/raw/`` (gitignored — your data never leaves
     your machine), and
  4. prints a privacy-safe *structure peek* of each response type (key names + value types
     only, no actual numbers) so the assumed Garmin schema can be reconciled.

Usage
-----
    python scripts/pull_garmin.py                         # test pull: the last 7 days
    python scripts/pull_garmin.py --start 2023-01-01 --end 2024-12-31   # the full range

Your password is typed interactively and never written to disk. The login session is
cached under ``~/.garminconnect`` (also private).
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import glob
import json
import os
from pathlib import Path

from garmin_nof1.pipeline.ingest_garmin import GarminClient, GarminConfig

# Cached login sessions (private). China-region and global accounts get separate stores so
# their sessions never clobber each other.
TOKENSTORE = os.path.expanduser("~/.garminconnect")
TOKENSTORE_CN = os.path.expanduser("~/.garminconnect-cn")


def get_api(is_cn: bool, tokenstore: str):
    """Return an authenticated garminconnect client, reusing a saved session if possible.

    ``is_cn=True`` routes to Garmin China (connect.garmin.cn) instead of the global servers.
    """
    from garminconnect import Garmin

    region = "中国区" if is_cn else "国际区"
    # Try to resume a previously-saved login (no password needed).
    if os.path.isdir(tokenstore):
        try:
            api = Garmin(is_cn=is_cn)
            api.login(tokenstore)
            print(f"✓ 复用已保存的{region}登录会话(无需输密码)")
            return api
        except Exception:
            print(f"（已保存的{region}登录失效,需要重新登录一次）")

    # Fresh login.
    print(f"== 登录 Garmin {region} ==")
    email = input("Garmin 邮箱: ").strip()
    password = getpass.getpass("Garmin 密码(输入时屏幕不显示,输完按回车): ")
    print("若账号开了二次验证,接下来会让你输手机/邮箱收到的 6 位码(没开就直接回车跳过)。")
    mfa_prompt = "二次验证码(没有就回车): "
    api = Garmin(email, password, is_cn=is_cn, prompt_mfa=lambda: input(mfa_prompt).strip())
    api.login()
    try:
        os.makedirs(tokenstore, exist_ok=True)
        api.garth.dump(tokenstore)
        print(f"✓ {region}登录成功,会话已保存,下次运行免密")
    except Exception:
        print(f"✓ {region}登录成功")
    return api


def _peek(obj, depth=0, maxdepth=3):
    """A privacy-safe structure summary: key names + value *types*, not the values.

    Dates (``YYYY-MM-DD`` strings) are shown verbatim because their format matters for
    reconciliation; all other values are reduced to their type name.
    """
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= maxdepth:
            return "{...}"
        lines = ["{"]
        for key, val in list(obj.items())[:40]:
            lines.append(f"{pad}  {key}: {_peek(val, depth + 1, maxdepth)}")
        lines.append(pad + "}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return "list[0]"
        return f"list[{len(obj)}] of " + _peek(obj[0], depth, maxdepth)
    if isinstance(obj, str):
        if len(obj) == 10 and obj[4] == "-" and obj[7] == "-":  # looks like a date
            return f'"{obj}" (date str)'
        return "str"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, (int, float)):
        return "number"
    if obj is None:
        return "null"
    return type(obj).__name__


def show_samples(raw_dir):
    """Print the structure of one archived file per type — copy this to Claude to reconcile."""
    print("\n" + "=" * 70)
    print("把下面这一整段(到结尾的 ==== 为止)发给 Claude,用来对账 garmin_schema 的假定键。")
    print("注意:这里只显示【键名 + 类型】,不含你的任何具体数值。")
    print("=" * 70)
    for prefix in ["hrv", "sleep", "rhr", "activities"]:
        files = sorted(glob.glob(os.path.join(str(raw_dir), f"{prefix}-*.json")))
        if not files:
            print(f"\n[{prefix}] —— 没有文件(这一类没拉到)")
            continue
        data = json.loads(open(files[0]).read())
        print(f"\n[{prefix}] 取自 {os.path.basename(files[0])}:")
        print(_peek(data))
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pull and archive your Garmin data locally.")
    today = datetime.date.today()
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD(默认:7 天前)")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD(默认:今天)")
    parser.add_argument("--cn", action="store_true", help="从 Garmin 中国(connect.garmin.cn)拉取")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="存档目录(默认:国际区 data/raw,中国区 data/raw_cn —— 两个账号自动分开存)",
    )
    args = parser.parse_args()

    end = args.end or today.isoformat()
    start = args.start or (today - datetime.timedelta(days=7)).isoformat()
    raw_dir = args.raw_dir or ("data/raw_cn" if args.cn else "data/raw")
    tokenstore = TOKENSTORE_CN if args.cn else TOKENSTORE

    api = get_api(args.cn, tokenstore)
    # creds unused here — the already-authenticated api is injected — so placeholders are fine.
    config = GarminConfig(email="-", password="-", raw_dir=Path(raw_dir))
    client = GarminClient(config, api=api)

    print(f"\n正在拉取 {start} → {end} 的日汇总 + 活动 ...(范围大时会比较久)")
    print("（若中途断了/被限流,直接重跑本命令即可续传 —— 已拉过的天会自动跳过)")
    counts = client.ingest_range(start, end)
    print("完成。本次新拉取条数:", counts)
    print("原始文件在:", config.raw_dir.resolve())

    show_samples(config.raw_dir)


if __name__ == "__main__":
    main()
