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
    python scripts/pull_garmin.py --cn --start 2024-01-01 --end 2025-12-15  # China account

Global accounts authenticate via garminconnect; China accounts (``--cn``) via garth, which
is the only one of the two that speaks connect.garmin.cn's OAuth correctly. Your password is
typed interactively and never written to disk. Login sessions are cached privately (global:
``~/.garminconnect``; China: ``~/.garth-cn``).
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

# Cached login sessions (private), separate per account so they never clobber each other.
# The global account uses garminconnect (DI tokens -> ~/.garminconnect); the China account
# uses garth (OAuth1/OAuth2 tokens -> ~/.garth-cn), because garminconnect 0.3.2 can't
# authenticate China data calls (see _get_api_cn / garth_cn).
TOKENSTORE = os.path.expanduser("~/.garminconnect")
TOKENSTORE_CN = os.path.expanduser("~/.garth-cn")


def _prompt_credentials(region: str):
    """Prompt for username/password (+ MFA callback) for ``region``. Password never stored."""
    print(f"== 登录 Garmin {region} ==")
    email = input("Garmin 邮箱/手机号: ").strip()
    password = getpass.getpass("Garmin 密码(输入时屏幕不显示,输完按回车): ")
    print("若账号开了二次验证,接下来会让你输手机/邮箱收到的 6 位码(没开就直接回车跳过)。")

    def prompt_mfa():
        return input("二次验证码(没有就回车): ").strip()

    return email, password, prompt_mfa


def _cn_connectapi(path, **kwargs):
    """``garth.connectapi`` wrapper: a 404 (no data for that day) returns ``None`` instead
    of raising, so a multi-year daily pull doesn't die on days that predate your data.
    Real auth failures (401/403) still propagate."""
    import garth
    from garth.exc import GarthHTTPError

    try:
        return garth.connectapi(path, **kwargs)
    except GarthHTTPError as exc:
        resp = getattr(getattr(exc, "error", None), "response", None)
        if getattr(resp, "status_code", None) == 404:
            return None
        raise


def _get_api_cn(garth_home: str):
    """Authenticate against Garmin China via garth and return a garminconnect-compatible
    adapter (:class:`~garmin_nof1.pipeline.garth_cn.GarthCnApi`).

    garminconnect 0.3.2 can't authenticate China data calls (its DI-token exchange is
    hardcoded to global ``.com`` hosts -> 403 from ``connectapi.garmin.cn``). garth speaks
    China's OAuth correctly, auto-refreshes tokens, and persists them — so a saved session
    resumes without a password.
    """
    import garth

    from garmin_nof1.pipeline.garth_cn import GarthCnApi

    prof = None
    if os.path.isdir(garth_home):  # try to resume a saved session and prove it still works
        try:
            garth.resume(garth_home)
            prof = garth.connectapi("/userprofile-service/socialProfile")
        except Exception:
            prof = None

    if prof:
        print("✓ 复用已保存的中国区登录会话(无需输密码)")
    else:
        email, password, prompt_mfa = _prompt_credentials("中国区(走 garth)")
        garth.configure(domain="garmin.cn")
        garth.login(email, password, prompt_mfa=prompt_mfa)
        try:
            garth.save(garth_home)
            print("✓ 中国区登录成功,会话已保存,下次运行免密")
        except Exception:
            print("✓ 中国区登录成功")
        prof = garth.connectapi("/userprofile-service/socialProfile")

    display_name = (prof or {}).get("displayName")
    if not display_name:
        raise SystemExit("拿不到中国区账号的 displayName(sleep/rhr 接口需要它),登录可能没成功。")
    return GarthCnApi(
        connectapi=_cn_connectapi, download=garth.download, display_name=display_name
    )


def _get_api_intl(tokenstore: str):
    """Authenticate against global Garmin (connect.garmin.com) via garminconnect, resuming a
    saved DI-token session when present."""
    from garminconnect import Garmin

    token_file = os.path.join(tokenstore, "garmin_tokens.json")
    if os.path.isfile(token_file):
        try:
            api = Garmin(is_cn=False)
            api.login(tokenstore)
            print("✓ 复用已保存的国际区登录会话(无需输密码)")
            return api
        except Exception:
            print("（已保存的国际区登录失效,需要重新登录一次）")

    email, password, prompt_mfa = _prompt_credentials("国际区")
    api = Garmin(email, password, is_cn=False, prompt_mfa=prompt_mfa)
    api.login()
    try:
        os.makedirs(tokenstore, exist_ok=True)
        api.client.dump(tokenstore)
        print("✓ 国际区登录成功,会话已保存,下次运行免密")
    except Exception:
        print("✓ 国际区登录成功")
    return api


def get_api(is_cn: bool, tokenstore: str):
    """Return an authenticated client exposing garminconnect's method surface.

    China and global Garmin use different auth backends (garth vs garminconnect — see the
    two helpers), but both return an object ``GarminClient`` can drive identically.
    """
    return _get_api_cn(tokenstore) if is_cn else _get_api_intl(tokenstore)


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
