#!/usr/bin/env python
"""Runner: log into Garmin Connect and archive raw data locally.

Pulls your own Garmin data. It:
  1. logs you in (handles two-factor codes; saves the session so re-runs need no password),
  2. pulls daily summaries (HRV / sleep / resting-HR) + the activity list for a date range,
  3. archives every raw response under ``data/raw/`` (gitignored — your data never leaves
     your machine), and
  4. prints a privacy-safe *structure peek* of each response type (key names + value types
     only, no actual numbers) for checking the schema adapters.

Usage
-----
    python scripts/pull_garmin.py                         # test pull: the last 7 days
    python scripts/pull_garmin.py --start 2024-01-01 --end 2024-12-31   # a date range
    python scripts/pull_garmin.py --cn --start 2024-01-01 --end 2024-12-31  # China account

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
    print(f"== Garmin login ({region}) ==")
    email = input("Garmin email / phone: ").strip()
    password = getpass.getpass("Garmin password (hidden): ")
    print("If two-factor auth is on, you will be asked for the 6-digit code next.")

    def prompt_mfa():
        return input("Two-factor code (Enter to skip): ").strip()

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
        print("✓ Resumed saved China session")
    else:
        email, password, prompt_mfa = _prompt_credentials("China, via garth")
        garth.configure(domain="garmin.cn")
        garth.login(email, password, prompt_mfa=prompt_mfa)
        try:
            garth.save(garth_home)
            print("✓ Logged in (China); session saved")
        except Exception:
            print("✓ Logged in (China)")
        prof = garth.connectapi("/userprofile-service/socialProfile")

    display_name = (prof or {}).get("displayName")
    if not display_name:
        raise SystemExit("No displayName for the China account; login may have failed.")
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
            print("✓ Resumed saved global session")
            return api
        except Exception:
            print("(Saved global session expired; logging in again)")

    email, password, prompt_mfa = _prompt_credentials("global")
    api = Garmin(email, password, is_cn=False, prompt_mfa=prompt_mfa)
    api.login()
    try:
        os.makedirs(tokenstore, exist_ok=True)
        api.client.dump(tokenstore)
        print("✓ Logged in (global); session saved")
    except Exception:
        print("✓ Logged in (global)")
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
    """Print the structure of one archived file per type, for checking ``garmin_schema``."""
    print("\n" + "=" * 70)
    print("Structure peek (key names + value types only; no values)")
    print("=" * 70)
    for prefix in ["hrv", "sleep", "rhr", "activities"]:
        files = sorted(glob.glob(os.path.join(str(raw_dir), f"{prefix}-*.json")))
        if not files:
            print(f"\n[{prefix}] (no files)")
            continue
        data = json.loads(open(files[0]).read())
        print(f"\n[{prefix}] from {os.path.basename(files[0])}:")
        print(_peek(data))
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pull and archive your Garmin data locally.")
    today = datetime.date.today()
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD (default: 7 days ago)")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    parser.add_argument("--cn", action="store_true", help="pull from Garmin China")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="archive dir (default: data/raw, or data/raw_cn with --cn)",
    )
    parser.add_argument(
        "--fits",
        action="store_true",
        help="also download original activity FITs (beat-to-beat RR, for the D-layer; slow)",
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

    print(f"\nPulling daily summaries + activities for {start} → {end} ...")
    print("(If interrupted or rate-limited, re-run the same command; fetched days are skipped.)")
    counts = client.ingest_range(start, end)
    print("Done. Newly fetched:", counts)
    print("Raw files:", config.raw_dir.resolve())

    if args.fits:
        from garmin_nof1.pipeline.ingest_garmin import activity_ids_in_range

        acts = []
        for path in sorted(glob.glob(os.path.join(raw_dir, "activities-*.json"))):
            acts.extend(json.loads(open(path).read()))
        ids = activity_ids_in_range(acts, start, end)
        print(f"\nDownloading original FITs for {len(ids)} activities (slow; resumable) ...")
        existing = {p.stem for p in (config.raw_dir / "activities").glob("*.fit")}
        todo = [i for i in ids if str(i) not in existing]
        paths = client.download_activity_fits(todo) if todo else []
        print(f"Downloaded {len(paths)} new .fit ({len(existing)} already present)")

    show_samples(config.raw_dir)


if __name__ == "__main__":
    main()
