#!/usr/bin/env python3
"""One-time migration of pre-fix ``reward_config.yaml`` files.

Older runs were written by ``train_sql.save_reward_config_to_yaml`` before the
writer was hardened, so their ``reward_config.yaml`` files contain
``!!python/tuple`` tags that ``yaml.safe_load`` rejects. This script walks
each given ``results-*`` directory, finds every ``iter_*/{redteam,blueteam}/
**/reward_config.yaml``, recursively converts tuples to lists, and rewrites
the file with ``yaml.safe_dump``. Idempotent: clean files are left alone.

Dry-run by default — pass ``--apply`` to write.

Usage:
    python util/migrate_reward_configs.py results-DIR [results-DIR ...] [--apply]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _to_yaml_safe(obj):
    """Recursively replace tuples with lists. Mirrors the writer-side helper."""
    if isinstance(obj, tuple):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    return obj


def _contains_tuple(obj) -> bool:
    if isinstance(obj, tuple):
        return True
    if isinstance(obj, list):
        return any(_contains_tuple(v) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_tuple(v) for v in obj.values())
    return False


def discover_configs(results_dir: Path) -> list[Path]:
    """Yield every reward_config.yaml under iter_*/redteam|blueteam/**."""
    paths: list[Path] = []
    for iter_dir in sorted(results_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        for role in ("redteam", "blueteam"):
            paths.extend(sorted((iter_dir / role).glob("**/reward_config.yaml")))
    return paths


def process_file(path: Path, *, apply: bool) -> str:
    """Return one of: ``clean``, ``migrated``, ``would migrate``, ``error: <msg>``."""
    try:
        with path.open() as f:
            cfg = yaml.unsafe_load(f)
    except Exception as e:
        return f"error: load failed ({e})"

    if cfg is None:
        return "error: empty yaml"

    if not _contains_tuple(cfg):
        # Verify it actually round-trips through safe_load — sanity check.
        try:
            with path.open() as f:
                yaml.safe_load(f)
        except Exception as e:
            return f"error: no tuples but safe_load failed ({e})"
        return "clean"

    safe_cfg = _to_yaml_safe(cfg)

    if not apply:
        return "would migrate"

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(safe_cfg, f, default_flow_style=False, sort_keys=False)

    # Verify round-trip before swapping.
    with tmp.open() as f:
        check = yaml.safe_load(f)
    if "total_honeypots" not in check:
        tmp.unlink(missing_ok=True)
        return "error: post-write check missing total_honeypots"
    tmp.replace(path)
    return "migrated"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "results_dirs",
        nargs="+",
        type=Path,
        metavar="DIR",
        help="One or more results-* directories to scan.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files. Without this flag the script is a dry run.",
    )
    args = ap.parse_args()

    n_clean = 0
    n_migrated = 0
    n_would = 0
    n_error = 0

    for rd in args.results_dirs:
        if not rd.is_dir():
            print(f"[{rd}] not a directory, skipping", file=sys.stderr)
            n_error += 1
            continue
        configs = discover_configs(rd)
        if not configs:
            print(f"[{rd}] no reward_config.yaml found")
            continue
        print(f"[{rd}] {len(configs)} config(s):")
        for p in configs:
            status = process_file(p, apply=args.apply)
            print(f"  {status:<16}  {p}")
            if status == "clean":
                n_clean += 1
            elif status == "migrated":
                n_migrated += 1
            elif status == "would migrate":
                n_would += 1
            else:
                n_error += 1

    print()
    print(
        f"Summary: clean={n_clean}  migrated={n_migrated}  "
        f"would_migrate={n_would}  errors={n_error}"
        + ("  (dry run — pass --apply to write)" if not args.apply else "")
    )
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
