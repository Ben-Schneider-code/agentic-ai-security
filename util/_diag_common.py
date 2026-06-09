"""Shared boilerplate for the self-play training/eval diagnostic scripts.

These three scripts share the same per-results-dir setup:
  - train_outcome_diagnostics.py
  - train_optimization_diagnostics.py
  - pvr_config_compare.py

This module fills the gaps not covered by ``plotting/_data.py`` / ``util/metrics.py``:
fail-fast config loading, exporting ``HONEYPOT_TYPE`` before any cross-eval load, and
the iter -> run_dir discovery dance. It does NOT reimplement any metric or loader that
already lives in those modules.

Import requirement: the importing script must put the repo root on ``sys.path`` BEFORE
``from util._diag_common import ...`` (this module imports ``plotting._data`` at load).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

from plotting._data import discover_iterations, find_run_dir

# Honeypot universe sizes per arm (from MARFT redteam_sql_env get_total_honeypots):
#   col    = 34 forbidden columns
#   row    = 30 enumerable owner ids
#   rowcol = 64 (34 + 30)
# Used only as a fallback when cross_eval episode_stats["honeypot_universe"] is absent.
HONEYPOT_UNIVERSE: dict[str, int] = {"col": 34, "row": 30, "rowcol": 64}


def note(msg: str) -> None:
    """Emit a diagnostic note to stderr (keeps stdout tables clean)."""
    print(f"[diag] {msg}", file=sys.stderr)


def load_run_summary(results_dir: str | Path) -> dict:
    """Load ``<results_dir>/summary.json``; fail fast with no silent defaults.

    Exits nonzero if summary.json is missing or lacks ``honeypot_type`` — the
    project convention is to crash early on missing config rather than guess.
    """
    path = Path(results_dir) / "summary.json"
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} not found — cannot determine run config.")
    with open(path) as fh:
        summary = json.load(fh)
    if "honeypot_type" not in summary:
        raise SystemExit(
            f"FATAL: {path} has no 'honeypot_type' — refusing to assume an arm."
        )
    return summary


def iter_runs(results_dir: str | Path) -> Iterator[tuple[int, str, Path]]:
    """Yield ``(iter_n, side, run_dir)`` for each trained team in a results dir.

    Wraps ``discover_iterations`` + ``find_run_dir``. Team dirs missing their
    ``.success`` marker come back as None from discover_iterations and are skipped;
    a team dir present but lacking ``training_state.json`` (find_run_dir -> None) is
    skipped with a note. ``side`` is "red" or "blue".
    """
    for it in discover_iterations(str(results_dir)):
        n = it["iter"]
        for side, key in (("red", "red_dir"), ("blue", "blue_dir")):
            team_dir = it[key]
            if team_dir is None:
                continue
            run_dir = find_run_dir(team_dir)
            if run_dir is None:
                note(f"iter_{n}/{side}team: no training_state.json — skipping side")
                continue
            yield n, side, run_dir
