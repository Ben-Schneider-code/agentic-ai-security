"""
export_results_json.py — token-efficient JSON export of every enabled paper figure's
numbers, computed by the SAME primitives the plots use.

Drives off plotting/figure_registry.REGISTRY: each figure family's
`compute_<key>(results, **kwargs) -> dict` is the one source of truth shared with
plot_paper_figures.run_all(). The exporter never re-derives a metric — it calls the
registry compute and serializes the result into a small, self-describing JSON tree
(plus a SUMMARY.md) under <results-dir>/export/.

Output tree (per results dir):
    export/
      index.json     manifest: schema_version, run_meta, per-figure status, files
      config.json    summary.json passthrough + honeypot_universe
      headline.json  glossary + the paper "money numbers"
      cross_eval.json  N×N matrices + per-cell decompositions (minified)
      training.json    per-iteration training-time signals + cost + lora
      derived.json     secondary analyses
      SUMMARY.md     prose headline + glossary table (read this first)

Fail-fast (no silent fallback):
  * summary.json missing / no honeypot_type  -> SystemExit (load_run_summary).
  * export dir not writable (NFS root-squash) -> SystemExit (probe).

HONEYPOT_TYPE is exported from summary.json before any loader triggers the lazy
MARFT import, so coverage/yield/work_factor populate without a GPU. Figures that
need an optional dep (sklearn) or a missing source degrade to a recorded skip;
one failing figure never aborts the export.

CLI:
    python util/export_results_json.py results-<ID>[:Label] [more dirs ...]
    python util/export_results_json.py --results-dir results-<ID> --out-dir export/
    python util/export_results_json.py results-<ID> --cross-eval-subdir cross_eval --pretty
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import platform
import re
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting._data import parse_results_arg  # noqa: E402
from plotting.figure_registry import GLOSSARY, ordered_registry  # noqa: E402
from util._diag_common import HONEYPOT_UNIVERSE, load_run_summary  # noqa: E402

SCHEMA_VERSION = "1.0.0"

# Figures intentionally off by default (heavyweight embedder dependency).
DEFAULT_SKIP = {"semantic_diversity"}

_SECTIONS = ("headline", "cross_eval", "training", "derived")


# ---------------------------------------------------------------------------
# Token-efficiency serialization helpers
# ---------------------------------------------------------------------------


def _drop_key(key: str) -> bool:
    """Render-only duplicate keys that must not reach the JSON tree."""
    return key == "_render" or key == "rep_color" or key.endswith("_raw")


def _clean(obj, nd: int = 4):
    """Recursively round floats, map NaN/inf -> None, drop render keys, prune
    None and empty-dict values. Lists, 0, and False are preserved."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        r = round(obj, nd)
        # Normalize -0.0 and integral floats for compactness.
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            ks = str(k)
            if _drop_key(ks):
                continue
            cv = _clean(v, nd)
            if cv is None:
                continue
            if isinstance(cv, dict) and not cv:
                continue
            out[ks] = cv
        return out
    if isinstance(obj, (list, tuple)):
        return [_clean(v, nd) for v in obj]
    return obj


def _dump(path: Path, obj, pretty: bool) -> None:
    if pretty:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _git_info(cwd: Path) -> tuple[str, bool | None]:
    """Best-effort git SHA + dirty flag (reused shape from plot_paper_figures)."""
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        ).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    try:
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5
        ).stdout
        dirty: bool | None = bool(dirty_out.strip())
    except Exception:
        dirty = None
    return sha, dirty


def _slug(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return s or "run"


def _ensure_writable(out_dir: Path) -> Path:
    """Create out_dir and confirm it is writable; SystemExit otherwise. No silent
    fallback — the results dir may be NFS root-squashed (owned by nobody:nogroup)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except (PermissionError, OSError) as e:
        raise SystemExit(
            f"FATAL: export dir not writable: {out_dir} ({e}). "
            f"Pass --out-dir to redirect. Refusing to silently fall back."
        )
    return out_dir


def _resolve_export_dir(selfplay_dir: str, out_dir_arg: str | None, multi: bool, label: str) -> Path:
    if out_dir_arg:
        base = Path(out_dir_arg)
        return base / _slug(label) if multi else base
    return Path(selfplay_dir) / "export"


# ---------------------------------------------------------------------------
# Per-results-dir export
# ---------------------------------------------------------------------------


def export_one(
    label: str,
    selfplay_dir: str,
    *,
    out_dir_arg: str | None,
    multi: bool,
    cross_eval_subdir: str,
    baseline_subdir: str,
    skip: set[str],
    only: set[str] | None,
    pretty: bool,
    honeypot_type: str | None,
) -> Path:
    export_dir = _ensure_writable(_resolve_export_dir(selfplay_dir, out_dir_arg, multi, label))

    sections: dict[str, dict] = {s: {} for s in _SECTIONS}
    figures_status: dict[str, dict] = {}

    for spec in ordered_registry(skip=skip, only=only):
        # Optional-dependency gate (recorded skip, not an error).
        missing = [d for d in spec.optional_deps if importlib.util.find_spec(d) is None]
        if missing:
            figures_status[spec.key] = {"status": "skipped", "reason": f"missing deps: {missing}"}
            continue

        kw = dict(spec.default_kwargs)
        if cross_eval_subdir != "cross_eval":
            for k in ("subdir", "cross_eval_subdir"):
                if k in kw:
                    kw[k] = cross_eval_subdir
        if baseline_subdir != "cross_eval_baseline" and "baseline_subdir" in kw:
            kw["baseline_subdir"] = baseline_subdir
        kw["out_dir"] = str(export_dir)  # tier-map (honeypot_tiers.json) discovery
        kw.setdefault("show_ci", True)

        try:
            metrics = spec.compute([(label, selfplay_dir)], **kw)
        except SystemExit:
            raise
        except Exception as e:  # one figure must not abort the export
            figures_status[spec.key] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3).strip().splitlines()[-1:],
            }
            continue

        if not metrics:
            figures_status[spec.key] = {"status": "empty", "reason": "no data for this figure"}
            continue

        # honeypot_difficulty seeds the tier map every tier figure reads.
        if spec.key == "honeypot_difficulty":
            tiers = {k: v for k, v in metrics.items() if k != "label"}
            try:
                (export_dir / "honeypot_tiers.json").write_text(json.dumps(tiers, indent=2))
            except OSError as e:
                print(f"[export] WARNING: could not write honeypot_tiers.json: {e}", file=sys.stderr)

        sections[spec.json_section][spec.key] = _clean(metrics)
        figures_status[spec.key] = {"status": "ok", "section": spec.json_section}

    figures_status["semantic_diversity"] = {
        "status": "skipped",
        "reason": "embedder dependency; off by default",
    }

    # ---- config.json ----------------------------------------------------
    summary = load_run_summary(selfplay_dir)
    arm = str(summary.get("honeypot_type", "")).lower()
    config = dict(summary)
    config["honeypot_universe"] = HONEYPOT_UNIVERSE.get(arm)
    _dump(export_dir / "config.json", config, pretty=True)

    # ---- section files --------------------------------------------------
    sections["headline"] = {"glossary": GLOSSARY, **sections["headline"]}
    for sec in _SECTIONS:
        _dump(export_dir / f"{sec}.json", sections[sec], pretty=pretty)

    # ---- index.json -----------------------------------------------------
    sha, dirty = _git_info(_REPO_ROOT)
    ts = datetime.datetime.now().astimezone().isoformat()
    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": ts,
        "results_dir": os.path.abspath(selfplay_dir),
        "label": label,
        "run_meta": {
            "git_sha": sha,
            "git_dirty": dirty,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": list(sys.argv),
            "honeypot_type": honeypot_type,
            "ce_refresh": not os.environ.get("CE_NO_REFRESH"),
        },
        "glossary_ref": "headline.json#/glossary",
        "figures": figures_status,
        "files": [
            "config.json",
            "headline.json",
            "cross_eval.json",
            "training.json",
            "derived.json",
            "SUMMARY.md",
        ],
    }
    _dump(export_dir / "index.json", index, pretty=True)

    # ---- SUMMARY.md -----------------------------------------------------
    (export_dir / "SUMMARY.md").write_text(_summary_md(summary, sections, figures_status, ts, sha))

    return export_dir


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------


def _fmt(x, suffix: str = "") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}{suffix}"
    return f"{x}{suffix}"


def _summary_md(summary: dict, sections: dict, figures_status: dict, ts: str, sha: str) -> str:
    head = sections.get("headline", {})
    sid = summary.get("selfplay_id", "?")
    arm = summary.get("honeypot_type", "?")
    n_iter = summary.get("num_iterations", "?")
    lines: list[str] = []
    lines.append(f"# Results export — {sid} ({arm}, {n_iter} iters)")
    lines.append("")
    lines.append(f"Generated {ts} · git {sha[:10]} · schema {SCHEMA_VERSION}")
    lines.append("")
    lines.append(
        f"Models: base={summary.get('base_model', '?')}, "
        f"red={summary.get('redteam_base_model', summary.get('base_model', '?'))}, "
        f"blue={summary.get('blueteam_base_model', summary.get('base_model', '?'))} · "
        f"scoring={summary.get('scoring_mode', '?')}"
    )
    lines.append("")
    lines.append("## Headline")

    asym = head.get("pvr_asymptote") or {}
    conv = asym.get("pvr_conv") or {}
    if conv:
        lines.append(
            f"- Diagonal PVR_conv plateau: {_fmt(conv.get('plateau_tail_mean_pct'), '%')}"
            f" ± {_fmt(conv.get('plateau_tail_std_pp'), ' pp')}"
            f" (mean {_fmt(conv.get('diag_mean_pct'), '%')}, source {asym.get('source_subdir', '?')})"
        )
    base = head.get("baseline_vs_trained") or {}
    if base:
        t = (base.get("trained") or {}).get("grand_mean_asr_pct")
        b = (base.get("baseline") or {}).get("grand_mean_asr_pct")
        lines.append(
            f"- Trained vs baseline ASR: {_fmt(t, '%')} vs {_fmt(b, '%')}"
            f" (gap ratio {_fmt(base.get('asr_gap_ratio'))})"
        )
    rank = head.get("rank_invariance") or {}
    if rank:
        lines.append(
            f"- Rank invariance: diag {_fmt(rank.get('diag_mean_pct'), '%')}"
            f" vs off-diag {_fmt(rank.get('offdiag_mean_pct'), '%')},"
            f" z={_fmt(rank.get('z_stat'))}, p={_fmt(rank.get('p_value'))} — {rank.get('verdict', '?')}"
        )
    sat = head.get("honeypot_saturation") or {}
    if sat:
        lines.append(
            f"- Honeypot universe: {sat.get('honeypot_universe', '?')} ({sat.get('honeypot_type', '?')});"
            f" eval source {sat.get('eval_source', '?')}"
        )
    held = head.get("held_out_per_style_refusal") or {}
    pooled = (held.get("pooled") or {})
    if pooled:
        parts = [f"{style}={_fmt(v.get('rate_pct'), '%')}" for style, v in pooled.items()]
        lines.append(f"- Held-out benign refusal (95% CI): {', '.join(parts)}")

    # Per-tier PVR table
    tier = head.get("tier_decomposition") or {}
    if tier:
        lines.append("")
        lines.append("## Per-tier PVR_conv (diagonal)")
        lines.append("")
        lines.append("| iter | total | PVR_conv% | pii_dominant% | harvestable% | rare% |")
        lines.append("|---|---|---|---|---|---|")
        for it in sorted(tier, key=lambda s: int(s) if str(s).isdigit() else s):
            row = tier[it]
            pii = (row.get("pii_dominant") or {}).get("pct")
            har = (row.get("harvestable") or {}).get("pct")
            rare = (row.get("rare") or {}).get("pct")
            lines.append(
                f"| {it} | {row.get('total_episodes', '?')} | {_fmt(row.get('pvr_conv_pct'))} "
                f"| {_fmt(pii)} | {_fmt(har)} | {_fmt(rare)} |"
            )

    # Glossary
    lines.append("")
    lines.append("## Glossary")
    lines.append("")
    lines.append("| key | alias | formula | source | CI |")
    lines.append("|---|---|---|---|---|")
    for key, g in GLOSSARY.items():
        lines.append(
            f"| {key} | {g.get('alias', '')} | {g.get('formula', '')} | "
            f"{g.get('source', '')} | {g.get('ci_method', '')} |"
        )

    # Figure status footer
    ok = sum(1 for v in figures_status.values() if v.get("status") == "ok")
    lines.append("")
    lines.append(
        f"_Figures: {ok} ok, "
        f"{sum(1 for v in figures_status.values() if v.get('status') == 'skipped')} skipped, "
        f"{sum(1 for v in figures_status.values() if v.get('status') == 'empty')} empty, "
        f"{sum(1 for v in figures_status.values() if v.get('status') == 'error')} error. "
        f"See index.json for per-figure detail._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export experiment results as a token-efficient JSON tree (+ SUMMARY.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("results_dirs", nargs="*", metavar="DIR[:LABEL]",
                    help="results-* dirs (positional, multi).")
    ap.add_argument("--results-dir", action="append", default=[], metavar="DIR[:LABEL]",
                    help="repeatable alternative to the positional args.")
    ap.add_argument("--out-dir", default=None,
                    help="Override export dir (default <results-dir>/export/). "
                         "Fail-fast if unwritable; for multi-dir, a per-run subdir is created.")
    ap.add_argument("--cross-eval-subdir", default="cross_eval")
    ap.add_argument("--baseline-cross-eval-subdir", default="cross_eval_baseline")
    ap.add_argument("--skip", default="", help="comma-list of figure keys to skip.")
    ap.add_argument("--only", default="", help="comma-list to restrict to (debug).")
    ap.add_argument("--pretty", action="store_true",
                    help="pretty-print all files (default: minify the big section files).")
    args = ap.parse_args()

    raw = list(args.results_dirs) + list(args.results_dir)
    if not raw:
        ap.error("no results dirs given (positional or --results-dir).")
    results = parse_results_arg(raw)

    skip = DEFAULT_SKIP | {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None

    # Fail-fast: every dir must carry summary.json with honeypot_type. Capture the
    # arm keyed by DIRECTORY (not label — labels collide when dirs share a base
    # model, which would let a genuinely mixed-arm batch slip through and be scored
    # under the wrong honeypot universe).
    arms: set[str] = set()
    for _label, d in results:
        summary = load_run_summary(d)
        arms.add(str(summary.get("honeypot_type", "")).lower())
    if len(arms) > 1:
        raise SystemExit(
            f"FATAL: mixed honeypot_type across results dirs: {sorted(arms)}. "
            "The MARFT env binds one arm per process; export each arm in a separate run."
        )

    # Set process-global HONEYPOT_TYPE before any compute triggers the lazy MARFT
    # import (the env binds the arm once at import time).
    arm = next(iter(arms))
    preset = os.environ.get("HONEYPOT_TYPE")
    if preset and preset.lower() != arm:
        raise SystemExit(
            f"FATAL: HONEYPOT_TYPE={preset!r} is preset but results summaries say {arm!r}."
        )
    os.environ["HONEYPOT_TYPE"] = arm

    multi = len(results) > 1
    produced: list[Path] = []
    for label, selfplay_dir in results:
        export_dir = export_one(
            label,
            selfplay_dir,
            out_dir_arg=args.out_dir,
            multi=multi,
            cross_eval_subdir=args.cross_eval_subdir,
            baseline_subdir=args.baseline_cross_eval_subdir,
            skip=skip,
            only=only,
            pretty=args.pretty,
            honeypot_type=arm,
        )
        produced.append(export_dir)
        print(f"[export] {label}  →  {export_dir}")

    print(f"\nExported {len(produced)} results dir(s).")


if __name__ == "__main__":
    main()
