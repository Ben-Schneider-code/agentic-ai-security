"""Shared data loading utilities and plot-style constants for plotting/."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# util.metrics is at the project root; add it to sys.path when running as a
# standalone script (the package import path handles the package case).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from util.metrics import compute_pairing_metrics, wilson_ci

# ---------------------------------------------------------------------------
# ACM two-column style constants (imported by every plot module)
# ---------------------------------------------------------------------------

RED_COL    = "#d62728"
BLUE_COL   = "#4c78a8"
GRAY_COL   = "#888888"
GREEN_COL  = "#2ca02c"
HUMAN_COL  = "#ff7f0e"
BENIGN_COL = "#17becf"

RED_MARKER    = "o"
BLUE_MARKER   = "s"
HUMAN_MARKER  = "*"
BENIGN_MARKER = "^"

FIG_SIZE_SINGLE = (5.0, 4.8)
FIG_SIZE_1x2    = (10.0, 4.8)
FIG_SIZE_1x3    = (14.5, 4.8)

# Ordered colour cycle for multi-run overlays
RUN_COLORS = [RED_COL, BLUE_COL, GREEN_COL, "#9467bd", HUMAN_COL, GRAY_COL]


def apply_paper_style() -> None:
    """Apply ACM two-column rcParams + seaborn theme. Call once at module level."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    sns.set_theme(context="paper", style="whitegrid")


# ---------------------------------------------------------------------------
# Checkpoint / run-dir discovery
# ---------------------------------------------------------------------------

def find_run_dir(team_dir: Path) -> Path | None:
    """Walk team_dir depth-first; return the first directory containing training_state.json."""
    for root, _dirs, files in os.walk(team_dir):
        if "training_state.json" in files:
            return Path(root)
    return None


def discover_iterations(selfplay_dir: str) -> list[dict]:
    """
    Scan selfplay_dir for iter_N/ subdirectories.

    A team's directory is included only when its .success marker exists.
    Missing-marker directories are excluded with a warning printed to stderr.

    Returns a list sorted by N:
        [{"iter": int, "red_dir": Path | None, "blue_dir": Path | None}, ...]
    """
    base = Path(selfplay_dir)
    iters: dict[int, dict] = {}

    for entry in sorted(base.iterdir()):
        m = re.match(r"^iter_(\d+)$", entry.name)
        if not (m and entry.is_dir()):
            continue
        n = int(m.group(1))
        iters[n] = {"iter": n, "red_dir": None, "blue_dir": None}

        for team, key in (("redteam", "red_dir"), ("blueteam", "blue_dir")):
            team_dir = entry / team
            if not team_dir.is_dir():
                continue
            if not (team_dir / ".success").is_file():
                print(
                    f"WARNING: iter_{n}/{team} missing .success — skipping "
                    f"(training may be incomplete)",
                    file=sys.stderr,
                )
                continue
            iters[n][key] = team_dir

    return [iters[k] for k in sorted(iters)]


# ---------------------------------------------------------------------------
# Training state / args
# ---------------------------------------------------------------------------

def load_training_state(run_dir: Path) -> dict:
    """Load training_state.json from run_dir. Raises FileNotFoundError if absent."""
    path = run_dir / "training_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"training_state.json not found in: {run_dir}")
    with open(path) as f:
        return json.load(f)


def load_args_yaml(run_dir: Path) -> dict:
    """
    Parse args.yaml using regex (avoids python/tuple YAML tag incompatibility).
    Returns a flat dict of key → str | int | float values.
    """
    path = run_dir / "args.yaml"
    if not path.is_file():
        return {}
    result: dict[str, object] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        try:
            result[key] = int(raw)
        except ValueError:
            try:
                result[key] = float(raw)
            except ValueError:
                result[key] = raw
    return result


# ---------------------------------------------------------------------------
# reward_debug.jsonl loading
# ---------------------------------------------------------------------------

def _find_reward_debug(run_dir: Path) -> Path | None:
    for candidate in (
        run_dir / "reward_debug.jsonl",
        run_dir / "debug_logs" / "reward_debug.jsonl",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_reward_debug_lines(
    run_dir: Path,
    mode: str = "training_time",
    tail_pct: float = 0.25,
    turn_type: str | None = None,
) -> list[dict]:
    """
    Load reward_debug.jsonl from run_dir (or run_dir/debug_logs/).

    mode:
        "training_time"   — all lines
        "final_episode"   — last tail_pct fraction of unique episode indices
        "evaluation_time" — all lines (caller should point run_dir at a cross-eval
                            pairings subdir; field names may differ)
    turn_type: optional filter ("attack" | "benign" | None=both).
    Returns [] if the file is not found.
    """
    fpath = _find_reward_debug(run_dir)
    if fpath is None:
        return []

    lines: list[dict] = []
    with open(fpath) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    if turn_type is not None:
        def _matches_turn_type(ln: dict) -> bool:
            t = ln.get("turn_type")
            if t is not None:
                return t == turn_type
            # Logs without turn_type: infer attack from presence of query fields
            if turn_type == "attack":
                return any(ln.get(f) for f in ("red_team_input", "user_message"))
            return False
        lines = [ln for ln in lines if _matches_turn_type(ln)]

    if mode == "final_episode":
        episodes = sorted({ln.get("episode", 0) for ln in lines})
        if episodes:
            cutoff = max(0, int(len(episodes) * (1.0 - tail_pct)))
            keep = set(episodes[cutoff:])
            lines = [ln for ln in lines if ln.get("episode", 0) in keep]

    return lines


def get_attack_query(line: dict) -> str | None:
    """Extract attack prompt text from a reward_debug.jsonl record."""
    for field in ("red_team_input", "user_message"):
        val = line.get(field)
        if val:
            return str(val)
    return None


# ---------------------------------------------------------------------------
# Cross-eval data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Benign denial helpers — shared by plot_brr and plot_per_style_refusal
# ---------------------------------------------------------------------------

def is_benign_denial(row: dict) -> bool:
    """Canonical denial predicate. Must stay in sync with per_style_pud_trend.py:81."""
    return bool(row.get("is_refusal")) or row.get("outcome_tier") == "false_negative"


def wilson_ci_pct(k: int, n: int, z: float = 2.576) -> tuple[float, float]:
    """Wilson 99% CI on a proportion; returns (lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, (centre - margin) * 100.0), min(100.0, (centre + margin) * 100.0)


def denial_rate_with_ci(
    rows: "list[dict]", z: float = 2.576
) -> "tuple[float, float, float, int, int]":
    """
    (rate_pct, ci_lo_pct, ci_hi_pct, k, n) for any iterable of reward_debug rows.
    Denial predicate: is_benign_denial. CI: Wilson 99% via wilson_ci_pct.
    """
    rows = list(rows)
    n = len(rows)
    k = sum(1 for r in rows if is_benign_denial(r))
    if n == 0:
        return float("nan"), 0.0, 100.0, 0, 0
    rate = k / n * 100.0
    lo, hi = wilson_ci_pct(k, n, z)
    return rate, lo, hi, k, n


def load_benign_eval_per_turn(selfplay_dir: str) -> "dict[int, list[dict]]":
    """
    Load per-turn rows from benign_eval/benign_only/blue_*/reward_debug.jsonl.
    Returns {blue_iter: [row, ...]} for each saved blue checkpoint.
    """
    base = Path(selfplay_dir) / "benign_eval" / "benign_only"
    result: dict[int, list[dict]] = {}
    if not base.is_dir():
        return result
    for entry in sorted(base.iterdir()):
        m = re.match(r"^blue_(\d+)$", entry.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        jsonl = entry / "reward_debug.jsonl"
        if not jsonl.is_file():
            continue
        rows: list[dict] = []
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("turn_type") == "benign":
                    rows.append(r)
        result[iter_num] = rows
    return result


def load_cross_eval_benign_per_turn(
    selfplay_dir: str, subdir: str = "cross_eval"
) -> "dict[int, list[dict]]":
    """
    Load per-turn rows from <selfplay_dir>/<subdir>/benign_only/blue_*/reward_debug.jsonl.

    These are the benign turns collected as part of the cross-eval pass (one
    benign run per blue checkpoint, independent of red). Returns {blue_iter:
    [row, ...]}; rows are filtered to turn_type == "benign" when the field is
    present, else kept as-is (the directory is benign-only by construction).
    Falls back to <subdir>_quick if the primary subdir is absent.
    """
    base_root = Path(selfplay_dir)
    base: Path | None = None
    for candidate in (subdir, f"{subdir}_quick"):
        cand = base_root / candidate / "benign_only"
        if cand.is_dir():
            base = cand
            break
    if base is None:
        return {}

    result: dict[int, list[dict]] = {}
    for entry in sorted(base.iterdir()):
        m = re.match(r"^blue_(\d+)$", entry.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        jsonl = entry / "reward_debug.jsonl"
        if not jsonl.is_file():
            continue
        rows: list[dict] = []
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tt = r.get("turn_type")
                if tt is None or tt == "benign":
                    rows.append(r)
        result[iter_num] = rows
    return result


def load_train_rollout_benign_turns(selfplay_dir: str) -> "dict[int, list[dict]]":
    """
    Load training-time benign turns from iter_*/blueteam/**/debug_logs/reward_debug.jsonl.
    Filters: turn_type=="benign" AND not is_eval — identical population to per_style_pud_trend.py.
    Returns {iter: [row, ...]}.
    """
    base = Path(selfplay_dir)
    result: dict[int, list[dict]] = {}
    for iter_dir in sorted(base.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            iter_num = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        candidates = list(iter_dir.glob("blueteam/**/debug_logs/reward_debug.jsonl"))
        if not candidates:
            continue
        rows: list[dict] = []
        with open(candidates[0]) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("turn_type") == "benign" and not r.get("is_eval"):
                    rows.append(r)
        result[iter_num] = rows
    return result


def load_benign_only(selfplay_dir: str, cross_eval_subdir: str = "cross_eval") -> dict[str, dict]:
    """
    Load benign-only TPR data keyed by "blue_N".

    Priority:
      1. <selfplay_dir>/benign_eval/benign_eval_results.json  (dedicated benign eval)
      2. <selfplay_dir>/<cross_eval_subdir>/cross_eval_results.json["benign_only"]
    Returns {} if neither source has data.
    """
    benign_eval_path = Path(selfplay_dir) / "benign_eval" / "benign_eval_results.json"
    if benign_eval_path.is_file():
        with open(benign_eval_path) as f:
            data = json.load(f)
        bo = data.get("benign_only", {})
        if bo:
            return bo

    cross_eval = load_cross_eval_results(selfplay_dir, cross_eval_subdir)
    if cross_eval:
        return cross_eval.get("benign_only", {})
    return {}


_CROSS_EVAL_CACHE: dict[tuple[str, str], dict] = {}


def _refresh_pairings_from_jsonl(data: dict, pairings_dir: Path) -> int:
    """Re-aggregate every pairing from raw reward_debug.jsonl.

    Mutates `data["pairings"]` in place; returns count of refreshed pairings.
    Used to override stale ASR/PVR values from cached cross_eval_results.json
    after a fix to compute_pairing_metrics.
    """
    from util.metrics import compute_pairing_metrics  # local to avoid cycles
    pairings = data.setdefault("pairings", {})
    n = 0
    for pdir in sorted(pairings_dir.iterdir()):
        if not pdir.is_dir():
            continue
        m = re.match(r"^red_(\d+)_blue_(\d+)$", pdir.name)
        if not m:
            continue
        ri, bi = int(m.group(1)), int(m.group(2))
        jsonl = pdir / "reward_debug.jsonl"
        if not jsonl.is_file():
            continue
        records: list[dict] = []
        with open(jsonl) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        res = compute_pairing_metrics(records)
        entry = pairings.setdefault(pdir.name, {})
        entry["n_attack_episodes"] = res["n_attack_episodes"]
        entry["n_benign_episodes"] = res["n_benign_episodes"]
        entry["n_total_records"] = res["n_total_records"]
        entry["metrics"] = res["metrics"]
        entry["confidence_intervals"] = res["confidence_intervals"]
        entry["episode_stats"] = res["episode_stats"]
        entry["red_iter"] = ri
        entry["blue_iter"] = bi
        entry["pairing_key"] = pdir.name
        n += 1
    return n


def load_cross_eval_results(selfplay_dir: str, subdir: str = "cross_eval") -> dict | None:
    """
    Load {selfplay_dir}/{subdir}/cross_eval_results.json.
    Falls back to {subdir}_quick. Returns None if neither exists.

    If the matching pairings/ directory exists, ALL pairings are re-aggregated
    in-memory from raw reward_debug.jsonl via util.metrics.compute_pairing_metrics.
    This guarantees the fast path returns the same numbers as the slow path —
    important when the on-disk JSON is stale relative to a metrics.py fix and
    the results dir is read-only (e.g., owned by a container user). Set the env
    var CE_NO_REFRESH=1 to suppress refresh and use cached values verbatim.
    Results are cached in memory keyed by (resolved_dir, subdir).
    """
    import os
    cache_key = (str(Path(selfplay_dir).resolve()), subdir)
    if cache_key in _CROSS_EVAL_CACHE:
        return _CROSS_EVAL_CACHE[cache_key]

    base = Path(selfplay_dir)
    for candidate in (subdir, f"{subdir}_quick"):
        path = base / candidate / "cross_eval_results.json"
        if not path.is_file():
            continue
        with open(path) as f:
            data = json.load(f)
        pairings_dir = base / candidate / "pairings"
        if pairings_dir.is_dir() and not os.environ.get("CE_NO_REFRESH"):
            n = _refresh_pairings_from_jsonl(data, pairings_dir)
            data.setdefault("metadata", {})["refreshed_in_memory"] = {
                "n_pairings": n,
                "source": str(pairings_dir),
                "reason": (
                    "Stale on-disk JSON: re-aggregated via compute_pairing_metrics "
                    "to apply the corrected PVR_conv denominator (n_eps_with_sql)."
                ),
            }
        _CROSS_EVAL_CACHE[cache_key] = data
        return data
    return None


def load_pairing_metrics_with_decomposed(
    selfplay_dir: str, subdir: str = "cross_eval"
) -> dict | None:
    """
    Like load_cross_eval_results, but guarantees decomposed-metric keys exist.

    If cross_eval_results.json already contains episode_stats + pvr_sql_turn for
    every pairing, it is returned as-is (fast path). Otherwise, walks
    <selfplay_dir>/<subdir>/pairings/red_*_blue_*/reward_debug.jsonl, re-runs
    compute_pairing_metrics (pure Python, no GPU), and merges the new keys in.
    Non-destructive — the JSON file on disk is never modified.
    Falls back to load_cross_eval_results if no pairings directory exists.
    """
    cross_eval = load_cross_eval_results(selfplay_dir, subdir=subdir)
    if cross_eval is None:
        return None

    pairings = cross_eval.get("pairings", {})
    _DECOMPOSED_KEYS = {"pvr_sql_turn", "work_factor", "coverage_pct", "yield_pct"}

    # Fast path: all pairings already have the new keys.
    if pairings and all(
        _DECOMPOSED_KEYS <= set(p.get("metrics", {}))
        for p in pairings.values()
    ):
        return cross_eval

    # Slow path: re-aggregate from raw JSONL.
    pairings_dir = Path(selfplay_dir) / subdir / "pairings"
    if not pairings_dir.is_dir():
        # No raw JSONL available — return cached data as-is.
        return cross_eval

    upgraded = 0
    for pairing_dir in sorted(pairings_dir.iterdir()):
        if not pairing_dir.is_dir():
            continue
        m = re.match(r"^red_(\d+)_blue_(\d+)$", pairing_dir.name)
        if not m:
            continue
        ri, bi = int(m.group(1)), int(m.group(2))
        key = pairing_dir.name

        jsonl = pairing_dir / "reward_debug.jsonl"
        if not jsonl.is_file():
            continue

        records: list[dict] = []
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        result = compute_pairing_metrics(records)

        # Merge new keys into the cached pairing dict.
        if key not in pairings:
            pairings[key] = {}
        pairing = pairings[key]
        pairing.setdefault("metrics", {}).update(result["metrics"])
        pairing.setdefault("confidence_intervals", {}).update(
            result["confidence_intervals"]
        )
        pairing["episode_stats"] = result["episode_stats"]
        pairing.setdefault("red_iter", ri)
        pairing.setdefault("blue_iter", bi)
        pairing.setdefault("pairing_key", key)
        upgraded += 1

    if upgraded:
        print(
            f"[_data] Upgraded {upgraded} pairings with decomposed metrics "
            f"(from {subdir}/pairings/*/reward_debug.jsonl).",
            file=sys.stderr,
        )

    return cross_eval


def extract_diagonal_metrics(
    cross_eval: dict, benign_only: dict | None = None
) -> dict[int, dict]:
    """
    Extract pairings where red_iter == blue_iter from cross_eval_results.json.

    Returns {iter_num: metrics_dict} with metrics on the 0–100 scale.
    CIs are attached as <key>_ci.
    Pass benign_only (from load_benign_only) to override TPR with dedicated
    benign-only measurements.
    """
    pairings = cross_eval.get("pairings", {})
    bo_data = benign_only or {}

    result: dict[int, dict] = {}
    for _key, pairing in pairings.items():
        ri = pairing.get("red_iter")
        bi = pairing.get("blue_iter")
        if ri is None or bi is None or ri != bi:
            continue
        m = dict(pairing.get("metrics", {}))
        for k, v in pairing.get("confidence_intervals", {}).items():
            m[f"{k}_ci"] = v
        # Carry the raw numerators/denominators so downstream plots can report
        # "k breaches / n episodes" next to each PVR% (a single event must not be
        # readable as a trend when n is tiny).
        m["raw_counts"] = pairing.get("raw_counts", {})
        bo = bo_data.get(f"blue_{bi}")
        if bo:
            if "tpr" in bo:
                m["tpr"] = bo["tpr"]
            if "tpr_ci" in bo:
                m["tpr_ci"] = bo["tpr_ci"]
        result[ri] = m

    return result


# ---------------------------------------------------------------------------
# Diagonal matrix from per-iteration blueteam training logs
# ---------------------------------------------------------------------------


def build_diagonal_matrix(
    selfplay_dir: str,
    *,
    source: str = "training",
) -> tuple:
    """
    Build a diagonal-only N×N metric matrix from per-iteration blueteam logs.

    source="training": PVR_conv (ASR) computed by compute_pairing_metrics on
        all non-eval training records (is_eval=False). Uses the same episode-
        level logic as cross_evaluate.py so results are directly comparable.
    source="eval": TPR (1−PUD) from benign eval turns (is_eval=True).

    Returns (mat, ci_lo, ci_hi, iter_nums, metric_label).
    mat is N×N with NaN everywhere off the diagonal.
    """
    import numpy as np

    iters = [it for it in discover_iterations(selfplay_dir) if it["blue_dir"] is not None]
    metric_label = r"$\mathrm{PVR_{conv}}$" if source == "training" else r"TPR $(1-\mathrm{PUD})$"
    if not iters:
        empty = np.array([])
        return empty, empty, empty, [], metric_label

    n = len(iters)
    iter_nums = [it["iter"] for it in iters]
    mat       = np.full((n, n), np.nan)
    ci_lo_mat = np.full((n, n), np.nan)
    ci_hi_mat = np.full((n, n), np.nan)

    for idx, it_info in enumerate(iters):
        run_dir = find_run_dir(it_info["blue_dir"])
        if run_dir is None:
            continue
        records = load_reward_debug_lines(run_dir, mode="training_time")
        if source == "training":
            # Filter to non-eval records; compute_pairing_metrics groups by episode
            # and computes PVR_conv (asr) as fraction of attack episodes with ≥1
            # false_positive — identical to the cross_eval definition.
            training_records = [r for r in records if not r.get("is_eval", False)]
            result = compute_pairing_metrics(training_records)
            val = result["metrics"]["asr"]
            lo, hi = result["confidence_intervals"]["asr"]
        else:
            # eval: fraction of benign eval turns the blue team answered correctly
            benign_eval = [
                r for r in records
                if r.get("is_eval", False) and r.get("turn_type") == "benign"
            ]
            if not benign_eval:
                continue
            k = sum(1 for r in benign_eval if r.get("outcome_tier") == "true_positive")
            n_b = len(benign_eval)
            lo, hi = wilson_ci(k, n_b)
            val = k / n_b * 100.0
        if not (val != val):  # skip NaN
            mat[idx, idx]       = val
            ci_lo_mat[idx, idx] = lo
            ci_hi_mat[idx, idx] = hi

    return mat, ci_lo_mat, ci_hi_mat, iter_nums, metric_label


# ---------------------------------------------------------------------------
# Human-eval summaries
# ---------------------------------------------------------------------------

def load_human_eval_summaries(human_eval_parent: str) -> dict[int, dict]:
    """
    Scan human_eval_parent for iter_N/summary.json (matches iter_N or iter_N_*).
    Returns {iter_num: summary_dict} where metric values are on the 0–1 scale.
    Returns {} if parent directory does not exist.
    """
    base = Path(human_eval_parent)
    if not base.is_dir():
        return {}

    result: dict[int, dict] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        m = re.match(r"^iter_(\d+)", entry.name)
        if not m:
            continue
        n = int(m.group(1))
        summary = entry / "summary.json"
        if summary.is_file():
            with open(summary) as f:
                result[n] = json.load(f)
    return result


# ---------------------------------------------------------------------------
# Label / base model detection
# ---------------------------------------------------------------------------

def get_base_model_label(selfplay_dir: str) -> str:
    """
    Read base_model from the first args.yaml found under selfplay_dir.
    Returns the last path component of the model ID (e.g. "Llama-3.1-8B-Instruct").
    Falls back to the basename of selfplay_dir.
    """
    base = Path(selfplay_dir)
    for root, _dirs, files in os.walk(base):
        if "args.yaml" in files:
            args = load_args_yaml(Path(root))
            raw = args.get("base_model") or args.get("base_model_name_or_path")
            if raw and isinstance(raw, str):
                return raw.rstrip("/").split("/")[-1]
            break
    return base.name


# ---------------------------------------------------------------------------
# Human jailbreak query loading
# ---------------------------------------------------------------------------

def load_human_queries(queries_file: str) -> list[str]:
    """
    Parse new_jailbreaks.txt:
      - Lines starting with '#' are comments/headers and are skipped.
      - Blank lines separate prompt blocks.
      - Consecutive non-blank, non-comment lines within a block are joined
        as one prompt (space-separated).

    Returns a list of prompt strings (expected: ~32 prompts).
    """
    path = Path(queries_file)
    if not path.is_file():
        raise FileNotFoundError(f"Human queries file not found: {queries_file}")

    prompts: list[str] = []
    current_block: list[str] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if not stripped:
            if current_block:
                prompts.append(" ".join(current_block))
                current_block = []
        else:
            current_block.append(stripped)

    if current_block:
        prompts.append(" ".join(current_block))

    return prompts


def load_benign_queries(
    benign_file: str = "data/benign_pool_stats.json",
    splits: tuple[str, ...] = ("train_pool", "eval_pool"),
) -> list[str]:
    """
    Load benign queries from benign_pool_stats.json.

    Flattens multi-turn entries (each turn becomes its own string) and returns
    deduplicated query texts across the requested splits.
    """
    path = Path(benign_file)
    if not path.is_file():
        raise FileNotFoundError(f"Benign queries file not found: {benign_file}")

    with open(path) as f:
        data = json.load(f)

    seen: set[str] = set()
    queries: list[str] = []
    for split in splits:
        pool = data.get(split, [])
        for entry in pool:
            if entry.get("type") == "multi_turn":
                turns = entry.get("turns", [])
            else:
                turns = [entry.get("text", "")]
            for t in turns:
                if not isinstance(t, str):
                    continue
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    queries.append(t)

    return queries


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _enrich_result_entry(label: str, selfplay_dir: str) -> dict:
    """Build a result-entry dict with abs_path + parsed summary.json (best-effort)."""
    import json

    entry: dict = {"label": label, "selfplay_dir": selfplay_dir}
    try:
        entry["abs_path"] = os.path.abspath(selfplay_dir)
    except Exception:
        entry["abs_path"] = None

    summary_path = os.path.join(selfplay_dir, "summary.json")
    try:
        with open(summary_path) as f:
            entry["summary"] = json.load(f)
    except Exception:
        entry["summary"] = None
    return entry


def write_sidecar(
    png_path: Path,
    description: str,
    results: list[tuple[str, str]],
    metrics: dict | None = None,
    *,
    plot_kwargs: dict | None = None,
    run_meta: dict | None = None,
) -> Path:
    """Write a JSON sidecar alongside a PNG with description, timestamp, metrics,
    per-plot kwargs, and a run-level metadata snapshot."""
    import json
    import datetime

    ts = datetime.datetime.now().astimezone().isoformat()
    payload = {
        "description": description,
        "generated_at": ts,
        "results": [_enrich_result_entry(lbl, d) for lbl, d in results],
        "metrics": metrics or {},
        "plot_kwargs": plot_kwargs or {},
        "run_meta": run_meta or {},
    }
    sidecar = png_path.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2, default=str))
    return sidecar


def parse_results_arg(raw: list[str]) -> list[tuple[str, str]]:
    """
    Parse --results arguments of the form DIR or DIR:Label.

    For absolute paths (starting with '/'), the label is separated by the LAST
    colon after the path; the path itself must exist on disk.
    Returns [(label, selfplay_dir), ...].
    """
    out: list[tuple[str, str]] = []
    for r in raw:
        # Try rfind(":") as label separator
        idx = r.rfind(":")
        if idx > 0:
            path_part = r[:idx]
            label_part = r[idx + 1:]
            if os.path.exists(path_part) and label_part:
                out.append((label_part, path_part))
                continue
        # No label — auto-detect from args.yaml
        out.append((get_base_model_label(r), r))
    return out
