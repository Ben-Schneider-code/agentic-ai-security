#!/usr/bin/env python3
"""
Analyze and visualize cross-evaluation results.

Reads cross_eval_results.json produced by cross_evaluate.py and generates:
  1. Win Rate Heatmap (ASR matrix)
  2. Dominance Heatmap
  3. Bradley-Terry Strength Ratings
  4. Generalization Analysis
  5. TPR vs TNR Pareto Frontier
  6. Nash Equilibrium Support
  + LaTeX tables + text summary

Usage:
    python util/plot_cross_eval.py <cross_eval_dir>

Example:
    python util/plot_cross_eval.py results-20260322-1641-m92p4/cross_eval/
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ──────────────────────────── Data Loading ──────────────────────────────────


def load_results(cross_eval_dir: str) -> dict:
    results_path = os.path.join(cross_eval_dir, "cross_eval_results.json")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run cross_evaluate.py first.")
        sys.exit(1)
    with open(results_path, "r") as f:
        return json.load(f)


def build_matrices(results: dict) -> tuple:
    """Build numpy matrices from results.

    Returns:
        (red_versions, blue_versions, asr_matrix, tnr_matrix, cfr_matrix, dominance_matrix,
         tpr_matrix, f1_matrix)
    """
    pairings = results["pairings"]
    if not pairings:
        print("Error: No pairing data found.")
        sys.exit(1)

    # Extract unique red and blue versions from pairing keys
    red_set = set()
    blue_set = set()
    for key in pairings:
        parts = key.split("_")
        red_set.add(int(parts[1]))
        blue_set.add(int(parts[3]))

    red_versions = sorted(red_set)
    blue_versions = sorted(blue_set)
    n_red = len(red_versions)
    n_blue = len(blue_versions)

    red_idx = {v: i for i, v in enumerate(red_versions)}
    blue_idx = {v: i for i, v in enumerate(blue_versions)}

    asr = np.full((n_red, n_blue), np.nan)
    tnr = np.full((n_red, n_blue), np.nan)
    cfr = np.full((n_red, n_blue), np.nan)
    dom = np.full((n_red, n_blue), np.nan)
    tpr = np.full((n_red, n_blue), np.nan)
    f1 = np.full((n_red, n_blue), np.nan)

    for key, data in pairings.items():
        ri = red_idx[data["red_iter"]]
        bi = blue_idx[data["blue_iter"]]
        m = data["metrics"]
        asr[ri, bi] = m["asr"]
        tnr[ri, bi] = m["tnr"]
        cfr[ri, bi] = m["cfr"]
        dom[ri, bi] = m["dominance"]
        tpr[ri, bi] = m["tpr"]
        f1[ri, bi] = m["f1"]

    return red_versions, blue_versions, asr, tnr, cfr, dom, tpr, f1


# ──────────────────────────── Bradley-Terry Model ───────────────────────────


def fit_bradley_terry(asr_matrix: np.ndarray, red_versions: list, blue_versions: list) -> dict:
    """Fit Bradley-Terry model from the ASR (win rate) matrix.

    Treats ASR/100 as P(red_i beats blue_j). Fits strength parameters via
    iterative MLE (no scipy needed — uses the classic iterative algorithm).

    Returns dict with 'red_ratings', 'blue_ratings' (lists of floats).
    """
    n_red = len(red_versions)
    n_blue = len(blue_versions)

    # Initialize strengths
    red_strength = np.ones(n_red)
    blue_strength = np.ones(n_blue)

    # Number of "games" per cell (assume 100 if not specified)
    # We use ASR as win fraction directly
    n_games = 100  # approximate

    max_iter = 200
    tol = 1e-6

    for iteration in range(max_iter):
        old_red = red_strength.copy()
        old_blue = blue_strength.copy()

        # Update red strengths
        for i in range(n_red):
            wins_i = 0.0
            denom_i = 0.0
            for j in range(n_blue):
                if np.isnan(asr_matrix[i, j]):
                    continue
                w_ij = asr_matrix[i, j] / 100 * n_games
                n_ij = n_games
                wins_i += w_ij
                denom_i += n_ij / (red_strength[i] + blue_strength[j])
            if denom_i > 0:
                red_strength[i] = wins_i / denom_i

        # Update blue strengths
        for j in range(n_blue):
            wins_j = 0.0
            denom_j = 0.0
            for i in range(n_red):
                if np.isnan(asr_matrix[i, j]):
                    continue
                w_ji = (1 - asr_matrix[i, j] / 100) * n_games
                n_ij = n_games
                wins_j += w_ji
                denom_j += n_ij / (red_strength[i] + blue_strength[j])
            if denom_j > 0:
                blue_strength[j] = wins_j / denom_j

        # Normalize (anchor first red to 1.0)
        scale = red_strength[0] if red_strength[0] > 0 else 1.0
        red_strength /= scale
        blue_strength /= scale

        # Check convergence
        delta = max(
            np.max(np.abs(red_strength - old_red)),
            np.max(np.abs(blue_strength - old_blue)),
        )
        if delta < tol:
            break

    # Approximate confidence intervals via bootstrap-like variance
    # Using the Fisher information approximation
    red_se = np.zeros(n_red)
    blue_se = np.zeros(n_blue)
    for i in range(n_red):
        info = 0.0
        for j in range(n_blue):
            if np.isnan(asr_matrix[i, j]):
                continue
            p = red_strength[i] / (red_strength[i] + blue_strength[j])
            info += n_games * p * (1 - p) / (red_strength[i] ** 2)
        red_se[i] = 1.0 / math.sqrt(info) if info > 0 else 0.0

    for j in range(n_blue):
        info = 0.0
        for i in range(n_red):
            if np.isnan(asr_matrix[i, j]):
                continue
            p = blue_strength[j] / (red_strength[i] + blue_strength[j])
            info += n_games * p * (1 - p) / (blue_strength[j] ** 2)
        blue_se[j] = 1.0 / math.sqrt(info) if info > 0 else 0.0

    return {
        "red_ratings": red_strength.tolist(),
        "blue_ratings": blue_strength.tolist(),
        "red_se": red_se.tolist(),
        "blue_se": blue_se.tolist(),
    }


# ──────────────────────────── Nash Equilibrium ──────────────────────────────


def compute_nash_equilibrium(asr_matrix: np.ndarray) -> tuple:
    """Compute Nash equilibrium mixed strategies via linear programming.

    The game: Red chooses a row (attack version), Blue chooses a column (defense).
    Payoff = ASR for Red, (100 - ASR) for Blue.

    Returns (red_mixture, blue_mixture, game_value) where mixtures are probability vectors.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        print("Warning: scipy not available, skipping Nash equilibrium computation.")
        return None, None, None

    # Replace NaN with 50% (neutral) for the LP
    M = np.nan_to_num(asr_matrix, nan=50.0) / 100.0  # Normalize to [0, 1]
    n_red, n_blue = M.shape

    # Solve for Red's maximin strategy
    # Maximize v subject to: sum_i(x_i * M[i,j]) >= v for all j, sum(x) = 1, x >= 0
    # Equivalent LP: minimize -v
    # Variables: [x_0, ..., x_{n_red-1}, v]
    c = np.zeros(n_red + 1)
    c[-1] = -1  # minimize -v

    # Constraints: v - sum_i(x_i * M[i,j]) <= 0 for each j
    A_ub = np.zeros((n_blue, n_red + 1))
    for j in range(n_blue):
        A_ub[j, :n_red] = -M[:, j]
        A_ub[j, -1] = 1
    b_ub = np.zeros(n_blue)

    # Equality constraint: sum(x) = 1
    A_eq = np.zeros((1, n_red + 1))
    A_eq[0, :n_red] = 1
    b_eq = [1.0]

    bounds = [(0, None)] * n_red + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if result.success:
        red_mixture = result.x[:n_red]
        game_value = result.x[-1]
    else:
        red_mixture = np.ones(n_red) / n_red
        game_value = 0.5

    # Solve for Blue's minimax strategy
    # Minimize w subject to: sum_j(y_j * M[i,j]) <= w for all i, sum(y) = 1, y >= 0
    c2 = np.zeros(n_blue + 1)
    c2[-1] = 1  # minimize w

    A_ub2 = np.zeros((n_red, n_blue + 1))
    for i in range(n_red):
        A_ub2[i, :n_blue] = M[i, :]
        A_ub2[i, -1] = -1
    b_ub2 = np.zeros(n_red)

    A_eq2 = np.zeros((1, n_blue + 1))
    A_eq2[0, :n_blue] = 1
    b_eq2 = [1.0]

    bounds2 = [(0, None)] * n_blue + [(None, None)]

    result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if result2.success:
        blue_mixture = result2.x[:n_blue]
    else:
        blue_mixture = np.ones(n_blue) / n_blue

    return red_mixture, blue_mixture, game_value


# ──────────────────────────── Transitivity ──────────────────────────────────


def compute_transitivity(asr_matrix: np.ndarray) -> float:
    """Compute transitivity score: fraction of ordered triples that satisfy transitivity.

    For each triple (i, j, k) of red versions evaluated against a fixed blue version,
    check if red_i > red_j > red_k (in terms of ASR) implies red_i > red_k.
    We average across all blue versions.
    """
    n_red, n_blue = asr_matrix.shape
    if n_red < 3:
        return 1.0  # Trivially transitive

    total_triples = 0
    transitive_triples = 0

    for b in range(n_blue):
        col = asr_matrix[:, b]
        valid = ~np.isnan(col)
        valid_indices = np.where(valid)[0]
        n = len(valid_indices)
        for a in range(n):
            for c in range(a + 1, n):
                for d in range(c + 1, n):
                    i, j, k = valid_indices[a], valid_indices[c], valid_indices[d]
                    vi, vj, vk = col[i], col[j], col[k]
                    # Check all orderings
                    total_triples += 1
                    if (vi >= vj >= vk) or (vk >= vj >= vi):
                        transitive_triples += 1

    return transitive_triples / total_triples if total_triples > 0 else 1.0


# ──────────────────────────── Plotting ──────────────────────────────────────


def plot_heatmap_asr(ax, red_versions, blue_versions, asr_matrix):
    """Figure 1: Win Rate (ASR) Heatmap."""
    im = ax.imshow(asr_matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    # Annotate cells
    for i in range(len(red_versions)):
        for j in range(len(blue_versions)):
            val = asr_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 60 or val < 20 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    # Highlight near-diagonal (training-adjacent pairings)
    for i, rv in enumerate(red_versions):
        for j, bv in enumerate(blue_versions):
            if rv > 0 and bv > 0 and rv == bv:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="gold", linewidth=2.5))

    ax.set_xticks(range(len(blue_versions)))
    ax.set_xticklabels([f"B{v}" for v in blue_versions], fontsize=8)
    ax.set_yticks(range(len(red_versions)))
    ax.set_yticklabels([f"R{v}" for v in red_versions], fontsize=8)
    ax.set_xlabel("Blue Team Version", fontsize=10)
    ax.set_ylabel("Red Team Version", fontsize=10)
    ax.set_title("Attack Success Rate (ASR %)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="ASR %", shrink=0.8)


def plot_heatmap_dominance(ax, red_versions, blue_versions, dom_matrix):
    """Figure 2: Dominance Score Heatmap."""
    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(dom_matrix)), abs(np.nanmax(dom_matrix)), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(dom_matrix, cmap="RdBu", norm=norm, aspect="auto")

    for i in range(len(red_versions)):
        for j in range(len(blue_versions)):
            val = dom_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_xticks(range(len(blue_versions)))
    ax.set_xticklabels([f"B{v}" for v in blue_versions], fontsize=8)
    ax.set_yticks(range(len(red_versions)))
    ax.set_yticklabels([f"R{v}" for v in red_versions], fontsize=8)
    ax.set_xlabel("Blue Team Version", fontsize=10)
    ax.set_ylabel("Red Team Version", fontsize=10)
    ax.set_title("Dominance Score (+ = Blue, \u2212 = Red)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Dominance", shrink=0.8)


def plot_bt_ratings(ax, red_versions, blue_versions, bt_result):
    """Figure 3: Bradley-Terry Strength Ratings."""
    red_ratings = bt_result["red_ratings"]
    blue_ratings = bt_result["blue_ratings"]
    red_se = bt_result["red_se"]
    blue_se = bt_result["blue_se"]

    x_red = np.arange(len(red_versions))
    x_blue = np.arange(len(blue_versions))
    width = 0.35

    ax.bar(x_red - width / 2, red_ratings, width, yerr=[s * 1.96 for s in red_se],
           color="#e74c3c", alpha=0.8, label="Red (Attacker)", capsize=3)
    ax.bar(x_blue + width / 2, blue_ratings, width, yerr=[s * 1.96 for s in blue_se],
           color="#3498db", alpha=0.8, label="Blue (Defender)", capsize=3)

    all_labels = [f"R{v}/B{v}" if v in red_versions and v in blue_versions
                  else (f"R{v}" if v in red_versions else f"B{v}")
                  for v in sorted(set(red_versions) | set(blue_versions))]
    ax.set_xticks(range(max(len(red_versions), len(blue_versions))))
    ax.set_xticklabels([f"Iter {v}" for v in sorted(set(red_versions) | set(blue_versions))],
                       fontsize=8, rotation=45)
    ax.set_ylabel("BT Strength Rating", fontsize=10)
    ax.set_title("Bradley-Terry Ratings (95% CI)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_generalization(axes, red_versions, blue_versions, asr_matrix, tnr_matrix):
    """Figure 4: Generalization Analysis (two subplots)."""
    ax_left, ax_right = axes

    # Left: Each red_i's ASR across all blue_j
    cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(red_versions)))
    for i, rv in enumerate(red_versions):
        vals = asr_matrix[i, :]
        valid = ~np.isnan(vals)
        if valid.any():
            ax_left.plot(np.array(blue_versions)[valid], vals[valid],
                        marker="o", markersize=4, color=cmap[i],
                        linewidth=1.5, label=f"R{rv}")
    ax_left.set_xlabel("Blue Version", fontsize=9)
    ax_left.set_ylabel("ASR %", fontsize=9)
    ax_left.set_title("Red Generalization", fontsize=10, fontweight="bold")
    ax_left.legend(fontsize=6, ncol=2, loc="best")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(-5, 105)

    # Right: Each blue_j's TNR across all red_i
    cmap = plt.cm.Blues(np.linspace(0.3, 0.9, len(blue_versions)))
    for j, bv in enumerate(blue_versions):
        vals = tnr_matrix[:, j]
        valid = ~np.isnan(vals)
        if valid.any():
            ax_right.plot(np.array(red_versions)[valid], vals[valid],
                         marker="s", markersize=4, color=cmap[j],
                         linewidth=1.5, label=f"B{bv}")
    ax_right.set_xlabel("Red Version", fontsize=9)
    ax_right.set_ylabel("TNR %", fontsize=9)
    ax_right.set_title("Blue Robustness", fontsize=10, fontweight="bold")
    ax_right.legend(fontsize=6, ncol=2, loc="best")
    ax_right.grid(True, alpha=0.3)
    ax_right.set_ylim(-5, 105)


def plot_pareto(ax, results, blue_versions, tnr_matrix):
    """Figure 5: TPR vs TNR Pareto Frontier."""
    benign = results.get("benign_only", {})

    points = []
    for j, bv in enumerate(blue_versions):
        key = f"blue_{bv}"
        if key in benign:
            tpr_val = benign[key]["tpr"]
        else:
            # Fall back to average TPR from pairings
            tpr_vals = []
            for pk, pdata in results["pairings"].items():
                if pdata["blue_iter"] == bv:
                    tpr_vals.append(pdata["metrics"]["tpr"])
            tpr_val = np.mean(tpr_vals) if tpr_vals else 0.0

        # Mean TNR across all red opponents
        tnr_vals = tnr_matrix[:, j]
        mean_tnr = np.nanmean(tnr_vals) if not np.all(np.isnan(tnr_vals)) else 0.0
        points.append((tpr_val, mean_tnr, bv))

    if not points:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    tpr_arr = np.array([p[0] for p in points])
    tnr_arr = np.array([p[1] for p in points])
    labels = [p[2] for p in points]

    ax.scatter(tpr_arr, tnr_arr, c="#3498db", s=60, zorder=5, edgecolors="black", linewidth=0.5)
    for tpr_val, tnr_val, label in points:
        ax.annotate(f"B{label}", (tpr_val, tnr_val), fontsize=7,
                   textcoords="offset points", xytext=(5, 5))

    # Compute and draw Pareto frontier
    # A point is Pareto-optimal if no other point dominates it on both axes
    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and tpr_arr[j] >= tpr_arr[i] and tnr_arr[j] >= tnr_arr[i]:
                if tpr_arr[j] > tpr_arr[i] or tnr_arr[j] > tnr_arr[i]:
                    pareto_mask[i] = False
                    break

    pareto_idx = np.where(pareto_mask)[0]
    if len(pareto_idx) > 1:
        sorted_pareto = pareto_idx[np.argsort(tpr_arr[pareto_idx])]
        ax.plot(tpr_arr[sorted_pareto], tnr_arr[sorted_pareto],
                color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7,
                label="Pareto frontier")

    ax.set_xlabel("TPR (Utility) %", fontsize=10)
    ax.set_ylabel("Mean TNR (Security) %", fontsize=10)
    ax.set_title("Security\u2013Utility Tradeoff", fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    if len(pareto_idx) > 1:
        ax.legend(fontsize=8)


def plot_nash(ax, red_versions, blue_versions, red_mixture, blue_mixture, game_value):
    """Figure 6: Nash Equilibrium Support."""
    if red_mixture is None:
        ax.text(0.5, 0.5, "Nash equilibrium\nrequires scipy",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    width = 0.35
    x = np.arange(max(len(red_versions), len(blue_versions)))

    red_bars = np.zeros(len(x))
    blue_bars = np.zeros(len(x))
    red_bars[:len(red_mixture)] = red_mixture
    blue_bars[:len(blue_mixture)] = blue_mixture

    ax.bar(x - width / 2, red_bars, width, color="#e74c3c", alpha=0.8, label="Red mixture")
    ax.bar(x + width / 2, blue_bars, width, color="#3498db", alpha=0.8, label="Blue mixture")

    all_versions = sorted(set(red_versions) | set(blue_versions))
    ax.set_xticks(range(len(all_versions)))
    ax.set_xticklabels([f"Iter {v}" for v in all_versions], fontsize=8, rotation=45)
    ax.set_ylabel("Mixture Weight", fontsize=10)
    ax.set_title(f"Nash Equilibrium (Game Value: {game_value:.1%} ASR)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(1.0, max(red_bars.max(), blue_bars.max()) * 1.3 + 0.05))


# ──────────────────────────── LaTeX Tables ──────────────────────────────────


def generate_latex_tables(
    results, red_versions, blue_versions, asr_matrix, bt_result, transitivity
) -> str:
    """Generate LaTeX table source for the paper."""
    lines = []

    # Table 1: Win Rate Matrix
    lines.append("% Table 1: ASR Win Rate Matrix")
    cols = "c" + "c" * len(blue_versions)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = " & ".join([""] + [f"B{v}" for v in blue_versions]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for i, rv in enumerate(red_versions):
        cells = [f"R{rv}"]
        for j in range(len(blue_versions)):
            val = asr_matrix[i, j]
            if np.isnan(val):
                cells.append("--")
            else:
                # Bold diagonal
                if rv == blue_versions[j] and rv > 0:
                    cells.append(f"\\textbf{{{val:.1f}}}")
                else:
                    cells.append(f"{val:.1f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    # Table 2: Bradley-Terry Ratings
    lines.append("% Table 2: Bradley-Terry Ratings")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Version & Rating & 95\\% CI & Rank \\\\")
    lines.append("\\midrule")

    # Combine and sort by rating
    all_entries = []
    for i, rv in enumerate(red_versions):
        r = bt_result["red_ratings"][i]
        se = bt_result["red_se"][i]
        all_entries.append((f"Red {rv}", r, se, "red"))
    for j, bv in enumerate(blue_versions):
        r = bt_result["blue_ratings"][j]
        se = bt_result["blue_se"][j]
        all_entries.append((f"Blue {bv}", r, se, "blue"))

    all_entries.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, rating, se, team) in enumerate(all_entries, 1):
        ci_lo = max(0, rating - 1.96 * se)
        ci_hi = rating + 1.96 * se
        lines.append(f"{name} & {rating:.3f} & [{ci_lo:.3f}, {ci_hi:.3f}] & {rank} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    # Table 3: Summary Statistics
    lines.append("% Table 3: Summary Statistics")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")

    mean_asr = np.nanmean(asr_matrix)
    mean_tnr_per_blue = [np.nanmean(asr_matrix[:, j]) for j in range(len(blue_versions))]
    lines.append(f"Mean ASR (all pairings) & {mean_asr:.1f}\\% \\\\")
    lines.append(f"Transitivity Score & {transitivity:.3f} \\\\")
    lines.append(f"Number of pairings & {(~np.isnan(asr_matrix)).sum()} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


# ──────────────────────────── Text Summary ──────────────────────────────────


def generate_summary(
    results, red_versions, blue_versions, asr_matrix, tnr_matrix, bt_result,
    transitivity, red_mixture, blue_mixture, game_value
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Self-play directory: {results['metadata'].get('selfplay_dir', 'N/A')}")
    lines.append(f"Base model: {results['metadata'].get('base_model', 'N/A')}")
    lines.append(f"Red versions: {red_versions}")
    lines.append(f"Blue versions: {blue_versions}")
    lines.append(f"Pairings evaluated: {len(results['pairings'])}")
    lines.append("")

    lines.append("--- Mean ASR per Red Version (across all Blues) ---")
    for i, rv in enumerate(red_versions):
        mean = np.nanmean(asr_matrix[i, :])
        lines.append(f"  Red {rv}: {mean:.1f}%")

    lines.append("")
    lines.append("--- Mean TNR per Blue Version (across all Reds) ---")
    for j, bv in enumerate(blue_versions):
        mean = np.nanmean(tnr_matrix[:, j])
        lines.append(f"  Blue {bv}: {mean:.1f}%")

    lines.append("")
    lines.append(f"--- Transitivity Score: {transitivity:.3f} ---")
    if transitivity > 0.9:
        lines.append("  Interpretation: Highly transitive — monotonic arms race.")
    elif transitivity > 0.7:
        lines.append("  Interpretation: Mostly transitive with some cycling.")
    else:
        lines.append("  Interpretation: Low transitivity — significant cycling/RPS dynamics.")

    lines.append("")
    lines.append("--- Bradley-Terry Ratings ---")
    for i, rv in enumerate(red_versions):
        lines.append(f"  Red {rv}: {bt_result['red_ratings'][i]:.3f}")
    for j, bv in enumerate(blue_versions):
        lines.append(f"  Blue {bv}: {bt_result['blue_ratings'][j]:.3f}")

    if game_value is not None:
        lines.append("")
        lines.append(f"--- Nash Equilibrium (Game Value: {game_value:.1%} ASR) ---")
        lines.append("  Red mixture:")
        for i, rv in enumerate(red_versions):
            if red_mixture[i] > 0.01:
                lines.append(f"    Iter {rv}: {red_mixture[i]:.3f}")
        lines.append("  Blue mixture:")
        for j, bv in enumerate(blue_versions):
            if blue_mixture[j] > 0.01:
                lines.append(f"    Iter {bv}: {blue_mixture[j]:.3f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ──────────────────────────── Main ──────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python util/plot_cross_eval.py <cross_eval_dir>")
        sys.exit(1)

    cross_eval_dir = sys.argv[1]
    results = load_results(cross_eval_dir)

    red_versions, blue_versions, asr, tnr, cfr, dom, tpr, f1 = build_matrices(results)
    print(f"Loaded {len(red_versions)}x{len(blue_versions)} matrix "
          f"({len(results['pairings'])} pairings)")

    # Compute advanced metrics
    bt_result = fit_bradley_terry(asr, red_versions, blue_versions)
    red_mixture, blue_mixture, game_value = compute_nash_equilibrium(asr)
    transitivity = compute_transitivity(asr)

    fig_dir = os.path.join(cross_eval_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure 1: ASR Heatmap ──
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_heatmap_asr(ax1, red_versions, blue_versions, asr)
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_dir, "cross_eval_heatmap_asr.pdf"), dpi=150)
    fig1.savefig(os.path.join(fig_dir, "cross_eval_heatmap_asr.png"), dpi=150)
    plt.close(fig1)
    print("  Saved: heatmap_asr")

    # ── Figure 2: Dominance Heatmap ──
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_heatmap_dominance(ax2, red_versions, blue_versions, dom)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "cross_eval_heatmap_dominance.pdf"), dpi=150)
    fig2.savefig(os.path.join(fig_dir, "cross_eval_heatmap_dominance.png"), dpi=150)
    plt.close(fig2)
    print("  Saved: heatmap_dominance")

    # ── Figure 3: Bradley-Terry Ratings ──
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_bt_ratings(ax3, red_versions, blue_versions, bt_result)
    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_dir, "cross_eval_bt_ratings.pdf"), dpi=150)
    fig3.savefig(os.path.join(fig_dir, "cross_eval_bt_ratings.png"), dpi=150)
    plt.close(fig3)
    print("  Saved: bt_ratings")

    # ── Figure 4: Generalization Analysis ──
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    plot_generalization((ax4a, ax4b), red_versions, blue_versions, asr, tnr)
    fig4.tight_layout()
    fig4.savefig(os.path.join(fig_dir, "cross_eval_generalization.pdf"), dpi=150)
    fig4.savefig(os.path.join(fig_dir, "cross_eval_generalization.png"), dpi=150)
    plt.close(fig4)
    print("  Saved: generalization")

    # ── Figure 5: Pareto Frontier ──
    fig5, ax5 = plt.subplots(figsize=(7, 6))
    plot_pareto(ax5, results, blue_versions, tnr)
    fig5.tight_layout()
    fig5.savefig(os.path.join(fig_dir, "cross_eval_pareto.pdf"), dpi=150)
    fig5.savefig(os.path.join(fig_dir, "cross_eval_pareto.png"), dpi=150)
    plt.close(fig5)
    print("  Saved: pareto")

    # ── Figure 6: Nash Equilibrium ──
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    plot_nash(ax6, red_versions, blue_versions, red_mixture, blue_mixture, game_value)
    fig6.tight_layout()
    fig6.savefig(os.path.join(fig_dir, "cross_eval_nash.pdf"), dpi=150)
    fig6.savefig(os.path.join(fig_dir, "cross_eval_nash.png"), dpi=150)
    plt.close(fig6)
    print("  Saved: nash")

    # ── LaTeX Tables ──
    latex = generate_latex_tables(results, red_versions, blue_versions, asr, bt_result, transitivity)
    latex_path = os.path.join(fig_dir, "cross_eval_tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {latex_path}")

    # ── Text Summary ──
    summary = generate_summary(
        results, red_versions, blue_versions, asr, tnr,
        bt_result, transitivity, red_mixture, blue_mixture, game_value
    )
    summary_path = os.path.join(fig_dir, "cross_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")
    print("")
    print(summary)


if __name__ == "__main__":
    main()
