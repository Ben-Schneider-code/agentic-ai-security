#!/usr/bin/env python3
"""
Analyze and visualize cross-evaluation results.

Reads cross_eval_results.json produced by cross_evaluate.py and generates:
  1. PVR_conv Heatmap (attack success matrix)
  2. Dominance Heatmap
  3. Bradley-Terry Strength Ratings
  4. Generalization Analysis
  5. Security-Utility Frontier: (1 - PUD) vs (1 - PVR_turn)
  6. Nash Equilibrium Support
  + LaTeX tables + text summary

Usage:
    python util/plot_cross_eval.py <cross_eval_dir> [--output-dir DIR]

Example:
    python util/plot_cross_eval.py results-20260322-1641-m92p4/cross_eval/

Naming note (display vs. data): the local numpy variables ``asr``, ``tnr``, and
``tpr`` mirror the JSON schema emitted by ``util/cross_evaluate.py`` and are
therefore kept verbatim. In terms of problem_statement.tex:

    asr  ≡ PVR_conv        (conversation-level policy violation rate)
    tnr  ≡ 1 − PVR_turn    (turn-level attack refusal)
    tpr  ≡ 1 − PUD         (turn-level benign utility)
    cfr  ≡ policy violation on confirmed-denied turn (retained)
    f1   ≡ HM(1 − PVR_turn, 1 − PUD)

All user-facing strings (titles, axis labels, legends, colorbar, LaTeX)
use the PVR/PUD names; only the JSON/variable layer keeps the legacy names
for log compatibility.
"""

import argparse
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
    pvr_turn = np.full((n_red, n_blue), np.nan)

    for key, data in pairings.items():
        ri = red_idx[data["red_iter"]]
        bi = blue_idx[data["blue_iter"]]
        m = data["metrics"]
        asr[ri, bi] = m["asr"]
        tnr[ri, bi] = m["tnr"]
        cfr[ri, bi] = m["cfr"]
        dom[ri, bi] = m["dominance"]
        tpr[ri, bi] = m["tpr"]
        # pvr_turn is new; older aggregations may not include it.
        pvr_turn[ri, bi] = m.get("pvr_turn", np.nan)
        f1[ri, bi] = m["f1"]

    return red_versions, blue_versions, asr, tnr, cfr, dom, tpr, f1, pvr_turn


# ──────────────────────────── Bradley-Terry Model ───────────────────────────


def fit_bradley_terry(asr_matrix: np.ndarray, red_versions: list, blue_versions: list) -> dict:
    """Fit Bradley-Terry model from the PVR_conv matrix.

    Treats PVR_conv/100 as P(red_i beats blue_j). Fits strength parameters via
    iterative MLE (no scipy needed — uses the classic iterative algorithm).

    Returns dict with 'red_ratings', 'blue_ratings' (lists of floats).
    """
    n_red = len(red_versions)
    n_blue = len(blue_versions)

    # Initialize strengths
    red_strength = np.ones(n_red)
    blue_strength = np.ones(n_blue)

    # Number of "games" per cell (assume 100 if not specified);
    # we use PVR_conv as the win fraction directly.
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
    Payoff = PVR_conv for Red, (100 - PVR_conv) for Blue.

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
    check if red_i > red_j > red_k (in terms of PVR_conv) implies red_i > red_k.
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


def plot_heatmap_pvr_conv(ax, red_versions, blue_versions, asr_matrix):
    """Figure 1: Policy Violation Rate (conversation-level) Heatmap."""
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
    ax.set_xticklabels([f"$\\mathcal{{A}}_{{{v}}}$" for v in blue_versions], fontsize=8)
    ax.set_yticks(range(len(red_versions)))
    ax.set_yticklabels([f"$\\mathcal{{R}}_{{{v}}}$" for v in red_versions], fontsize=8)
    ax.set_xlabel(r"Blue-team ($\mathcal{A}$) iteration", fontsize=10)
    ax.set_ylabel(r"Red-team ($\mathcal{R}$) iteration", fontsize=10)
    ax.set_title(
        r"Policy Violation Rate (conversation-level) $\mathrm{PVR}_{\mathrm{conv}}$ %",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label=r"$\mathrm{PVR}_{\mathrm{conv}}$ %", shrink=0.8)


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
    ax.set_xticklabels([f"$\\mathcal{{A}}_{{{v}}}$" for v in blue_versions], fontsize=8)
    ax.set_yticks(range(len(red_versions)))
    ax.set_yticklabels([f"$\\mathcal{{R}}_{{{v}}}$" for v in red_versions], fontsize=8)
    ax.set_xlabel(r"Blue-team ($\mathcal{A}$) iteration", fontsize=10)
    ax.set_ylabel(r"Red-team ($\mathcal{R}$) iteration", fontsize=10)
    ax.set_title(
        r"Dominance Score (+ = $\mathcal{A}$ dominant, $-$ = $\mathcal{R}$ dominant)",
        fontsize=11,
        fontweight="bold",
    )
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
           color="#e74c3c", alpha=0.8, label=r"$\mathcal{R}$ (Red team)", capsize=3)
    ax.bar(x_blue + width / 2, blue_ratings, width, yerr=[s * 1.96 for s in blue_se],
           color="#3498db", alpha=0.8, label=r"$\mathcal{A}$ (Blue team)", capsize=3)

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


def plot_generalization(axes, red_versions, blue_versions, asr_matrix, pvr_turn_matrix):
    """Figure 4: Generalization Analysis (two subplots).

    Left: each R_i's PVR_conv across all A_j — red transfer view.
    Right: each A_j's (1 - PVR_turn) across all R_i — blue robustness view.

    For legacy aggregations that predate the pvr_turn field, the right plot
    falls back to (1 - PVR_conv) so it still stays in PVR vocabulary.
    """
    ax_left, ax_right = axes

    # Left: each R_i's PVR_conv across all A_j
    cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(red_versions)))
    for i, rv in enumerate(red_versions):
        vals = asr_matrix[i, :]
        valid = ~np.isnan(vals)
        if valid.any():
            ax_left.plot(np.array(blue_versions)[valid], vals[valid],
                        marker="o", markersize=4, color=cmap[i],
                        linewidth=1.5, label=f"$\\mathcal{{R}}_{{{rv}}}$")
    ax_left.set_xlabel(r"Blue-team ($\mathcal{A}$) iteration", fontsize=9)
    ax_left.set_ylabel(r"$\mathrm{PVR}_{\mathrm{conv}}$ %", fontsize=9)
    ax_left.set_title(r"Red generalization ($\mathcal{R}_{i}$ vs all $\mathcal{A}_{j}$)",
                      fontsize=10, fontweight="bold")
    ax_left.legend(fontsize=6, ncol=2, loc="best")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(-5, 105)

    # Right: each A_j's (1 - PVR_turn) across all R_i.
    # Fall back to (1 - PVR_conv) if pvr_turn unavailable (legacy dirs).
    use_pvr_turn = pvr_turn_matrix is not None and not np.all(np.isnan(pvr_turn_matrix))
    if use_pvr_turn:
        robustness_matrix = 100 - pvr_turn_matrix
        right_ylabel = r"$(1 - \mathrm{PVR}_{\mathrm{turn}})$ %"
    else:
        robustness_matrix = 100 - asr_matrix
        right_ylabel = r"$(1 - \mathrm{PVR}_{\mathrm{conv}})$ % (legacy)"

    cmap = plt.cm.Blues(np.linspace(0.3, 0.9, len(blue_versions)))
    for j, bv in enumerate(blue_versions):
        vals = robustness_matrix[:, j]
        valid = ~np.isnan(vals)
        if valid.any():
            ax_right.plot(np.array(red_versions)[valid], vals[valid],
                         marker="s", markersize=4, color=cmap[j],
                         linewidth=1.5, label=f"$\\mathcal{{A}}_{{{bv}}}$")
    ax_right.set_xlabel(r"Red-team ($\mathcal{R}$) iteration", fontsize=9)
    ax_right.set_ylabel(right_ylabel, fontsize=9)
    ax_right.set_title(r"Blue robustness ($\mathcal{A}_{j}$ vs all $\mathcal{R}_{i}$)",
                       fontsize=10, fontweight="bold")
    ax_right.legend(fontsize=6, ncol=2, loc="best")
    ax_right.grid(True, alpha=0.3)
    ax_right.set_ylim(-5, 105)


def plot_security_utility_frontier(ax, results, blue_versions, pvr_turn_matrix, tnr_matrix):
    """Figure 5: (1 - PUD) vs (1 - PVR_turn) frontier per methodology §sec:visualization.

    Falls back to (1-PUD) vs (1-PVR_conv) when pvr_turn is unavailable
    (legacy result dirs) so older aggregations still render; the axis label
    reflects which quantity is actually plotted.
    """
    benign = results.get("benign_only", {})

    # Prefer PVR_turn (new); fall back to (1 - PVR_conv) if older aggregation lacks it.
    use_pvr_turn = not np.all(np.isnan(pvr_turn_matrix))
    if use_pvr_turn:
        security_matrix = 100 - pvr_turn_matrix
        security_label = r"Mean $(1 - \mathrm{PVR}_{\mathrm{turn}})$ %"
    else:
        # tnr_matrix holds (1 - PVR_turn) when the field was present; when it
        # is not, fall back to the conversation-level complement.
        security_matrix = tnr_matrix
        security_label = r"Mean $(1 - \mathrm{PVR}_{\mathrm{conv}})$ % (legacy proxy)"

    points = []
    for j, bv in enumerate(blue_versions):
        key = f"blue_{bv}"
        if key in benign:
            tpr_val = benign[key]["tpr"]
        else:
            tpr_vals = []
            for pk, pdata in results["pairings"].items():
                if pdata["blue_iter"] == bv:
                    tpr_vals.append(pdata["metrics"]["tpr"])
            tpr_val = np.mean(tpr_vals) if tpr_vals else 0.0

        sec_vals = security_matrix[:, j]
        mean_sec = np.nanmean(sec_vals) if not np.all(np.isnan(sec_vals)) else 0.0
        points.append((tpr_val, mean_sec, bv))

    if not points:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    tpr_arr = np.array([p[0] for p in points])
    sec_arr = np.array([p[1] for p in points])

    ax.scatter(tpr_arr, sec_arr, c="#3498db", s=60, zorder=5, edgecolors="black", linewidth=0.5)
    for tpr_val, sec_val, label in points:
        ax.annotate(f"$\\mathcal{{A}}_{{{label}}}$", (tpr_val, sec_val), fontsize=7,
                   textcoords="offset points", xytext=(5, 5))

    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and tpr_arr[j] >= tpr_arr[i] and sec_arr[j] >= sec_arr[i]:
                if tpr_arr[j] > tpr_arr[i] or sec_arr[j] > sec_arr[i]:
                    pareto_mask[i] = False
                    break

    pareto_idx = np.where(pareto_mask)[0]
    if len(pareto_idx) > 1:
        sorted_pareto = pareto_idx[np.argsort(tpr_arr[pareto_idx])]
        ax.plot(tpr_arr[sorted_pareto], sec_arr[sorted_pareto],
                color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7,
                label="Pareto frontier")

    ax.set_xlabel(r"$(1 - \mathrm{PUD})$ % (Utility)", fontsize=10)
    ax.set_ylabel(security_label + " (Security)", fontsize=10)
    ax.set_title("Security\u2013Utility Frontier", fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    if len(pareto_idx) > 1:
        ax.legend(fontsize=8)


def load_significance_matrix(cross_eval_dir: str) -> dict | None:
    """Load significance_matrix.json if produced by mcnemar_cross_eval.py.

    Returns None if the file is absent, so callers can render without
    significance annotations when the post-hoc test has not been run.
    """
    import json
    candidates = [
        os.path.join(cross_eval_dir, "significance_matrix.json"),
        # Also allow side-car dir (useful when cross_eval dir is read-only).
        os.path.join(os.path.dirname(cross_eval_dir.rstrip("/")),
                     "smoke_tests", "significance_matrix.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def plot_significance_heatmap(ax, red_versions, blue_versions, sig_matrix, axis: str = "blue"):
    """Render pairwise p-values (two-proportion test) as a heatmap.

    Cells with ``p > 0.05`` are rendered in a distinct style (gray,
    italic), as promised in the paper's §sec:episode-protocol.

    axis=='blue' means for each fixed red_iter we plot p(B_i vs B_j).
    axis=='red' means for each fixed blue_iter we plot p(R_i vs R_j).
    """
    # Restrict to the adapted-agent axis: use the LAST fixed index
    # (most-adapted red / blue) because that's the most decision-relevant
    # slice of the comparison grid. Users can edit this if they want a
    # different slice.
    if axis == "blue":
        group_key = "blue_vs_blue"
        versions = blue_versions
        label_prefix = "B"
        # Pick the last (most adapted) red iteration for which we have data
        fixed = max(
            (int(k) for k in sig_matrix.get(group_key, {})),
            default=None,
        )
    else:
        group_key = "red_vs_red"
        versions = red_versions
        label_prefix = "R"
        fixed = max(
            (int(k) for k in sig_matrix.get(group_key, {})),
            default=None,
        )
    if fixed is None:
        ax.text(0.5, 0.5, "No significance data", ha="center", va="center",
                transform=ax.transAxes)
        return

    rows = sig_matrix[group_key][str(fixed)]
    n = len(versions)
    p_matrix = np.full((n, n), np.nan)
    for row in rows:
        # Pairing keys look like "red_<r>_blue_<b>"; extract the variable side.
        a_key, b_key = row["a"], row["b"]
        def _idx(key: str) -> int:
            import re
            if axis == "blue":
                m = re.search(r"blue_(\d+)", key)
            else:
                m = re.search(r"red_(\d+)", key)
            return int(m.group(1))
        i = versions.index(_idx(a_key))
        j = versions.index(_idx(b_key))
        p = row["p_two_prop"]
        if p is not None:
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    # Render using a perceptually flipped colormap so small p (significant) is dark.
    im = ax.imshow(np.log10(np.clip(p_matrix, 1e-6, 1.0)),
                   cmap="viridis_r", vmin=-3, vmax=0, aspect="auto")
    for i in range(n):
        for j in range(n):
            v = p_matrix[i, j]
            if np.isnan(v):
                continue
            is_sig = v < 0.05
            label = f"{v:.3f}" if v >= 0.001 else "<.001"
            style = {"fontsize": 7,
                     "color": "white" if is_sig else "#ccc",
                     "fontweight": "bold" if is_sig else "normal",
                     "fontstyle": "normal" if is_sig else "italic"}
            ax.text(j, i, label, ha="center", va="center", **style)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{label_prefix}{v}" for v in versions], fontsize=8)
    ax.set_yticklabels([f"{label_prefix}{v}" for v in versions], fontsize=8)
    ax.set_title(
        f"Pairwise significance ({label_prefix} vs {label_prefix}, "
        f"{'red' if axis=='blue' else 'blue'} iter fixed={fixed})\n"
        f"italic/gray = $p>0.05$",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="$\\log_{10} p$", shrink=0.8)


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

    ax.bar(x - width / 2, red_bars, width, color="#e74c3c", alpha=0.8,
           label=r"$\mathcal{R}$ mixture")
    ax.bar(x + width / 2, blue_bars, width, color="#3498db", alpha=0.8,
           label=r"$\mathcal{A}$ mixture")

    all_versions = sorted(set(red_versions) | set(blue_versions))
    ax.set_xticks(range(len(all_versions)))
    ax.set_xticklabels([f"Iter {v}" for v in all_versions], fontsize=8, rotation=45)
    ax.set_ylabel("Mixture Weight", fontsize=10)
    ax.set_title(
        r"Nash Equilibrium (Game Value: "
        f"{game_value:.1%} " + r"$\mathrm{PVR}_{\mathrm{conv}}$)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(1.0, max(red_bars.max(), blue_bars.max()) * 1.3 + 0.05))


# ──────────────────────────── LaTeX Tables ──────────────────────────────────


def generate_latex_tables(
    results, red_versions, blue_versions, asr_matrix, bt_result, transitivity
) -> str:
    """Generate LaTeX table source for the paper."""
    lines = []

    # Table 1: PVR_conv matrix
    lines.append("% Table 1: PVR_conv (conversation-level policy violation rate) matrix")
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
    lines.append(f"Mean $\\mathrm{{PVR}}_{{\\mathrm{{conv}}}}$ (all pairings) & {mean_asr:.1f}\\% \\\\")
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

    lines.append("--- Mean PVR_conv per R-version (across all A) ---")
    for i, rv in enumerate(red_versions):
        mean = np.nanmean(asr_matrix[i, :])
        lines.append(f"  R{rv}: {mean:.1f}%")

    lines.append("")
    lines.append("--- Mean (1 - PVR_turn) per A-version (across all R) ---")
    for j, bv in enumerate(blue_versions):
        mean = np.nanmean(tnr_matrix[:, j])
        lines.append(f"  A{bv}: {mean:.1f}%")

    lines.append("")
    lines.append(f"--- Transitivity Score: {transitivity:.3f} ---")
    if transitivity > 0.9:
        lines.append("  Interpretation: Highly transitive — monotonic arms race.")
    elif transitivity > 0.7:
        lines.append("  Interpretation: Mostly transitive with some cycling.")
    else:
        lines.append("  Interpretation: Low transitivity — significant cycling/RPS dynamics.")

    if bt_result is not None:
        lines.append("")
        lines.append("--- Bradley-Terry Ratings ---")
        for i, rv in enumerate(red_versions):
            lines.append(f"  R{rv}: {bt_result['red_ratings'][i]:.3f}")
        for j, bv in enumerate(blue_versions):
            lines.append(f"  A{bv}: {bt_result['blue_ratings'][j]:.3f}")
    else:
        lines.append("")
        lines.append("--- Bradley-Terry Ratings: skipped (partial pairing coverage) ---")

    if game_value is not None:
        lines.append("")
        lines.append(f"--- Nash Equilibrium (Game Value: {game_value:.1%} PVR_conv) ---")
        lines.append("  R mixture:")
        for i, rv in enumerate(red_versions):
            if red_mixture[i] > 0.01:
                lines.append(f"    Iter {rv}: {red_mixture[i]:.3f}")
        lines.append("  A mixture:")
        for j, bv in enumerate(blue_versions):
            if blue_mixture[j] > 0.01:
                lines.append(f"    Iter {bv}: {blue_mixture[j]:.3f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ──────────────────────────── Main ──────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-evaluation results (PVR/PUD metric vocabulary)."
    )
    parser.add_argument(
        "cross_eval_dir",
        help="Directory containing cross_eval_results.json (produced by cross_evaluate.py).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write figures/tables into. "
             "Defaults to <cross_eval_dir>/figures/.",
    )
    args = parser.parse_args()

    cross_eval_dir = args.cross_eval_dir
    results = load_results(cross_eval_dir)

    red_versions, blue_versions, asr, tnr, cfr, dom, tpr, f1, pvr_turn = build_matrices(results)
    print(f"Loaded {len(red_versions)}x{len(blue_versions)} matrix "
          f"({len(results['pairings'])} pairings)")

    # Detect partial coverage (e.g. quick mode with --pairing-subset != full).
    # BT / Nash / Pareto assume the full (K+1)^2 matrix; on a sparse matrix
    # they still run but their outputs are misleading, so we skip them.
    n_cells = len(red_versions) * len(blue_versions)
    n_present = int((~np.isnan(asr)).sum())
    full_coverage = n_present == n_cells
    if not full_coverage:
        print(f"  [partial coverage] {n_present}/{n_cells} cells filled — "
              "skipping Bradley-Terry, Nash, and Pareto plots.")

    # Compute advanced metrics (only meaningful with full coverage)
    if full_coverage:
        bt_result = fit_bradley_terry(asr, red_versions, blue_versions)
        red_mixture, blue_mixture, game_value = compute_nash_equilibrium(asr)
    else:
        bt_result = None
        red_mixture, blue_mixture, game_value = None, None, None
    transitivity = compute_transitivity(asr)

    fig_dir = args.output_dir if args.output_dir else os.path.join(cross_eval_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure 1: PVR_conv Heatmap ──
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_heatmap_pvr_conv(ax1, red_versions, blue_versions, asr)
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_dir, "heatmap_pvr_conv.pdf"), dpi=150)
    fig1.savefig(os.path.join(fig_dir, "heatmap_pvr_conv.png"), dpi=150)
    plt.close(fig1)
    print("  Saved: heatmap_pvr_conv")

    # ── Figure 2: Dominance Heatmap ──
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_heatmap_dominance(ax2, red_versions, blue_versions, dom)
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "heatmap_dominance.pdf"), dpi=150)
    fig2.savefig(os.path.join(fig_dir, "heatmap_dominance.png"), dpi=150)
    plt.close(fig2)
    print("  Saved: heatmap_dominance")

    # ── Figure 3: Bradley-Terry Ratings ──
    if bt_result is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        plot_bt_ratings(ax3, red_versions, blue_versions, bt_result)
        fig3.tight_layout()
        fig3.savefig(os.path.join(fig_dir, "bt_ratings.pdf"), dpi=150)
        fig3.savefig(os.path.join(fig_dir, "bt_ratings.png"), dpi=150)
        plt.close(fig3)
        print("  Saved: bt_ratings")

    # ── Figure 4: Generalization Analysis ──
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    plot_generalization((ax4a, ax4b), red_versions, blue_versions, asr, pvr_turn)
    fig4.tight_layout()
    fig4.savefig(os.path.join(fig_dir, "generalization.pdf"), dpi=150)
    fig4.savefig(os.path.join(fig_dir, "generalization.png"), dpi=150)
    plt.close(fig4)
    print("  Saved: generalization")

    # ── Figure 5: Security-Utility Frontier ──
    # Pass pvr_turn if aggregation produced it; otherwise fall back to tnr so
    # older result dirs still render a (proxy) frontier.
    if full_coverage:
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        plot_security_utility_frontier(ax5, results, blue_versions, pvr_turn, tnr)
        fig5.tight_layout()
        fig5.savefig(os.path.join(fig_dir, "security_utility_frontier.pdf"), dpi=150)
        fig5.savefig(os.path.join(fig_dir, "security_utility_frontier.png"), dpi=150)
        plt.close(fig5)
        print("  Saved: security_utility_frontier")

    # ── Figure 6: Nash Equilibrium ──
    if red_mixture is not None:
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        plot_nash(ax6, red_versions, blue_versions, red_mixture, blue_mixture, game_value)
        fig6.tight_layout()
        fig6.savefig(os.path.join(fig_dir, "nash.pdf"), dpi=150)
        fig6.savefig(os.path.join(fig_dir, "nash.png"), dpi=150)
        plt.close(fig6)
        print("  Saved: nash")

    # ── Figure 7: Pairwise significance (if produced by mcnemar_cross_eval.py) ──
    sig_matrix = load_significance_matrix(cross_eval_dir)
    if sig_matrix is not None:
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))
        plot_significance_heatmap(ax7a, red_versions, blue_versions, sig_matrix, axis="blue")
        plot_significance_heatmap(ax7b, red_versions, blue_versions, sig_matrix, axis="red")
        fig7.tight_layout()
        try:
            fig7.savefig(os.path.join(fig_dir, "significance.pdf"), dpi=150)
            fig7.savefig(os.path.join(fig_dir, "significance.png"), dpi=150)
            print("  Saved: significance")
        except PermissionError:
            print("  [warn] figures/ is read-only; skipping significance figure")
        plt.close(fig7)
    else:
        print("  (no significance_matrix.json found — run util/mcnemar_cross_eval.py first)")

    # ── LaTeX Tables ──
    if bt_result is not None:
        latex = generate_latex_tables(results, red_versions, blue_versions, asr, bt_result, transitivity)
        latex_path = os.path.join(fig_dir, "tables.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"  Saved: {latex_path}")

    # ── Text Summary ──
    summary = generate_summary(
        results, red_versions, blue_versions, asr, tnr,
        bt_result, transitivity, red_mixture, blue_mixture, game_value
    )
    summary_path = os.path.join(fig_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")
    print("")
    print(summary)


if __name__ == "__main__":
    main()
