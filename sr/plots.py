"""Publication-Quality Matplotlib Visualizers for SR-Lab (Neuroscience & Physics Views)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import Coord, SREnvironment
from .core import (
    compute_eigenvectors,
    compute_sr_analytic,
    compute_value,
    discover_subgoals,
    rollout_greedy_path,
    spatial_autocorrelogram_2d,
)
from .metrics import place_field_metrics, track_skew_metrics


def _render_grid_ax(
    ax: plt.Axes,
    img: np.ndarray,
    title: str,
    cmap: str = "viridis",
    wall_color: str = "#262a33",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
):
    masked = np.ma.masked_invalid(img)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(wall_color)
    im = ax.imshow(masked, cmap=cm, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_place_field_comparison(
    env: SREnvironment,
    M: np.ndarray,
    target_cell: Coord,
    domain: str = "neuro",
) -> plt.Figure:
    """3-panel figure: Euclidean Gaussian vs Geodesic Gaussian vs Exact SR Place Field / Green's Function."""
    info = place_field_metrics(env, M, target_cell)
    tr, tc = info["target_cell"]

    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0))
    label_term = "Place Field M(:, s')" if domain == "neuro" else "Lattice Green's Function G(:, s')"

    _render_grid_ax(
        axs[0],
        env.to_grid(info["euclid_field"]),
        f"1. Euclidean Gaussian ('GPS')\nIgnores barriers (r={info['corr_with_euclidean_gaussian']})",
        cmap="viridis",
    )
    _render_grid_ax(
        axs[1],
        env.to_grid(info["geo_field"]),
        f"2. Geodesic Shortest-Path\nRespects walls, ignores drift (r={info['corr_with_geodesic_gaussian']})",
        cmap="viridis",
    )
    _render_grid_ax(
        axs[2],
        env.to_grid(info["place_field"]),
        f"3. Exact SR {label_term}\nPeak={info['peak_value']}, Cross-wall ratio={info['cross_wall_leakage_ratio']}",
        cmap="viridis",
    )

    for ax in axs:
        ax.plot(tc, tr, "r+", markersize=13, markeredgewidth=2.5, label=f"Target ({tr},{tc})")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    sup = (
        f"Hippocampal Predictive Place Field at s'=({tr},{tc}) — Warps Around Obstacles (Stachenfeld Fig. 2a–c)"
        if domain == "neuro"
        else f"Statistical Physics Lattice Green's Function G(s, s') at s'=({tr},{tc}) — Screened Propagator"
    )
    fig.suptitle(sup, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_track_skew(
    length: int = 41,
    target_idx: int = 25,
    p_right_biased: float = 0.9,
    gamma: float = 0.90,
    domain: str = "neuro",
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Plot 1D track place-field / advection-diffusion backward skew (Mehta 2000 / Paper Fig. 3b)."""
    from .env import build_track_env

    env_sym = build_track_env(length=length, p_right=0.5)
    env_bias = build_track_env(length=length, p_right=p_right_biased)
    M_sym = compute_sr_analytic(env_sym.T, gamma=gamma)
    M_bias = compute_sr_analytic(env_bias.T, gamma=gamma)

    f_sym = M_sym[:, target_idx]
    f_bias = M_bias[:, target_idx]
    m_sym = track_skew_metrics(f_sym, target_idx)
    m_bias = track_skew_metrics(f_bias, target_idx)

    fig, axs = plt.subplots(1, 2, figsize=(11.8, 3.9))
    x = np.arange(length)

    axs[0].plot(x, f_sym / f_sym.max(), color="#d62728", lw=2.4, label="Symmetric (p_right = 0.50)")
    axs[0].plot(
        x,
        f_bias / f_bias.max(),
        color="#1f77b4",
        lw=2.6,
        label=f"Directed / Drift (p_right = {p_right_biased:.2f})",
    )
    axs[0].axvline(target_idx, color="black", ls="--", lw=1.2, label=f"Encoded State s* = {target_idx}")
    axs[0].axvline(m_bias["center_of_mass"], color="#1f77b4", ls=":", lw=1.5, label=f"Biased CoM = {m_bias['center_of_mass']}")
    axs[0].annotate(
        "Direction of travel / drift ->",
        xy=(min(length - 2, target_idx + 8), 0.18),
        xytext=(max(1, target_idx - 12), 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
        fontsize=9,
        fontweight="bold",
    )
    axs[0].set_xlabel("State Position along 1D Track / Channel")
    axs[0].set_ylabel("Normalized Activity M(s, s*)")
    t1 = (
        "Backward Expansion of Place Field (Mehta 2000 / Fig. 3b)"
        if domain == "neuro"
        else "Upstream Advection-Diffusion Skew of Propagator G(x, x*)"
    )
    axs[0].set_title(t1, fontsize=10, fontweight="bold")
    axs[0].legend(fontsize=8.2, loc="upper left")
    axs[0].grid(alpha=0.25)

    # Panel 2: Full SR Matrix M under directional bias showing asymmetric triangular wake
    im = axs[1].imshow(M_bias, cmap="magma", origin="lower", aspect="auto")
    axs[1].axvline(target_idx, color="cyan", ls="--", lw=1.2)
    axs[1].set_xlabel("Future Visited State s'")
    axs[1].set_ylabel("Current Starting State s")
    axs[1].set_title(
        f"Full Resolvent Matrix M (gamma={gamma})\nBackward/Forward Mass = {m_bias['mass_behind_target']} vs {m_bias['mass_ahead_of_target']}",
        fontsize=10,
        fontweight="bold",
    )
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, {"unbiased": m_sym, "biased": m_bias}


def plot_reward_revaluation(
    env: SREnvironment,
    M: np.ndarray,
    start_cell: Coord,
    goal_a: Coord,
    goal_b: Coord,
    domain: str = "neuro",
) -> Tuple[plt.Figure, Dict[str, object]]:
    """3-panel figure demonstrating zero-shot reward / sink revaluation V = M R (Paper Fig. 1)."""
    s_cell = env.nearest_free_cell(*start_cell)
    ga = env.nearest_free_cell(*goal_a)
    gb = env.nearest_free_cell(*goal_b)

    s_idx = env.cell_to_idx[s_cell]
    ga_idx = env.cell_to_idx[ga]
    gb_idx = env.cell_to_idx[gb]

    R_a = np.zeros(env.n_states, dtype=float)
    R_a[ga_idx] = 1.0
    V_a = compute_value(M, R_a)
    path_a = rollout_greedy_path(env, V_a, s_cell, ga)

    R_b = np.zeros(env.n_states, dtype=float)
    R_b[gb_idx] = 1.0
    V_b = compute_value(M, R_b)
    path_b = rollout_greedy_path(env, V_b, s_cell, gb)

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.1))

    _render_grid_ax(
        axs[0],
        env.to_grid(M[s_idx, :]),
        f"1. Predictive Map Row M(s, :)\nLearned ONCE from Start s={s_cell}",
        cmap="magma",
    )
    axs[0].plot(s_cell[1], s_cell[0], "c+", markersize=13, markeredgewidth=2.5, label=f"Start {s_cell}")
    axs[0].legend(loc="lower right", fontsize=8)

    for ax, V, goal, path, label in [
        (axs[1], V_a, ga, path_a, f"2. Goal A at {ga}\nV = M·R_A (0 relearning steps)"),
        (axs[2], V_b, gb, path_b, f"3. Goal Moved to B at {gb}\nV = M·R_B (Instant Re-planning)"),
    ]:
        _render_grid_ax(ax, env.to_grid(V), label, cmap="magma")
        pr = [r for r, c in path]
        pc = [c for r, c in path]
        ax.plot(pc, pr, color="#00ffcc", lw=2.3, marker="o", markersize=3.5, label=f"Path ({len(path)-1} steps)")
        ax.plot(s_cell[1], s_cell[0], "w+", markersize=12, markeredgewidth=2.2)
        ax.plot(goal[1], goal[0], "y*", markersize=15, markeredgecolor="black", label=f"Goal {goal}")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    sup = (
        "Decoupled Predictive Map (M) & Reward (R): Instant Zero-Shot Revaluation V = M·R (Paper Fig. 1)"
        if domain == "neuro"
        else "Green's Function Resolvent (M) & Target Sink (R): Instant Multi-Sink Capture Yield V = M·R"
    )
    fig.suptitle(sup, fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    metrics = {
        "start_cell": s_cell,
        "goal_a": ga,
        "goal_b": gb,
        "value_at_start_for_goal_a": round(float(V_a[s_idx]), 4),
        "value_at_start_for_goal_b": round(float(V_b[s_idx]), 4),
        "path_a_steps": len(path_a) - 1,
        "path_b_steps": len(path_b) - 1,
        "relearning_steps_required": 0,
    }
    return fig, metrics


def plot_eigenvectors_and_subgoals(
    env: SREnvironment,
    gamma: float = 0.95,
    mode_indices: Sequence[int] = (1, 3, 6, 11, 18),
    top_n_subgoals: int = 2,
    domain: str = "neuro",
) -> Tuple[plt.Figure, Dict[str, object]]:
    """6-panel figure: Fiedler Partition + Bottleneck Subgoals/Saddle Points + Grid Cell Harmonics."""
    spec = compute_eigenvectors(env, gamma=gamma, threshold_zero=True)
    sub_info = discover_subgoals(env, gamma=gamma, top_n=top_n_subgoals)
    U_grid = spec["U_grid"]
    eigvals = spec["eigvals_T"]

    fig, axs = plt.subplots(2, 3, figsize=(12.4, 6.2))
    flat_axs = list(axs.flat)

    # Panel 0: Signed Mode #1 (Fiedler bottleneck partition) + discovered subgoals/saddle points
    u1_img = env.to_grid(sub_info["u1"])
    _render_grid_ax(
        flat_axs[0],
        u1_img,
        f"Mode #1 Partition (λ={sub_info['eigenvalue_1']})\n★ = Bottleneck: {sub_info['subgoals']}",
        cmap="coolwarm",
    )
    for sr, sc in sub_info["subgoals"]:
        flat_axs[0].plot(sc, sr, "y*", markersize=16, markeredgecolor="black")

    # Remaining 5 panels: Thresholded SR Eigenvectors (Grid cells / Relaxation modes)
    for ax, k_idx in zip(flat_axs[1:], mode_indices):
        k_safe = min(int(k_idx), env.n_states - 1)
        grid_img = env.to_grid(U_grid[:, k_safe])
        lam_val = round(float(eigvals[k_safe]), 4)
        mode_name = f"Grid Cell Eigenvector #{k_safe}" if domain == "neuro" else f"Transfer Mode #{k_safe}"
        _render_grid_ax(ax, grid_img, f"{mode_name} (λ={lam_val})\nmax(0, U_{k_safe})", cmap="inferno")

    sup = (
        "Entorhinal Grid Cells as SR Eigenvectors & Automated Subgoal Discovery (Paper Figs 6–8, Supp. 13)"
        if domain == "neuro"
        else "Transfer Operator Eigenmodes & Automated Transition-State / Saddle-Point Discovery"
    )
    fig.suptitle(sup, fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    metrics = {
        "discovered_subgoals": sub_info["subgoals"],
        "partition_sizes": sub_info["partition_sizes"],
        "eigenvalue_1": sub_info["eigenvalue_1"],
        "mode_eigenvalues": {int(k): round(float(eigvals[min(int(k), env.n_states - 1)]), 4) for k in mode_indices},
    }
    return fig, metrics


def plot_geometry_grid_comparison(gamma: float = 0.95, mode_k: int = 10) -> Tuple[plt.Figure, Dict[str, object]]:
    """Compare SR eigenvector grid cells and 2D spatial autocorrelograms across Square, Circle, Trapezoid (Fig. 6)."""
    from .env import build_grid_env

    shapes = [("Square Arena", "rectangle"), ("Circular Arena", "circle"), ("Trapezoidal Arena", "trapezoid")]
    fig, axs = plt.subplots(2, 3, figsize=(12.2, 6.4))
    summary = {}

    for col_idx, (title, sh) in enumerate(shapes):
        env = build_grid_env(height=19, width=19, shape=sh)
        spec = compute_eigenvectors(env, gamma=gamma, threshold_zero=True)
        k_use = min(mode_k, env.n_states - 1)
        field_img = env.to_grid(spec["U_grid"][:, k_use])
        ac_img = spatial_autocorrelogram_2d(field_img)

        _render_grid_ax(axs[0, col_idx], field_img, f"{title}\nEigenvector #{k_use} ({env.n_states} states)", cmap="inferno")
        _render_grid_ax(axs[1, col_idx], ac_img, f"{title}\n2D Spatial Autocorrelogram", cmap="RdBu_r")

        # Left-half vs Right-half split-halves correlation (Krupic et al. 2015, Paper Fig. 6d)
        mid = env.width // 2
        left_half = np.nan_to_num(field_img[:, :mid], nan=0.0).ravel()
        right_half = np.nan_to_num(field_img[:, -mid:], nan=0.0).ravel()
        l0, r0 = left_half - left_half.mean(), right_half - right_half.mean()
        half_corr = float(np.dot(l0, r0) / ((np.linalg.norm(l0) + 1e-12) * (np.linalg.norm(r0) + 1e-12)))
        summary[sh] = {
            "n_states": env.n_states,
            "eigenvalue": round(float(spec["eigvals_T"][k_use]), 4),
            "split_halves_symmetry_corr": round(abs(half_corr), 4),
        }

    fig.suptitle(
        "Boundary Geometry Shapes SR Grid-Cell Symmetry & Split-Halves Correlation (Krupic 2015 / Paper Fig. 6)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, summary


def plot_energy_landscape_analysis(
    env: SREnvironment,
    gamma: float = 0.96,
    source_cell: Optional[Coord] = None,
    sink_a: Optional[Coord] = None,
    sink_b: Optional[Coord] = None,
) -> Tuple[plt.Figure, Dict[str, object]]:
    """4-panel Statistical Physics & Chemistry Potential Energy Surface (PES) SR analysis."""
    assert env.energy_grid is not None, "Requires an energy_2d environment"
    M = compute_sr_analytic(env.T, gamma=gamma)
    sub_info = discover_subgoals(env, gamma=gamma, top_n=2)

    h, w = env.height, env.width
    src = source_cell or (h // 2, w // 4)
    sa = sink_a or (h // 2, (3 * w) // 4)
    sb = sink_b or (2, w - 3)

    src_idx = env.cell_to_idx[env.nearest_free_cell(*src)]
    sa_idx = env.cell_to_idx[env.nearest_free_cell(*sa)]
    sb_idx = env.cell_to_idx[env.nearest_free_cell(*sb)]

    R_multi = np.zeros(env.n_states, dtype=float)
    R_multi[sa_idx] = 1.0
    R_multi[sb_idx] = 0.85
    V_yield = compute_value(M, R_multi)
    path_to_a = rollout_greedy_path(env, V_yield, src, sa)

    fig, axs = plt.subplots(2, 2, figsize=(11.8, 6.8))

    # Panel 1: Potential Energy Surface E(x, y) + discovered Saddle Point (Transition State)
    _render_grid_ax(
        axs[0, 0],
        env.energy_grid,
        f"1. Potential Energy Surface E(x, y) (kT={env.kT})\n★ = Spectral Saddle Point / Bottleneck {sub_info['subgoals'][0]}",
        cmap="terrain",
    )
    for sr, sc in sub_info["subgoals"]:
        axs[0, 0].plot(sc, sr, "r*", markersize=16, markeredgecolor="white", label=f"Saddle ({sr},{sc})")
    axs[0, 0].legend(loc="lower right", fontsize=8)

    # Panel 2: Thermal Green's Function M(s_0, :) starting in Left Metastable Basin
    _render_grid_ax(
        axs[0, 1],
        env.to_grid(M[src_idx, :]),
        f"2. Thermal Green's Function M(s0, :) from {src}\nDiscounted Metastable Basin Occupancy",
        cmap="magma",
    )
    axs[0, 1].plot(src[1], src[0], "c+", markersize=13, markeredgewidth=2.5)

    # Panel 3: Relaxation Eigenmode #1 (Kramers Basin Partition / Committor Separatrix)
    _render_grid_ax(
        axs[1, 0],
        env.to_grid(sub_info["u1"]),
        f"3. Transfer Operator Slowest Mode U_1 (λ={sub_info['eigenvalue_1']})\nDivides Reactant & Product Basins at Separatrix",
        cmap="coolwarm",
    )
    for sr, sc in sub_info["subgoals"]:
        axs[1, 0].plot(sc, sr, "y*", markersize=15, markeredgecolor="black")

    # Panel 4: Instant Multi-Sink Reactive Yield V = M R + Minimum-Energy Reaction Path
    _render_grid_ax(
        axs[1, 1],
        env.to_grid(V_yield),
        f"4. Multi-Sink Capture Yield V = M·R\nPath crosses Saddle {sub_info['subgoals'][0]}",
        cmap="viridis",
    )
    pr = [r for r, c in path_to_a]
    pc = [c for r, c in path_to_a]
    axs[1, 1].plot(pc, pr, color="#00ffcc", lw=2.2, marker="o", markersize=3)
    axs[1, 1].plot(sa[1], sa[0], "y*", markersize=15, markeredgecolor="black", label=f"Sink A {sa}")
    axs[1, 1].plot(sb[1], sb[0], "m*", markersize=13, markeredgecolor="black", label=f"Sink B {sb}")
    axs[1, 1].legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Statistical Physics & Chemistry: SR as the Boltzmann Transfer Operator Resolvent M = (I − γT)⁻¹",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()

    saddle = sub_info["subgoals"][0]
    saddle_energy = round(float(env.energy_grid[saddle[0], saddle[1]]), 3)
    metrics = {
        "landscape_type": env.metadata.get("landscape_type", "double_well"),
        "kT": env.kT,
        "drift_rc": env.drift,
        "discovered_transition_state_saddle": saddle,
        "saddle_point_energy": saddle_energy,
        "min_basin_energy": round(float(env.energy_grid.min()), 3),
        "max_ridge_energy": round(float(env.energy_grid.max()), 3),
        "slowest_relaxation_eigenvalue": sub_info["eigenvalue_1"],
        "reactive_yield_at_source": round(float(V_yield[src_idx]), 4),
        "reaction_path_steps": len(path_to_a) - 1,
    }
    return fig, metrics


def plot_tolman_detour_remapping(gamma: float = 0.95) -> Tuple[plt.Figure, Dict[str, object]]:
    """Plot local vs distal place-field remapping when a barrier is inserted (Alvernhe 2011 / Fig. 3c–h)."""
    from .env import build_grid_env
    from .metrics import barrier_remapping_metrics

    h, w = 11, 15
    env_open = build_grid_env(height=h, width=w, wall_preset="open")
    # Insert a central vertical barrier blocking rows 2..8 at column 7
    barrier_walls = [(r, 7) for r in range(2, 9)]
    env_bar = build_grid_env(height=h, width=w, walls=barrier_walls)

    M_open = compute_sr_analytic(env_open.T, gamma=gamma)
    M_bar = compute_sr_analytic(env_bar.T, gamma=gamma)

    near_cells = [(3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (4, 8), (5, 8), (6, 8)]
    far_cells = [(1, 1), (2, 1), (8, 1), (9, 1), (1, 13), (2, 13), (8, 13), (9, 13)]
    remap = barrier_remapping_metrics(env_open, M_open, env_bar, M_bar, near_cells, far_cells)

    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.9))
    probe_near = (5, 6)
    f_open = M_open[:, env_open.cell_to_idx[probe_near]]
    f_bar = M_bar[:, env_bar.cell_to_idx[probe_near]]

    _render_grid_ax(axs[0], env_open.to_grid(f_open), f"1. Before Barrier: Open Maze\nPlace cell at Near Zone {probe_near}", cmap="viridis")
    axs[0].plot(probe_near[1], probe_near[0], "r+", markersize=13, markeredgewidth=2.5)

    _render_grid_ax(axs[1], env_bar.to_grid(f_bar), f"2. After Barrier Inserted (Col 7)\nLocal Remapping Around Obstacle", cmap="viridis")
    axs[1].plot(probe_near[1], probe_near[0], "r+", markersize=13, markeredgewidth=2.5)

    # Panel 3: Bar chart comparing Near vs Far spatial similarity (r and Fisher z)
    categories = ["Near Barrier\n(n=8 cells)", "Far from Barrier\n(n=8 cells)"]
    means_r = [remap["near_mean_r"], remap["far_mean_r"]]
    bars = axs[2].bar(categories, means_r, color=["#e15759", "#4e79a7"], width=0.5, edgecolor="black")
    axs[2].set_ylim(0.0, 1.1)
    axs[2].set_ylabel("Spatial Correlation r (Open vs Barrier)")
    axs[2].set_title(
        f"3. Local vs Distal Field Stability (Fig. 3e,h)\nNear r={remap['near_mean_r']} vs Far r={remap['far_mean_r']}",
        fontsize=10,
        fontweight="bold",
    )
    for bar, val in zip(bars, means_r):
        axs[2].text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"r = {val:.3f}", ha="center", fontweight="bold")
    axs[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Tolman Detour Remapping: Place Fields Remap Locally Near New Barriers (Alvernhe 2011 / Paper Fig. 3c–h)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, remap


def plot_schapiro_community_graph(gamma: float = 0.90) -> Tuple[plt.Figure, Dict[str, object]]:
    """Plot the 15-node Schapiro community graph, its block-diagonal SR matrix, and within/between similarity (Fig. 4)."""
    from .env import build_schapiro_graph
    from .metrics import community_similarity_metrics

    env = build_schapiro_graph()
    M = compute_sr_analytic(env.T, gamma=gamma)
    comm_metrics = community_similarity_metrics(M, env.communities)
    sub_info = discover_subgoals(env, gamma=gamma, top_n=4)

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))

    # Panel 1: Graph Topology colored by community
    palette = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    coords = env.coords
    for i in range(env.n_states):
        for j in range(i + 1, env.n_states):
            if env.A[i, j] > 0:
                axs[0].plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#888888", lw=1.4, zorder=1)
    for i in range(env.n_states):
        c_id = env.communities[i]
        is_bridge = i in (0, 4, 5, 9, 10, 14)
        axs[0].scatter(
            coords[i, 0],
            coords[i, 1],
            s=190 if is_bridge else 140,
            color=palette[c_id],
            edgecolors="black",
            linewidths=2.0 if is_bridge else 1.0,
            zorder=3,
        )
        axs[0].text(coords[i, 0], coords[i, 1], str(i), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
    axs[0].set_title("1. Schapiro 15-Node Community Graph\n3 Clusters Joined by Bottleneck Nodes", fontsize=10, fontweight="bold")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Panel 2: Block-diagonal SR Matrix M (Paper Fig. 4e)
    im = axs[1].imshow(M, cmap="viridis", interpolation="nearest")
    axs[1].set_title(f"2. Learned SR Matrix M (gamma={gamma})\nBlock-Diagonal Community Structure (Fig. 4e)", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Visited State s'")
    axs[1].set_ylabel("Starting State s")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Panel 3: Within vs Between community SR similarity (Paper Fig. 4g)
    labels = ["Within Community", "Between Communities"]
    vals = [comm_metrics["within_community_mean_corr"], comm_metrics["between_community_mean_corr"]]
    bars = axs[2].bar(labels, vals, color=["#2ca02c", "#d62728"], width=0.5, edgecolor="black")
    axs[2].axhline(0, color="black", lw=0.8)
    axs[2].set_ylim(-0.6, 0.85)
    axs[2].set_ylabel("Mean Row Correlation r of M(s, :)")
    axs[2].set_title(
        f"3. Representational Similarity (Fig. 4g)\nWithin={vals[0]:.3f} vs Between={vals[1]:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    for bar, v in zip(bars, vals):
        y_pos = v + 0.05 if v >= 0 else v - 0.1
        axs[2].text(bar.get_x() + bar.get_width() / 2, y_pos, f"r = {v:.3f}", ha="center", fontweight="bold")
    axs[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Non-Spatial Relational Hippocampal Coding: SR Captures Graph Community Structure (Schapiro 2016 / Fig. 4)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, {**comm_metrics, "bottleneck_nodes": [c[1] for c in sub_info["subgoals"]]}


def plot_td_and_spectral_regularization(
    env: SREnvironment,
    gamma: float = 0.95,
    n_steps: int = 8000,
    eta: float = 0.1,
    k_keep: int = 18,
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Compare Online TD(0) SR learning with Spectral Regularization (Eq. 8 & Supp. Fig. 12)."""
    from .core import compute_sr_td, spectral_regularize_sr

    # Run a short noisy TD pass and a full TD pass
    td_early = compute_sr_td(env, gamma=gamma, eta=eta, n_steps=max(800, n_steps // 5), seed=42)
    reg_early = spectral_regularize_sr(env, gamma=gamma, k_keep=k_keep, M_noisy=td_early["M_hat"])
    td_full = compute_sr_td(env, gamma=gamma, eta=eta, n_steps=n_steps, seed=42)

    mid_idx = env.n_states // 2
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9))

    axs[0].plot(td_full["steps"], td_full["correlations"], marker="o", color="#1f77b4", lw=2.2, label="Online TD(0) M_hat")
    axs[0].axhline(
        reg_early["corr_recon"],
        color="#2ca02c",
        ls="--",
        lw=2.0,
        label=f"Early TD + Spectral Smooth (k={k_keep}): r={reg_early['corr_recon']}",
    )
    axs[0].set_xlabel("Online TD(0) Transition Steps")
    axs[0].set_ylabel("Correlation with Exact Analytic M")
    axs[0].set_title(f"1. TD(0) Convergence (Eq. 8)\nFinal r = {td_full['final_correlation']}", fontsize=10, fontweight="bold")
    axs[0].legend(fontsize=8, loc="lower right")
    axs[0].grid(alpha=0.3)

    _render_grid_ax(
        axs[1],
        env.to_grid(td_early["M_hat"][:, mid_idx]),
        f"2. Early Noisy TD Field ({max(800, n_steps//5)} steps)\nr = {reg_early['corr_noisy']}",
        cmap="viridis",
    )
    _render_grid_ax(
        axs[2],
        env.to_grid(reg_early["M_recon"][:, mid_idx]),
        f"3. Spectrally Regularized (Top-{k_keep} Grid Modes)\nr = {reg_early['corr_recon']} (Supp. Fig. 12)",
        cmap="viridis",
    )

    fig.suptitle(
        "Online TD(0) Learning & Entorhinal Spectral Regularization (Stachenfeld Eq. 8 & Supp. Fig. 12)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()

    metrics = {
        "td_steps": n_steps,
        "learning_rate_eta": eta,
        "early_td_correlation": reg_early["corr_noisy"],
        "spectrally_smoothed_early_correlation": reg_early["corr_recon"],
        "final_td_correlation": td_full["final_correlation"],
        "final_td_mae": td_full["final_mae"],
        "spectral_k_modes": k_keep,
    }
    return fig, metrics
