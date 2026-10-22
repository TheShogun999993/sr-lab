"""Flagship Preset Experiments (P1–P10) across Neuroscience RL & Physical Sciences."""
from __future__ import annotations

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from .env import (
    SREnvironment,
    build_energy_landscape_env,
    build_grid_env,
    build_schapiro_graph,
    build_track_env,
)
from .core import compute_sr_analytic
from .metrics import check_row_sums, place_field_metrics
from .plots import (
    plot_eigenvectors_and_subgoals,
    plot_energy_landscape_analysis,
    plot_geometry_grid_comparison,
    plot_place_field_comparison,
    plot_reward_revaluation,
    plot_schapiro_community_graph,
    plot_tolman_detour_remapping,
    plot_track_skew,
)


PRESETS: Dict[str, Dict[str, str]] = {
    "P1_wall_warp": {
        "title": "P1: Place Fields Warp Around Obstacles (Fig. 2c)",
        "domain": "neuro",
        "paper_ref": "Stachenfeld et al. (2017) Fig. 2a–c",
        "summary": "12x12 room with a vertical barrier. Compares Euclidean Gaussian ('GPS'), Geodesic, and SR place fields.",
    },
    "P2_track_skew": {
        "title": "P2: Directed Motion Skews Place Fields Backwards (Fig. 3b)",
        "domain": "neuro",
        "paper_ref": "Mehta et al. (2000) / Stachenfeld Fig. 3a–b",
        "summary": "41-state 1D track comparing 50/50 random walk vs 90% rightward habit. Predictive coding shifts firing upstream.",
    },
    "P3_reward_reval": {
        "title": "P3: Zero-Shot Reward Revaluation V = M·R (Fig. 1)",
        "domain": "neuro",
        "paper_ref": "Stachenfeld et al. (2017) Fig. 1 & Eq. 2",
        "summary": "Decouples state dynamics M from reward R. Moving the goal recomputes optimal paths in 0 learning steps.",
    },
    "P4_two_rooms_subgoal": {
        "title": "P4: Grid-Cell Eigenvectors & Doorway Subgoal Discovery (Supp. Fig. 13)",
        "domain": "neuro",
        "paper_ref": "Stachenfeld et al. (2017) Figs 6–8 & Supp. Fig. 13",
        "summary": "11x23 two-room maze connected by a single doorway. Spectral min-cut of U_1 pinpoints the doorway bottleneck.",
    },
    "P5_geometry_grids": {
        "title": "P5: Boundary Geometry Distorts Grid Symmetry (Fig. 6)",
        "domain": "neuro",
        "paper_ref": "Krupic et al. (2015) / Stachenfeld Fig. 6",
        "summary": "Compares SR eigenvector grid fields and 2D spatial autocorrelograms across Square, Circle, and Trapezoid arenas.",
    },
    "P6_tolman_detour": {
        "title": "P6: Local Place-Field Remapping in Tolman Detour Maze (Fig. 3c–h)",
        "domain": "neuro",
        "paper_ref": "Alvernhe et al. (2011) / Stachenfeld Fig. 3c–h",
        "summary": "Inserting a barrier causes strong place-field remapping near the obstacle while distal fields remain stable.",
    },
    "P7_schapiro_graph": {
        "title": "P7: Non-Spatial Relational Community Graph (Fig. 4)",
        "domain": "neuro",
        "paper_ref": "Schapiro et al. (2016) / Stachenfeld Fig. 4",
        "summary": "15-node graph with 3 communities. The SR matrix learns block-diagonal cluster structure and bridge bottlenecks.",
    },
    "P8_double_well_saddle": {
        "title": "P8 [Physics/Chem]: Double-Well PES & Automated Saddle-Point Discovery",
        "domain": "physics",
        "paper_ref": "Kramers Transition State Theory / Spectral Transfer Operators",
        "summary": "2D Potential Energy Surface E(x,y) with two metastable basins. Mode #1 of M=(I-γT)⁻¹ finds the exact mountain-pass transition state.",
    },
    "P9_advection_diffusion_drift": {
        "title": "P9 [Physics]: Driven Advection-Diffusion Green's Function Under Drift Field",
        "domain": "physics",
        "paper_ref": "Smoluchowski / Fokker-Planck Lattice Resolvent",
        "summary": "2D thermal lattice with a repulsive obstacle and rightward electric drift field F. Shows upstream wake in G=(I-γT)⁻¹.",
    },
    "P10_crystal_vacancy_sinks": {
        "title": "P10 [Materials]: Crystal Grain-Boundary Diffusion & Multi-Sink Capture Yield",
        "domain": "physics",
        "paper_ref": "Lattice Green's Function for Point-Defect Capture",
        "summary": "Periodic crystal lattice with a grain-boundary barrier and two low-energy channels. Evaluates multi-sink defect capture V=M·R instantly.",
    },
}


def list_presets() -> List[Dict[str, str]]:
    return [{"id": k, **v} for k, v in PRESETS.items()]


def run_preset(preset_id: str) -> Tuple[SREnvironment, plt.Figure, Dict[str, object]]:
    """Execute one of the 10 flagship presets and return (env, figure, grounded_metrics_dict)."""
    pid = preset_id.strip()
    if pid not in PRESETS:
        raise ValueError(f"Unknown preset_id '{pid}'. Valid presets: {list(PRESETS.keys())}")

    meta = PRESETS[pid]

    if pid == "P1_wall_warp":
        env = build_grid_env(height=12, width=12, wall_preset="middle_wall_bottom_gap")
        gamma = 0.95
        M = compute_sr_analytic(env.T, gamma=gamma)
        target = (3, 5)
        fig = plot_place_field_comparison(env, M, target, domain="neuro")
        pf = place_field_metrics(env, M, target)
        row_info = check_row_sums(M, gamma)
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": gamma,
            "n_free_states": env.n_states,
            "n_walls": len(env.walls),
            "target_cell": pf["target_cell"],
            "peak_sr_value": pf["peak_value"],
            "corr_with_euclidean_gaussian": pf["corr_with_euclidean_gaussian"],
            "corr_with_geodesic_gaussian": pf["corr_with_geodesic_gaussian"],
            "near_wall_same_side_mean": pf["near_wall_same_side_mean"],
            "near_wall_opposite_side_mean": pf["near_wall_opposite_side_mean"],
            "cross_wall_leakage_ratio": pf["cross_wall_leakage_ratio"],
            **row_info,
        }
        return env, fig, metrics

    if pid == "P2_track_skew":
        gamma = 0.90
        env = build_track_env(length=41, p_right=0.90)
        fig, skew_data = plot_track_skew(length=41, target_idx=25, p_right_biased=0.90, gamma=gamma, domain="neuro")
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": gamma,
            "track_length": 41,
            "target_state": 25,
            "unbiased_p_right": 0.50,
            "biased_p_right": 0.90,
            "unbiased_mass_behind": skew_data["unbiased"]["mass_behind_target"],
            "unbiased_mass_ahead": skew_data["unbiased"]["mass_ahead_of_target"],
            "biased_mass_behind": skew_data["biased"]["mass_behind_target"],
            "biased_mass_ahead": skew_data["biased"]["mass_ahead_of_target"],
            "biased_backward_to_forward_ratio": skew_data["biased"]["backward_to_forward_ratio"],
            "biased_center_of_mass": skew_data["biased"]["center_of_mass"],
            "biased_com_shift": skew_data["biased"]["com_shift_from_target"],
        }
        return env, fig, metrics

    if pid == "P3_reward_reval":
        env = build_grid_env(height=12, width=12, wall_preset="middle_wall_bottom_gap")
        gamma = 0.95
        M = compute_sr_analytic(env.T, gamma=gamma)
        fig, rev_metrics = plot_reward_revaluation(
            env, M, start_cell=(10, 2), goal_a=(1, 10), goal_b=(1, 1), domain="neuro"
        )
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": gamma,
            "n_free_states": env.n_states,
            **rev_metrics,
            **check_row_sums(M, gamma),
        }
        return env, fig, metrics

    if pid == "P4_two_rooms_subgoal":
        env = build_grid_env(height=11, width=23, wall_preset="middle_wall_center_door")
        gamma = 0.95
        fig, eig_metrics = plot_eigenvectors_and_subgoals(
            env, gamma=gamma, mode_indices=(4, 7, 12, 18, 27), top_n_subgoals=2, domain="neuro"
        )
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": gamma,
            "grid_dimensions": (11, 23),
            "n_free_states": env.n_states,
            "doorway_coordinate": (5, 11),
            **eig_metrics,
        }
        return env, fig, metrics

    if pid == "P5_geometry_grids":
        env = build_grid_env(height=19, width=19, shape="trapezoid")
        fig, geo_metrics = plot_geometry_grid_comparison(gamma=0.95, mode_k=10)
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": 0.95,
            "eigenvector_index": 10,
            "arenas": geo_metrics,
        }
        return env, fig, metrics

    if pid == "P6_tolman_detour":
        env = build_grid_env(height=11, width=15, walls=[(r, 7) for r in range(2, 9)])
        fig, remap_metrics = plot_tolman_detour_remapping(gamma=0.95)
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": 0.95,
            **remap_metrics,
        }
        return env, fig, metrics

    if pid == "P7_schapiro_graph":
        env = build_schapiro_graph()
        fig, graph_metrics = plot_schapiro_community_graph(gamma=0.90)
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": 0.90,
            "n_nodes": 15,
            "n_communities": 3,
            **graph_metrics,
        }
        return env, fig, metrics

    if pid == "P8_double_well_saddle":
        env = build_energy_landscape_env(
            height=15, width=21, landscape_type="double_well", kT=0.55, barrier_height=3.6, drift_rc=(0.0, 0.0)
        )
        fig, pes_metrics = plot_energy_landscape_analysis(
            env, gamma=0.96, source_cell=(7, 4), sink_a=(7, 16), sink_b=(3, 17)
        )
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": 0.96,
            "true_saddle_coordinate": (7, 10),
            **pes_metrics,
        }
        return env, fig, metrics

    if pid == "P9_advection_diffusion_drift":
        env = build_energy_landscape_env(
            height=15,
            width=21,
            landscape_type="advection_obstacle",
            kT=0.65,
            barrier_height=4.0,
            drift_rc=(0.0, 0.45),  # Rightward external drift field F_c = +0.45
        )
        gamma = 0.95
        M = compute_sr_analytic(env.T, gamma=gamma)
        target = (7, 14)
        fig = plot_place_field_comparison(env, M, target_cell=target, domain="physics")
        pf = place_field_metrics(env, M, target)
        metrics = {
            "preset_id": pid,
            "paper_ref": meta["paper_ref"],
            "gamma": gamma,
            "kT": env.kT,
            "drift_field_rc": env.drift,
            "target_state": target,
            "center_of_mass_rc": pf["center_of_mass_rc"],
            "upstream_com_shift_cols": round(pf["center_of_mass_rc"][1] - target[1], 2),
            "peak_greens_function": pf["peak_value"],
            "corr_with_isotropic_gaussian": pf["corr_with_euclidean_gaussian"],
        }
        return env, fig, metrics

    # P10_crystal_vacancy_sinks
    env = build_energy_landscape_env(
        height=15,
        width=21,
        landscape_type="crystal_grain_boundary",
        kT=0.55,
        barrier_height=3.8,
        drift_rc=(0.0, 0.0),
    )
    fig, pes_metrics = plot_energy_landscape_analysis(
        env, gamma=0.96, source_cell=(7, 3), sink_a=(3, 17), sink_b=(11, 17)
    )
    metrics = {
        "preset_id": pid,
        "paper_ref": meta["paper_ref"],
        "gamma": 0.96,
        **pes_metrics,
    }
    return env, fig, metrics
