"""Stateful LabSession and 8 Structured Scientific Tools for the SR-Lab Agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from sr.env import (
    SREnvironment,
    build_energy_landscape_env,
    build_grid_env,
)
from sr.core import compute_sr_analytic
from sr.metrics import check_row_sums, place_field_metrics
from sr.plots import (
    plot_eigenvectors_and_subgoals,
    plot_energy_landscape_analysis,
    plot_place_field_comparison,
    plot_reward_revaluation,
    plot_td_and_spectral_regularization,
    plot_track_skew,
)
from sr.presets import run_preset
from .schemas import (
    BuildEnergyLandscapeArgs,
    BuildGridEnvArgs,
    GridAndSubgoalsArgs,
    InspectPlaceFieldArgs,
    RewardRevaluationArgs,
    RunPresetArgs,
    TdVsAnalyticArgs,
    TrackSkewArgs,
)


@dataclass
class LabSession:
    """Holds the current environment, SR matrix, latest figure, and tool execution log."""

    env: SREnvironment = field(default_factory=lambda: build_grid_env(12, 12, wall_preset="middle_wall_bottom_gap"))
    gamma: float = 0.95
    domain: str = "neuro"  # "neuro" or "physics"
    M: Optional[np.ndarray] = None
    current_fig: Optional[plt.Figure] = None
    current_metrics: Dict[str, Any] = field(default_factory=dict)
    active_title: str = "P1: Place Fields Warp Around Obstacles (Fig. 2c)"
    tool_logs: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.M is None:
            self.M = compute_sr_analytic(self.env.T, gamma=self.gamma)
        if self.current_fig is None:
            self.execute_tool("run_paper_preset", {"preset_id": "P1_wall_warp"})

    def execute_tool(self, tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate arguments with Pydantic, run the simulation tool, update state, and return JSON metrics."""
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}")
        schema_cls, handler = TOOL_REGISTRY[tool_name]
        validated = schema_cls(**(raw_args or {}))
        result = handler(self, validated)
        self.current_metrics = result
        self.tool_logs.append(
            {
                "tool": tool_name,
                "args": validated.model_dump(),
                "output": result,
            }
        )
        return result


# ------------------------------------------------------------------ Handlers --
def _handle_build_grid_env(session: LabSession, args: BuildGridEnvArgs) -> Dict[str, Any]:
    bias = {}
    if args.bias_direction and args.bias_direction.lower() != "none":
        bias[args.bias_direction.lower()] = float(args.bias_strength)

    session.env = build_grid_env(
        height=args.height,
        width=args.width,
        wall_preset=args.wall_preset,
        shape=args.shape,
        lattice=args.lattice,
        bias=bias,
    )
    session.gamma = float(args.gamma)
    session.M = compute_sr_analytic(session.env.T, gamma=session.gamma)

    tr = args.target_row if args.target_row is not None else args.height // 3
    tc = args.target_col if args.target_col is not None else max(0, args.width // 2 - 1)
    target = session.env.nearest_free_cell(tr, tc)

    session.current_fig = plot_place_field_comparison(session.env, session.M, target, domain=session.domain)
    session.active_title = f"Custom {args.height}x{args.width} Grid ({args.wall_preset}, γ={session.gamma})"

    pf = place_field_metrics(session.env, session.M, target)
    row_chk = check_row_sums(session.M, session.gamma)
    return {
        "environment": f"{args.height}x{args.width} {args.shape} ({args.wall_preset})",
        "lattice": args.lattice,
        "n_free_states": session.env.n_states,
        "n_walls": len(session.env.walls),
        "gamma": session.gamma,
        "target_cell": pf["target_cell"],
        "peak_sr_value": pf["peak_value"],
        "center_of_mass_rc": pf["center_of_mass_rc"],
        "corr_with_euclidean_gaussian": pf["corr_with_euclidean_gaussian"],
        "corr_with_geodesic_gaussian": pf["corr_with_geodesic_gaussian"],
        "near_wall_same_side_mean": pf["near_wall_same_side_mean"],
        "near_wall_opposite_side_mean": pf["near_wall_opposite_side_mean"],
        "cross_wall_leakage_ratio": pf["cross_wall_leakage_ratio"],
        **row_chk,
    }


def _handle_build_energy_landscape(session: LabSession, args: BuildEnergyLandscapeArgs) -> Dict[str, Any]:
    session.domain = "physics"
    session.env = build_energy_landscape_env(
        height=args.height,
        width=args.width,
        landscape_type=args.landscape_type,
        kT=args.kT,
        barrier_height=args.barrier_height,
        drift_rc=(args.drift_r, args.drift_c),
    )
    session.gamma = float(args.gamma)
    session.M = compute_sr_analytic(session.env.T, gamma=session.gamma)
    fig, metrics = plot_energy_landscape_analysis(session.env, gamma=session.gamma)
    session.current_fig = fig
    session.active_title = f"Boltzmann PES: {args.landscape_type} (kT={args.kT}, γ={session.gamma})"
    return {
        "grid_dimensions": (args.height, args.width),
        "n_states": session.env.n_states,
        "gamma": session.gamma,
        **metrics,
        **check_row_sums(session.M, session.gamma),
    }


def _handle_inspect_place_field(session: LabSession, args: InspectPlaceFieldArgs) -> Dict[str, Any]:
    target = session.env.nearest_free_cell(args.row, args.col)
    if session.M is None:
        session.M = compute_sr_analytic(session.env.T, gamma=session.gamma)
    session.current_fig = plot_place_field_comparison(session.env, session.M, target, domain=args.domain)
    session.active_title = f"Place Field / Green's Function at s'={target}"
    pf = place_field_metrics(session.env, session.M, target)
    return {
        "target_cell": pf["target_cell"],
        "peak_value": pf["peak_value"],
        "field_sum": pf["field_sum"],
        "center_of_mass_rc": pf["center_of_mass_rc"],
        "corr_with_euclidean_gaussian": pf["corr_with_euclidean_gaussian"],
        "corr_with_geodesic_gaussian": pf["corr_with_geodesic_gaussian"],
        "near_wall_same_side_mean": pf["near_wall_same_side_mean"],
        "near_wall_opposite_side_mean": pf["near_wall_opposite_side_mean"],
        "cross_wall_leakage_ratio": pf["cross_wall_leakage_ratio"],
    }


def _handle_track_skew(session: LabSession, args: TrackSkewArgs) -> Dict[str, Any]:
    session.gamma = float(args.gamma)
    fig, skew_data = plot_track_skew(
        length=args.length,
        target_idx=min(args.target_idx, args.length - 2),
        p_right_biased=args.p_right,
        gamma=args.gamma,
        domain=args.domain,
    )
    session.current_fig = fig
    session.active_title = f"1D Track Predictive Skew (L={args.length}, p_right={args.p_right})"
    return {
        "track_length": args.length,
        "target_state": min(args.target_idx, args.length - 2),
        "gamma": args.gamma,
        "biased_p_right": args.p_right,
        "unbiased_mass_behind": skew_data["unbiased"]["mass_behind_target"],
        "unbiased_mass_ahead": skew_data["unbiased"]["mass_ahead_of_target"],
        "biased_mass_behind": skew_data["biased"]["mass_behind_target"],
        "biased_mass_ahead": skew_data["biased"]["mass_ahead_of_target"],
        "biased_backward_to_forward_ratio": skew_data["biased"]["backward_to_forward_ratio"],
        "biased_center_of_mass": skew_data["biased"]["center_of_mass"],
        "biased_com_shift": skew_data["biased"]["com_shift_from_target"],
    }


def _handle_reward_revaluation(session: LabSession, args: RewardRevaluationArgs) -> Dict[str, Any]:
    if session.env.kind in ("track_1d", "graph"):
        session.env = build_grid_env(12, 12, wall_preset="middle_wall_bottom_gap")
    session.M = compute_sr_analytic(session.env.T, gamma=session.gamma)
    fig, metrics = plot_reward_revaluation(
        session.env,
        session.M,
        start_cell=(args.start_row, args.start_col),
        goal_a=(args.goal_a_row, args.goal_a_col),
        goal_b=(args.goal_b_row, args.goal_b_col),
        domain=args.domain,
    )
    session.current_fig = fig
    session.active_title = f"Zero-Shot Reward Revaluation V = M·R (Goal {metrics['goal_a']} → {metrics['goal_b']})"
    return {"gamma": session.gamma, **metrics}


def _handle_grid_and_subgoals(session: LabSession, args: GridAndSubgoalsArgs) -> Dict[str, Any]:
    fig, metrics = plot_eigenvectors_and_subgoals(
        session.env,
        gamma=session.gamma,
        mode_indices=args.mode_indices,
        top_n_subgoals=args.top_n_subgoals,
        domain=args.domain,
    )
    session.current_fig = fig
    session.active_title = f"Spectral Grid Modes & Bottleneck Discovery ({session.env.height}x{session.env.width})"
    return {"gamma": session.gamma, "n_free_states": session.env.n_states, **metrics}


def _handle_td_vs_analytic(session: LabSession, args: TdVsAnalyticArgs) -> Dict[str, Any]:
    fig, metrics = plot_td_and_spectral_regularization(
        session.env,
        gamma=session.gamma,
        n_steps=args.n_steps,
        eta=args.eta,
        k_keep=args.k_keep,
    )
    session.current_fig = fig
    session.active_title = f"Online TD(0) Convergence & Spectral Regularization ({args.n_steps} steps)"
    return {"gamma": session.gamma, **metrics}


def _handle_run_preset(session: LabSession, args: RunPresetArgs) -> Dict[str, Any]:
    env, fig, metrics = run_preset(args.preset_id)
    session.env = env
    session.gamma = float(metrics.get("gamma", 0.95))
    session.M = compute_sr_analytic(env.T, gamma=session.gamma)
    session.current_fig = fig
    from sr.presets import PRESETS

    session.active_title = PRESETS[args.preset_id]["title"]
    session.domain = PRESETS[args.preset_id]["domain"]
    return metrics


TOOL_REGISTRY: Dict[str, tuple[Any, Callable[[LabSession, Any], Dict[str, Any]]]] = {
    "build_grid_environment": (BuildGridEnvArgs, _handle_build_grid_env),
    "build_energy_landscape": (BuildEnergyLandscapeArgs, _handle_build_energy_landscape),
    "inspect_place_field": (InspectPlaceFieldArgs, _handle_inspect_place_field),
    "analyze_1d_track_skew": (TrackSkewArgs, _handle_track_skew),
    "evaluate_reward_revaluation": (RewardRevaluationArgs, _handle_reward_revaluation),
    "compute_grid_and_subgoals": (GridAndSubgoalsArgs, _handle_grid_and_subgoals),
    "compare_td_vs_analytic": (TdVsAnalyticArgs, _handle_td_vs_analytic),
    "run_paper_preset": (RunPresetArgs, _handle_run_preset),
}

TOOL_DEFINITIONS = [
    {
        "name": name,
        "description": handler.__doc__ or f"Execute {name} on the SR-Lab simulation engine.",
        "parameters": schema_cls.model_json_schema(),
    }
    for name, (schema_cls, handler) in TOOL_REGISTRY.items()
]
