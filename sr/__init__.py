"""Unified Successor Representation & Lattice Physics Engine (sr/).

Implements both:
1. Hippocampal & Reinforcement Learning Spatial Navigation (Stachenfeld et al., 2017)
2. Statistical Physics & Physical Chemistry Boltzmann Energy Landscapes (Resolvent / Green's Function)
"""
from .env import (
    SREnvironment,
    build_grid_env,
    build_track_env,
    build_schapiro_graph,
    build_energy_landscape_env,
    compute_softmax_policy_T,
)
from .core import (
    compute_sr_analytic,
    compute_sr_td,
    compute_value,
    rollout_greedy_path,
    compute_eigenvectors,
    spatial_autocorrelogram_2d,
    spectral_regularize_sr,
    discover_subgoals,
)
from .metrics import (
    check_row_sums,
    place_field_metrics,
    track_skew_metrics,
    barrier_remapping_metrics,
    community_similarity_metrics,
)
from .presets import PRESETS, run_preset, list_presets

__all__ = [
    "SREnvironment",
    "build_grid_env",
    "build_track_env",
    "build_schapiro_graph",
    "build_energy_landscape_env",
    "compute_softmax_policy_T",
    "compute_sr_analytic",
    "compute_sr_td",
    "compute_value",
    "rollout_greedy_path",
    "compute_eigenvectors",
    "spatial_autocorrelogram_2d",
    "spectral_regularize_sr",
    "discover_subgoals",
    "check_row_sums",
    "place_field_metrics",
    "track_skew_metrics",
    "barrier_remapping_metrics",
    "community_similarity_metrics",
    "PRESETS",
    "run_preset",
    "list_presets",
]
