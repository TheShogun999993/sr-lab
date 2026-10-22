"""Grounded Scientific Metrics for Hippocampal SR and Lattice Physics.

Computes exact numerical diagnostics returned to the LLM agent so every explanation
cites verified ground-truth numbers from the simulation.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence
import numpy as np

from .env import Coord, SREnvironment


def check_row_sums(M: np.ndarray, gamma: float) -> Dict[str, float]:
    """Verify the Markov Resolvent invariant: each row of M sums to 1 / (1 - gamma)."""
    row_sums = M.sum(axis=1)
    expected = 1.0 / (1.0 - gamma)
    return {
        "expected_row_sum": round(float(expected), 4),
        "mean_row_sum": round(float(np.mean(row_sums)), 4),
        "max_abs_error": float(np.max(np.abs(row_sums - expected))),
    }


def shortest_path_distances(env: SREnvironment, source_idx: int) -> np.ndarray:
    """Compute geodesic (shortest graph path) distances from `source_idx` to all states."""
    n = env.n_states
    dist = np.full(n, np.inf, dtype=float)
    dist[source_idx] = 0.0
    q = deque([source_idx])
    while q:
        u = q.popleft()
        for v in np.where(env.A[u] > 0)[0]:
            if np.isinf(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(int(v))
    return dist


def place_field_metrics(
    env: SREnvironment,
    M: np.ndarray,
    target_cell: Coord,
    sigma: float = 2.2,
) -> Dict[str, object]:
    """Compute quantitative metrics for the SR place field / Green's function column M(:, s')."""
    t_cell = env.nearest_free_cell(*target_cell)
    t_idx = env.cell_to_idx[t_cell]

    place_field = M[:, t_idx]  # Column M(:, s'): how soon each state s visits target_cell
    pop_vector = M[t_idx, :]   # Row M(s', :): future states predicted from target_cell

    # Euclidean distance & Gaussian control field (Paper Fig. 2a)
    tr, tc = t_cell
    euclid_dist = np.array([np.hypot(r - tr, c - tc) for r, c in env.cells], dtype=float)
    euclid_field = np.exp(-(euclid_dist**2) / (2.0 * sigma**2))

    # Geodesic distance & Geodesic Gaussian control field (Paper Fig. 2b)
    geo_dist = shortest_path_distances(env, t_idx)
    geo_field = np.exp(-(geo_dist**2) / (2.0 * sigma**2))

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a0, b0 = a - a.mean(), b - b.mean()
        denom = (np.linalg.norm(a0) + 1e-12) * (np.linalg.norm(b0) + 1e-12)
        return float(np.dot(a0, b0) / denom)

    # Center of mass of the place field
    weights = place_field / max(1e-12, place_field.sum())
    com_r = float(sum(w * r for w, (r, c) in zip(weights, env.cells)))
    com_c = float(sum(w * c for w, (r, c) in zip(weights, env.cells)))

    # If there is a vertical wall or barrier near the middle column, measure cross-wall suppression
    mid_c = env.width // 2
    same_side = [
        place_field[i]
        for i, (r, c) in enumerate(env.cells)
        if (c < mid_c if tc < mid_c else c > mid_c) and abs(r - tr) <= 2 and abs(c - mid_c) <= 2
    ]
    opp_side = [
        place_field[i]
        for i, (r, c) in enumerate(env.cells)
        if (c > mid_c if tc < mid_c else c < mid_c) and abs(r - tr) <= 2 and abs(c - mid_c) <= 2
    ]
    mean_same = float(np.mean(same_side)) if same_side else float(place_field[t_idx])
    mean_opp = float(np.mean(opp_side)) if opp_side else 0.0
    wall_suppression_ratio = round(mean_opp / max(1e-9, mean_same), 4)

    return {
        "target_cell": t_cell,
        "peak_value": round(float(place_field[t_idx]), 4),
        "field_sum": round(float(place_field.sum()), 4),
        "center_of_mass_rc": (round(com_r, 2), round(com_c, 2)),
        "corr_with_euclidean_gaussian": round(_corr(place_field, euclid_field), 4),
        "corr_with_geodesic_gaussian": round(_corr(place_field, geo_field), 4),
        "near_wall_same_side_mean": round(mean_same, 4),
        "near_wall_opposite_side_mean": round(mean_opp, 4),
        "cross_wall_leakage_ratio": wall_suppression_ratio,
        "euclid_field": euclid_field,
        "geo_field": geo_field,
        "place_field": place_field,
        "pop_vector": pop_vector,
    }


def track_skew_metrics(field_1d: np.ndarray, target_idx: int) -> Dict[str, float]:
    """Quantify backward vs forward skew of a 1D place field (Mehta et al. 2000, Paper Fig. 3b)."""
    n = len(field_1d)
    positions = np.arange(n, dtype=float)
    mass_behind = float(field_1d[:target_idx].sum())
    mass_ahead = float(field_1d[target_idx + 1 :].sum())
    total_mass = float(field_1d.sum())
    com = float(np.dot(positions, field_1d) / max(1e-12, total_mass))
    com_shift = com - float(target_idx)  # Negative = shifted backwards (upstream of travel)
    skew_ratio = mass_behind / max(1e-6, mass_ahead)

    return {
        "target_idx": int(target_idx),
        "peak_firing": round(float(field_1d[target_idx]), 4),
        "mass_behind_target": round(mass_behind, 3),
        "mass_ahead_of_target": round(mass_ahead, 3),
        "backward_to_forward_ratio": round(skew_ratio, 2),
        "center_of_mass": round(com, 2),
        "com_shift_from_target": round(com_shift, 2),
    }


def barrier_remapping_metrics(
    env_open: SREnvironment,
    M_open: np.ndarray,
    env_barrier: SREnvironment,
    M_barrier: np.ndarray,
    near_cells: Sequence[Coord],
    far_cells: Sequence[Coord],
) -> Dict[str, object]:
    """Compute local vs distal place-field remapping when a barrier is inserted (Alvernhe 2011, Fig. 3e,h)."""
    common_cells = [c for c in env_open.cells if c in env_barrier.cell_to_idx]
    idx_open = [env_open.cell_to_idx[c] for c in common_cells]
    idx_bar = [env_barrier.cell_to_idx[c] for c in common_cells]

    def _cell_similarity(cell: Coord) -> Optional[float]:
        if cell not in env_open.cell_to_idx or cell not in env_barrier.cell_to_idx:
            return None
        f_open = M_open[idx_open, env_open.cell_to_idx[cell]]
        f_bar = M_barrier[idx_bar, env_barrier.cell_to_idx[cell]]
        a = f_open - f_open.mean()
        b = f_bar - f_bar.mean()
        r = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
        return float(np.clip(r, -0.9999, 0.9999))

    near_r = [r for c in near_cells if (r := _cell_similarity(c)) is not None]
    far_r = [r for c in far_cells if (r := _cell_similarity(c)) is not None]

    near_z = [float(np.arctanh(r)) for r in near_r]
    far_z = [float(np.arctanh(r)) for r in far_r]

    return {
        "near_mean_r": round(float(np.mean(near_r)), 4) if near_r else 0.0,
        "far_mean_r": round(float(np.mean(far_r)), 4) if far_r else 0.0,
        "near_mean_fisher_z": round(float(np.mean(near_z)), 3) if near_z else 0.0,
        "far_mean_fisher_z": round(float(np.mean(far_z)), 3) if far_z else 0.0,
        "remapping_drop_r": round(float(np.mean(far_r) - np.mean(near_r)), 4) if (near_r and far_r) else 0.0,
        "near_r_list": [round(x, 4) for x in near_r],
        "far_r_list": [round(x, 4) for x in far_r],
    }


def community_similarity_metrics(M: np.ndarray, communities: Sequence[int]) -> Dict[str, float]:
    """Compute within-community vs between-community SR row correlations (Schapiro et al., Fig. 4g)."""
    n = M.shape[0]
    corr_mat = np.corrcoef(M)
    within: List[float] = []
    between: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            c_val = float(corr_mat[i, j])
            if communities[i] == communities[j]:
                within.append(c_val)
            else:
                between.append(c_val)

    return {
        "within_community_mean_corr": round(float(np.mean(within)), 4),
        "between_community_mean_corr": round(float(np.mean(between)), 4),
        "community_contrast_diff": round(float(np.mean(within) - np.mean(between)), 4),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
    }
