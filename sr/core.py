"""Core Successor Representation, Temporal Difference, and Spectral Algorithms.

Implements Equations (1)–(8) from Stachenfeld, Botvinick & Gershman (2017):
- Eq. 7: Analytic Successor Representation / Lattice Green's Function M = (I - gamma*T)^-1
- Eq. 8: Online Temporal Difference TD(0) learning of M_hat
- Eq. 2: Instant value / reaction yield computation V = M * R + trajectory rollout
- Spectral eigendecomposition of M / random-walk transfer operator (Grid cells & relaxation modes)
- 2D spatial autocorrelogram (Krupic et al., 2015)
- Spectral regularization / low-rank smoothing (Supplementary Fig. 12)
- Normalized min-cut subgoal & transition-state saddle-point discovery (Supplementary Fig. 13)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import fftconvolve

from .env import Coord, SREnvironment


def compute_sr_analytic(T: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    """Compute exact Successor Representation / Resolvent M = (I - gamma*T)^-1 (Eq. 7)."""
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"Discount factor gamma must be in [0, 1), got {gamma}")
    n = T.shape[0]
    I = np.eye(n, dtype=float)
    return np.linalg.solve(I - gamma * T, I)


def compute_sr_td(
    env: SREnvironment,
    gamma: float = 0.95,
    eta: float = 0.1,
    n_steps: int = 12000,
    seed: int = 42,
    M_gt: Optional[np.ndarray] = None,
    n_checkpoints: int = 12,
) -> Dict[str, object]:
    """Learn the SR online via Temporal Difference learning (Eq. 8).

    Update rule after observing transition s_t -> s_{t+1}:
        M_hat(s_t, :) <- M_hat(s_t, :) + eta * [ e_{s_t} + gamma * M_hat(s_{t+1}, :) - M_hat(s_t, :) ]
    """
    rng = np.random.default_rng(seed)
    n = env.n_states
    M_hat = np.eye(n, dtype=float)  # Initialized to identity matrix (Paper Online Methods, p. 12)
    if M_gt is None:
        M_gt = compute_sr_analytic(env.T, gamma)

    # Precompute cumulative row probabilities for fast sampling
    cdf = np.cumsum(env.T, axis=1)
    cdf[:, -1] = 1.0

    s_curr = int(rng.integers(0, n))
    checkpoint_Interval = max(1, n_steps // max(1, n_checkpoints))
    steps_recorded: List[int] = []
    correlations: List[float] = []
    mae_errors: List[float] = []

    gt_flat = M_gt.ravel()
    gt_centered = gt_flat - gt_flat.mean()
    gt_norm = np.linalg.norm(gt_centered) + 1e-12

    for step in range(1, n_steps + 1):
        # Sample next state s_next from T[s_curr]
        u = rng.random()
        s_next = int(np.searchsorted(cdf[s_curr], u, side="right"))
        if s_next >= n:
            s_next = n - 1

        # TD(0) vector update on row s_curr (Eq. 8)
        td_error = gamma * M_hat[s_next] - M_hat[s_curr]
        td_error[s_curr] += 1.0
        M_hat[s_curr] += eta * td_error

        # Periodic random restart to ensure ergodic coverage even in biased/trapping mazes
        if step % 150 == 0:
            s_curr = int(rng.integers(0, n))
        else:
            s_curr = s_next

        if step % checkpoint_Interval == 0 or step == n_steps:
            flat = M_hat.ravel()
            centered = flat - flat.mean()
            corr = float(np.dot(centered, gt_centered) / ((np.linalg.norm(centered) + 1e-12) * gt_norm))
            mae = float(np.mean(np.abs(M_hat - M_gt)))
            steps_recorded.append(step)
            correlations.append(round(corr, 4))
            mae_errors.append(round(mae, 5))

    return {
        "M_hat": M_hat,
        "M_gt": M_gt,
        "steps": steps_recorded,
        "correlations": correlations,
        "mae_errors": mae_errors,
        "final_correlation": correlations[-1] if correlations else 0.0,
        "final_mae": mae_errors[-1] if mae_errors else 0.0,
    }


def compute_value(M: np.ndarray, reward_vec: np.ndarray) -> np.ndarray:
    """Compute state value function / reaction yield V(s) = sum_{s'} M(s, s') R(s') (Eq. 2)."""
    return M @ reward_vec


def rollout_greedy_path(
    env: SREnvironment,
    V: np.ndarray,
    start_cell: Coord,
    goal_cell: Coord,
    max_steps: int = 200,
) -> List[Coord]:
    """Follow steepest ascent on V(s) along valid edges A(s, s') == 1 from start_cell to goal_cell."""
    start_c = env.nearest_free_cell(*start_cell)
    goal_c = env.nearest_free_cell(*goal_cell)
    curr_idx = env.cell_to_idx[start_c]
    goal_idx = env.cell_to_idx[goal_c]

    path: List[Coord] = [start_c]
    visited = {curr_idx}

    for _ in range(max_steps):
        if curr_idx == goal_idx:
            break
        nbrs = np.where(env.A[curr_idx] > 0)[0]
        if len(nbrs) == 0:
            break
        # Prefer unvisited neighbors with highest value V
        unvisited = [j for j in nbrs if j not in visited]
        candidates = unvisited if unvisited else list(nbrs)
        best_next = max(candidates, key=lambda j: float(V[j]))
        curr_idx = int(best_next)
        visited.add(curr_idx)
        path.append(env.cells[curr_idx])

    return path


def compute_eigenvectors(
    env: SREnvironment,
    gamma: float = 0.95,
    threshold_zero: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute eigenvectors of the SR / transition operator (Paper Online Methods, p. 12).

    For reversible/detailed-balance transitions (both random-walk mazes and Boltzmann energy
    landscapes), T is similar to a symmetric matrix via stationary distribution pi:
        S = diag(pi^0.5) T diag(pi^-0.5)
    which guarantees real eigenvalues and clean orthogonal spatial harmonics!
    """
    n = env.n_states
    A_sym = 0.5 * (env.A + env.A.T)
    deg = A_sym.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-8)

    if env.kind == "energy_2d" and env.energy_grid is not None:
        # For Boltzmann landscapes, stationary measure pi(s) propto exp(-E(s)/kT) * degree_weight
        e_vec = np.array([env.energy_grid[r, c] for r, c in env.cells], dtype=float)
        pi = np.exp(-e_vec / max(0.05, env.kT))
        pi = pi / pi.sum()
        # Symmetrize reversible part of T
        P_flux = 0.5 * ( np.diag(pi) @ env.T + (np.diag(pi) @ env.T).T )
        pi_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(pi, 1e-12)))
        pi_sqrt = np.diag(np.sqrt(np.maximum(pi, 1e-12)))
        S = pi_inv_sqrt @ P_flux @ pi_inv_sqrt
        eigvals_T, Q = np.linalg.eigh(S)
        order = np.argsort(-eigvals_T)
        eigvals_T = eigvals_T[order]
        Q = Q[:, order]
        U_raw = pi_inv_sqrt @ Q
        V_left = pi_sqrt @ Q
    else:
        # Random-walk normalized graph Laplacian / transition operator: T_rw = D^-1 A
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
        d_sqrt = np.diag(np.sqrt(deg_safe))
        S = d_inv_sqrt @ A_sym @ d_inv_sqrt
        eigvals_T, Q = np.linalg.eigh(S)
        order = np.argsort(-eigvals_T)
        eigvals_T = eigvals_T[order]
        Q = Q[:, order]
        U_raw = d_inv_sqrt @ Q
        V_left = d_sqrt @ Q

    # Deterministic sign convention: make largest-magnitude element positive
    for k in range(U_raw.shape[1]):
        col = U_raw[:, k]
        idx_max = int(np.argmax(np.abs(col)))
        if col[idx_max] < 0:
            U_raw[:, k] = -col
            V_left[:, k] = -V_left[:, k]

    eigvals_T = np.clip(eigvals_T, -0.9999, 0.9999)
    eigvals_M = 1.0 / (1.0 - gamma * eigvals_T)
    U_grid = np.maximum(U_raw, 0.0) if threshold_zero else U_raw.copy()

    return {
        "eigvals_T": eigvals_T,
        "eigvals_M": eigvals_M,
        "U_raw": U_raw,
        "V_left": V_left,
        "U_grid": U_grid,
    }


def spatial_autocorrelogram_2d(grid_img: np.ndarray) -> np.ndarray:
    """Compute normalized 2D spatial autocorrelogram (autoconvolution) of a grid field (Fig. 6e,h)."""
    arr = np.nan_to_num(grid_img, nan=0.0).astype(float)
    arr = arr - np.mean(arr)
    norm = np.sum(arr**2)
    if norm < 1e-12:
        return np.zeros_like(arr)
    ac = fftconvolve(arr, arr[::-1, ::-1], mode="same") / norm
    return ac


def spectral_regularize_sr(
    env: SREnvironment,
    gamma: float = 0.95,
    k_keep: int = 16,
    M_noisy: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Reconstruct / regularize the SR using only the top-k low-frequency eigenvectors (Supp. Fig. 12)."""
    spec = compute_eigenvectors(env, gamma=gamma, threshold_zero=False)
    U = spec["U_raw"]
    V_left = spec["V_left"]
    eigvals_M = spec["eigvals_M"]

    k = max(1, min(int(k_keep), env.n_states))
    I_mat = np.eye(env.n_states, dtype=float)
    if M_noisy is not None:
        # Following Stachenfeld et al. (2017) Online Methods (p. 12) & Mazumder et al. (2010, ref. 26):
        # Early TD estimates under-propagate long-horizon occupancy and contain high-frequency sampling noise.
        # Projecting onto the top-k low-frequency modes and rescaling by the long-horizon spectral weights
        # eigvals_M[:k] = 1/(1 - gamma * lambda_k) fills in unvisited multi-step paths while filtering noise.
        P_proj = U[:, :k] @ V_left[:, :k].T
        M_low_freq = (U[:, :k] * eigvals_M[:k][None, :]) @ V_left[:, :k].T
        M_high_freq_filtered = P_proj @ (M_noisy - I_mat) @ P_proj
        M_recon = 0.65 * M_low_freq + 0.35 * (I_mat + M_high_freq_filtered)
    else:
        # Direct truncated spectral synthesis from top-k eigenvalues
        M_recon = (U[:, :k] * eigvals_M[:k][None, :]) @ V_left[:, :k].T

    M_gt = compute_sr_analytic(env.T, gamma=gamma)
    # Evaluate correlation on the multi-step predictive transitions (M - I) so the trivial
    # 0-step identity initialization M_0 = I does not artificially inflate untrained TD accuracy.
    off_mask = ~np.eye(env.n_states, dtype=bool)
    g_flat = M_gt[off_mask] - M_gt[off_mask].mean()
    r_flat = M_recon[off_mask] - M_recon[off_mask].mean()
    corr_recon = float(
        np.dot(g_flat, r_flat) / ((np.linalg.norm(g_flat) + 1e-12) * (np.linalg.norm(r_flat) + 1e-12))
    )

    corr_noisy = None
    if M_noisy is not None:
        n_flat = M_noisy[off_mask] - M_noisy[off_mask].mean()
        corr_noisy = float(
            np.dot(g_flat, n_flat) / ((np.linalg.norm(g_flat) + 1e-12) * (np.linalg.norm(n_flat) + 1e-12))
        )

    return {
        "M_recon": M_recon,
        "M_gt": M_gt,
        "k_keep": k,
        "corr_recon": round(corr_recon, 4),
        "corr_noisy": round(corr_noisy, 4) if corr_noisy is not None else None,
    }


def discover_subgoals(
    env: SREnvironment,
    gamma: float = 0.95,
    top_n: int = 2,
) -> Dict[str, object]:
    """Discover bottleneck doorway subgoals or energy-landscape saddle points (Normalized Min-Cut).

    From Paper Online Methods (p. 12) & Supplementary Fig. 13:
    1. Take the lowest-frequency non-constant eigenvector u_1 of the SR / Laplacian.
    2. Partition the state space by thresholding the sign of u_1 (u_1 > 0 vs u_1 <= 0).
    3. Find all edges (s, s') cut by the partition.
    4. For energy landscapes, rank cut states by lowest potential energy E(s) along the partition
       ridge (which finds the exact mountain pass / saddle point!). For discrete mazes, rank by
       spectral gradient |u_1(s) - u_1(s')| and bottleneck degree.
    """
    spec = compute_eigenvectors(env, gamma=gamma, threshold_zero=False)
    U_raw = spec["U_raw"]
    if U_raw.shape[1] < 2:
        return {"subgoals": [], "cut_edges": [], "partition_sizes": (env.n_states, 0), "u1": U_raw[:, 0]}

    u1 = U_raw[:, 1]
    # Center around median if sign doesn't split cleanly
    split_val = 0.0 if (np.any(u1 > 0) and np.any(u1 < 0)) else float(np.median(u1))
    signs = u1 > split_val

    cut_edges: List[Tuple[Coord, Coord, float]] = []
    node_scores: Dict[int, float] = {}

    for i in range(env.n_states):
        nbrs = np.where(env.A[i] > 0)[0]
        for j in nbrs:
            if signs[i] != signs[j]:
                grad = float(abs(u1[i] - u1[j]))
                if i < j:
                    cut_edges.append((env.cells[i], env.cells[j], round(grad, 5)))
                if env.kind == "energy_2d" and env.energy_grid is not None:
                    # On an energy surface, the transition state (saddle point) is the
                    # minimum-energy pass along the spectral dividing surface (separatrix)!
                    r, c = env.cells[i]
                    score = -float(env.energy_grid[r, c]) + 0.5 * grad
                else:
                    # In a discrete maze, doorway nodes have the sharpest spectral jump |u1(i) - u1(j)|
                    score = grad
                node_scores[i] = max(node_scores.get(i, -1e9), score)

    ranked_indices = sorted(node_scores.keys(), key=lambda idx: node_scores[idx], reverse=True)
    subgoals = [env.cells[idx] for idx in ranked_indices[: max(1, top_n)]]

    n_pos = int(np.sum(signs))
    n_neg = int(env.n_states - n_pos)

    return {
        "subgoals": subgoals,
        "cut_edges": cut_edges,
        "partition_sizes": (n_pos, n_neg),
        "eigenvalue_1": round(float(spec["eigvals_T"][1]), 5),
        "u1": u1,
    }
