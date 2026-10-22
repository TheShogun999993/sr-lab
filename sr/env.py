"""Spatial, Relational, and Potential Energy Surface (PES) Environment Builders.

Unifies:
- 2D discrete mazes (4-neighbor square or 6-neighbor triangular lattice) with walls,
  doors, detours, and geometric arenas (square, circle, trapezoid, hairpin).
- 1D linear and circular (annular ring) tracks with directional transition bias.
- Non-spatial relational graphs (Schapiro et al. 15-node community graph).
- 2D Statistical Physics Potential Energy Surfaces E(x, y) with thermal energy k_B*T
  and external drift field F = (F_r, F_c), governed by Metropolis-Hastings / Arrhenius
  transition rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

Coord = Tuple[int, int]


@dataclass
class SREnvironment:
    """Unified discrete Markov state-space environment for SR and Lattice Physics."""

    kind: str  # "grid_2d", "track_1d", "ring_1d", "graph", "energy_2d"
    height: int
    width: int
    cells: List[Coord]
    cell_to_idx: Dict[Coord, int]
    walls: Set[Coord]
    A: np.ndarray  # (N, N) adjacency matrix (0/1)
    T: np.ndarray  # (N, N) row-stochastic transition matrix
    coords: np.ndarray  # (N, 2) float (x, y) plotting coordinates
    lattice: str = "square"
    energy_grid: Optional[np.ndarray] = None  # (H, W) potential energy E(r, c) for physics mode
    kT: float = 1.0
    drift: Tuple[float, float] = (0.0, 0.0)
    communities: Optional[List[int]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.cells)

    def to_grid(self, vec: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Reshape a length-N state vector onto an (H, W) 2D array (walls = fill_value)."""
        img = np.full((self.height, self.width), fill_value, dtype=float)
        for val, (r, c) in zip(vec, self.cells):
            if 0 <= r < self.height and 0 <= c < self.width:
                img[r, c] = float(val)
        return img

    def nearest_free_cell(self, r: int, c: int) -> Coord:
        """Return `(r, c)` if free, otherwise the closest free cell by Euclidean distance."""
        if (r, c) in self.cell_to_idx:
            return (r, c)
        best = min(self.cells, key=lambda rc: (rc[0] - r) ** 2 + (rc[1] - c) ** 2)
        return best


def parse_wall_preset(height: int, width: int, preset_name: str) -> Set[Coord]:
    """Generate standard wall sets by name for quick natural-language construction."""
    walls: Set[Coord] = set()
    name = (preset_name or "open").lower().strip()

    if name in ("open", "none", "empty", ""):
        return walls

    mid_c = width // 2
    mid_r = height // 2

    if name in ("middle_wall_bottom_gap", "wall", "barrier", "vertical_wall"):
        # Wall from top down to ~75% height, leaving a gap at the bottom (Paper Fig. 2c)
        gap_start = max(1, int(round(height * 0.75)))
        for r in range(0, gap_start):
            walls.add((r, mid_c))

    elif name in ("middle_wall_center_door", "two_rooms", "doorway", "bottleneck"):
        # Full vertical wall splitting two rooms with a single central doorway (Paper Supp. Fig. 13)
        for r in range(height):
            if r != mid_r:
                walls.add((r, mid_c))

    elif name in ("four_rooms", "4_rooms", "quad_rooms"):
        door_r1, door_r2 = height // 4, (3 * height) // 4
        door_c1, door_c2 = width // 4, (3 * width) // 4
        for r in range(height):
            if r not in (door_r1, door_r2):
                walls.add((r, mid_c))
        for c in range(width):
            if c not in (door_c1, door_c2):
                walls.add((mid_r, c))

    elif name in ("hairpin", "hairpin_maze"):
        # Derdikman et al. (2009) hairpin maze with alternating vertical barriers
        n_arms = 5
        col_step = max(2, width // n_arms)
        for arm_idx in range(1, n_arms):
            c = arm_idx * col_step
            if c >= width - 1:
                continue
            if arm_idx % 2 == 1:
                for r in range(0, height - 2):
                    walls.add((r, c))
            else:
                for r in range(2, height):
                    walls.add((r, c))

    elif name in ("tolman_detour", "detour"):
        # Central corridor with a short left detour and longer right detour
        for r in range(height):
            for c in range(width):
                walls.add((r, c))
        # Carve main vertical stem
        for r in range(1, height - 1):
            walls.discard((r, mid_c))
        # Carve top and bottom horizontal bars
        for c in range(2, width - 2):
            walls.discard((1, c))
            walls.discard((height - 2, c))
        # Carve left short detour
        left_c = max(2, mid_c - 3)
        for r in range(2, height - 2):
            walls.discard((r, left_c))
        for c in range(left_c, mid_c + 1):
            walls.discard((mid_r + 1, c))
            walls.discard((2, c))
        # Carve right long detour
        right_c = min(width - 2, mid_c + 4)
        for r in range(1, height - 1):
            walls.discard((r, right_c))

    return walls


def shape_mask_walls(height: int, width: int, shape: str) -> Set[Coord]:
    """Return blocked cells outside a geometric boundary ('rectangle', 'circle', 'trapezoid')."""
    shape = (shape or "rectangle").lower().strip()
    walls: Set[Coord] = set()
    if shape in ("rectangle", "square"):
        return walls

    if shape == "circle":
        cr = (height - 1) / 2.0
        cc = (width - 1) / 2.0
        radius = min(height, width) / 2.0 - 0.35
        for r in range(height):
            for c in range(width):
                if (r - cr) ** 2 + (c - cc) ** 2 > radius ** 2:
                    walls.add((r, c))

    elif shape == "trapezoid":
        # Krupic et al. (2015) asymmetric trapezoid: narrow on the left, wide on the right
        for r in range(height):
            for c in range(width):
                frac_x = c / max(1, width - 1)  # 0 at left, 1 at right
                half_span = (0.22 + 0.28 * frac_x) * height
                center_r = (height - 1) / 2.0
                if abs(r - center_r) > half_span:
                    walls.add((r, c))

    return walls


def build_grid_env(
    height: int = 12,
    width: int = 12,
    walls: Optional[Iterable[Coord]] = None,
    wall_preset: str = "open",
    shape: str = "rectangle",
    lattice: str = "square",
    bias: Optional[Dict[str, float]] = None,
    stay_prob: float = 0.0,
) -> SREnvironment:
    """Construct a 2D discrete spatial environment on a square or triangular lattice."""
    wall_set: Set[Coord] = set(walls) if walls is not None else set()
    wall_set |= parse_wall_preset(height, width, wall_preset)
    wall_set |= shape_mask_walls(height, width, shape)

    cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in wall_set]
    idx = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)

    A = np.zeros((n, n), dtype=float)
    W_mat = np.zeros((n, n), dtype=float)

    bias = bias or {}
    w_north = float(bias.get("north", 1.0))
    w_south = float(bias.get("south", 1.0))
    w_east = float(bias.get("east", 1.0))
    w_west = float(bias.get("west", 1.0))

    coords = np.zeros((n, 2), dtype=float)

    for (r, c), i in idx.items():
        if lattice == "triangular":
            coords[i] = [c + (0.5 if r % 2 == 1 else 0.0), r * (np.sqrt(3) / 2.0)]
            if r % 2 == 0:
                neighbors = [(-1, 0), (-1, -1), (0, -1), (0, 1), (1, 0), (1, -1)]
            else:
                neighbors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        else:
            coords[i] = [float(c), float(r)]
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in neighbors:
            j = idx.get((r + dr, c + dc))
            if j is not None:
                A[i, j] = 1.0
                weight = 1.0
                if dr < 0:
                    weight *= w_north
                elif dr > 0:
                    weight *= w_south
                if dc > 0:
                    weight *= w_east
                elif dc < 0:
                    weight *= w_west
                W_mat[i, j] = max(1e-6, weight)

    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_sum = W_mat[i].sum()
        if row_sum > 0:
            T[i] = (1.0 - stay_prob) * (W_mat[i] / row_sum)
            T[i, i] += stay_prob
        else:
            T[i, i] = 1.0

    return SREnvironment(
        kind="grid_2d",
        height=height,
        width=width,
        cells=cells,
        cell_to_idx=idx,
        walls=wall_set,
        A=A,
        T=T,
        coords=coords,
        lattice=lattice,
        metadata={
            "wall_preset": wall_preset,
            "shape": shape,
            "bias": bias,
            "stay_prob": stay_prob,
        },
    )


def build_track_env(
    length: int = 41,
    p_right: float = 0.5,
    circular: bool = False,
    linger_zone: Optional[Sequence[int]] = None,
    linger_stay_prob: float = 0.0,
) -> SREnvironment:
    """Build a 1D linear track (Mehta 2000) or annular ring track (Hollup 2001)."""
    cells = [(0, c) for c in range(length)]
    idx = {cell: c for c, cell in enumerate(cells)}
    A = np.zeros((length, length), dtype=float)
    T = np.zeros((length, length), dtype=float)
    linger_set = set(linger_zone or [])

    for i in range(length):
        if circular:
            r_nb = (i + 1) % length
            l_nb = (i - 1) % length
        else:
            r_nb = min(i + 1, length - 1)
            l_nb = max(i - 1, 0)

        A[i, r_nb] = 1.0
        A[i, l_nb] = 1.0

        stay = linger_stay_prob if i in linger_set else 0.0
        move = 1.0 - stay
        T[i, r_nb] += move * p_right
        T[i, l_nb] += move * (1.0 - p_right)
        T[i, i] += stay

    coords = np.column_stack([np.arange(length, dtype=float), np.zeros(length, dtype=float)])
    return SREnvironment(
        kind="ring_1d" if circular else "track_1d",
        height=1,
        width=length,
        cells=cells,
        cell_to_idx=idx,
        walls=set(),
        A=A,
        T=T,
        coords=coords,
        metadata={
            "p_right": p_right,
            "circular": circular,
            "linger_zone": list(linger_set),
            "linger_stay_prob": linger_stay_prob,
        },
    )


def build_schapiro_graph() -> SREnvironment:
    """Build the 15-node 3-community graph from Schapiro et al. (2013/2016), Paper Fig. 4."""
    n = 15
    communities = [i // 5 for i in range(n)]  # 3 communities of 5 nodes each: 0..4, 5..9, 10..14
    A = np.zeros((n, n), dtype=float)

    # Within each 5-node community: all pairs connected EXCEPT the two boundary nodes (0 and 4)
    for c_idx in range(3):
        base = c_idx * 5
        for u in range(5):
            for v in range(u + 1, 5):
                if (u, v) == (0, 4):
                    continue  # bottleneck nodes don't connect directly to each other inside cluster
                A[base + u, base + v] = 1.0
                A[base + v, base + u] = 1.0

    # Bridges between communities: (4 <-> 5), (9 <-> 10), (14 <-> 0)
    bridges = [(4, 5), (9, 10), (14, 0)]
    for u, v in bridges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    T = A / A.sum(axis=1, keepdims=True)

    # Ring-of-clusters 2D layout for clean plotting
    coords = np.zeros((n, 2), dtype=float)
    cluster_angles = [np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]
    for c_idx, theta_c in enumerate(cluster_angles):
        cx, cy = 2.6 * np.cos(theta_c), 2.6 * np.sin(theta_c)
        base = c_idx * 5
        local_angles = np.linspace(theta_c - 0.9, theta_c + 0.9, 5)
        for u in range(5):
            r_loc = 1.0 if u in (1, 2, 3) else 1.15
            coords[base + u] = [cx + r_loc * np.cos(local_angles[u]), cy + r_loc * np.sin(local_angles[u])]

    cells = [(0, i) for i in range(n)]
    idx = {cell: i for i, cell in enumerate(cells)}
    return SREnvironment(
        kind="graph",
        height=1,
        width=n,
        cells=cells,
        cell_to_idx=idx,
        walls=set(),
        A=A,
        T=T,
        coords=coords,
        communities=communities,
        metadata={"bridges": bridges, "n_communities": 3},
    )


def build_energy_landscape_env(
    height: int = 15,
    width: int = 21,
    landscape_type: str = "double_well",
    kT: float = 0.6,
    barrier_height: float = 3.5,
    drift_rc: Tuple[float, float] = (0.0, 0.0),
) -> SREnvironment:
    """Build a 2D Statistical Physics Potential Energy Surface E(r, c) with Boltzmann transitions.

    Under Metropolis-Hastings / Arrhenius kinetics on a lattice, the transition weight from
    state s=(r,c) to neighbor s'=(r',c') is:
        w(s -> s') = exp( - max(0, E(s') - E(s) - F . (s' - s)) / kT )
    Once normalized to a row-stochastic matrix T, the Successor Representation M = (I - gamma*T)^-1
    is the exact Lattice Green's Function / Resolvent of the transfer operator!
    """
    y = np.linspace(-1.5, 1.5, height)
    x = np.linspace(-2.0, 2.0, width)
    X, Y = np.meshgrid(x, y)

    ltype = (landscape_type or "double_well").lower().strip()
    if ltype == "double_well":
        # Two deep metastable basins at (x = -1.1, y = 0) and (x = +1.1, y = 0)
        # separated by an energy ridge at x = 0 with a saddle point (mountain pass) at (x = 0, y = 0)
        E = 1.2 * (X**2 - 1.2) ** 2 + 1.6 * (Y**2) + barrier_height * np.exp(-4.0 * X**2) * (0.35 + 1.8 * Y**2)
        E -= E.min()

    elif ltype in ("advection_obstacle", "drift_diffusion"):
        # Repulsive central circular energy obstacle in a flat potential channel
        E = barrier_height * np.exp(-3.5 * ((X + 0.2) ** 2 + Y**2))
        E -= E.min()

    elif ltype in ("crystal_grain_boundary", "crystal_sinks"):
        # Periodic atomic lattice modulation + a high-energy grain boundary with two low-energy channels
        periodic = 0.35 * (np.cos(3 * np.pi * X) ** 2 + np.cos(3 * np.pi * Y) ** 2)
        ridge = barrier_height * np.exp(-8.0 * X**2)
        # Two vacancy migration channels through the grain boundary at y = -0.7 and y = +0.7
        passages = 1.0 - 0.78 * (np.exp(-6.0 * (Y - 0.75) ** 2) + np.exp(-6.0 * (Y + 0.75) ** 2))
        E = periodic + ridge * passages
        E -= E.min()
    else:
        E = 0.5 * (X**2 + Y**2)

    cells = [(r, c) for r in range(height) for c in range(width)]
    idx = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    A = np.zeros((n, n), dtype=float)
    W_mat = np.zeros((n, n), dtype=float)
    coords = np.zeros((n, 2), dtype=float)

    fr, fc = float(drift_rc[0]), float(drift_rc[1])
    kT_safe = max(0.05, float(kT))

    for (r, c), i in idx.items():
        coords[i] = [float(c), float(r)]
        e_curr = E[r, c]
        # Self-transition weight (thermal residence in current microstate)
        W_mat[i, i] = 0.25
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                A[i, j] = 1.0
                e_next = E[r + dr, c + dc]
                work = fr * dr + fc * dc
                delta_e = (e_next - e_curr) - work
                # Metropolis-Arrhenius rate
                rate = np.exp(-max(0.0, delta_e) / kT_safe)
                W_mat[i, j] = max(1e-7, rate)

    T = W_mat / W_mat.sum(axis=1, keepdims=True)
    return SREnvironment(
        kind="energy_2d",
        height=height,
        width=width,
        cells=cells,
        cell_to_idx=idx,
        walls=set(),
        A=A,
        T=T,
        coords=coords,
        energy_grid=E,
        kT=kT_safe,
        drift=(fr, fc),
        metadata={
            "landscape_type": ltype,
            "kT": kT_safe,
            "barrier_height": barrier_height,
            "drift_rc": drift_rc,
        },
    )


def compute_softmax_policy_T(
    env: SREnvironment,
    reward_vec: np.ndarray,
    gamma: float = 0.95,
    beta: float = 5.0,
    n_iter: int = 250,
) -> np.ndarray:
    """Compute optimal policy via Value Iteration + Softmax (Paper Online Methods, Eq. 5)."""
    n = env.n_states
    V = np.zeros(n, dtype=float)
    for _ in range(n_iter):
        # Q(s, s') = R(s') + gamma * V(s') for valid neighbors A(s, s') == 1
        Q_next = reward_vec[None, :] + gamma * V[None, :]
        Q_masked = np.where(env.A > 0, Q_next, -1e9)
        V_new = np.max(Q_masked, axis=1)
        if np.max(np.abs(V_new - V)) < 1e-8:
            V = V_new
            break
        V = V_new

    Q_final = reward_vec[None, :] + gamma * V[None, :]
    T_opt = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = np.where(env.A[i] > 0)[0]
        if len(nbrs) == 0:
            T_opt[i, i] = 1.0
        else:
            q_vals = beta * (Q_final[i, nbrs] - np.max(Q_final[i, nbrs]))
            probs = np.exp(q_vals)
            T_opt[i, nbrs] = probs / probs.sum()
    return T_opt
