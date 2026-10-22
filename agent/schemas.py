"""Pydantic Schemas for SR-Lab Agent Tool Calls."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class BuildGridEnvArgs(BaseModel):
    height: int = Field(12, ge=4, le=35, description="Grid height (rows)")
    width: int = Field(12, ge=4, le=35, description="Grid width (columns)")
    wall_preset: str = Field(
        "middle_wall_bottom_gap",
        description="Wall layout preset: 'open', 'middle_wall_bottom_gap', 'middle_wall_center_door', 'four_rooms', 'hairpin', 'tolman_detour'",
    )
    shape: str = Field("rectangle", description="Boundary shape: 'rectangle', 'circle', 'trapezoid'")
    lattice: str = Field("square", description="Lattice connectivity: 'square' (4-neighbor) or 'triangular' (6-neighbor)")
    gamma: float = Field(0.95, ge=0.1, le=0.995, description="Discount factor gamma in (0, 1)")
    bias_direction: str = Field("none", description="Optional directional bias: 'none', 'east', 'west', 'north', 'south'")
    bias_strength: float = Field(1.0, ge=1.0, le=10.0, description="Directional transition multiplier (>= 1.0)")
    target_row: Optional[int] = Field(None, description="Optional row of cell to inspect")
    target_col: Optional[int] = Field(None, description="Optional column of cell to inspect")


class BuildEnergyLandscapeArgs(BaseModel):
    height: int = Field(15, ge=7, le=31, description="Grid height")
    width: int = Field(21, ge=7, le=35, description="Grid width")
    landscape_type: str = Field(
        "double_well",
        description="PES type: 'double_well', 'advection_obstacle', 'crystal_grain_boundary'",
    )
    kT: float = Field(0.55, ge=0.1, le=5.0, description="Thermal energy k_B * T")
    barrier_height: float = Field(3.6, ge=0.5, le=10.0, description="Energy barrier height")
    drift_r: float = Field(0.0, ge=-2.0, le=2.0, description="Vertical external force F_r")
    drift_c: float = Field(0.0, ge=-2.0, le=2.0, description="Horizontal external force F_c")
    gamma: float = Field(0.96, ge=0.5, le=0.995, description="Resolvent discount factor gamma")


class InspectPlaceFieldArgs(BaseModel):
    row: int = Field(3, description="Row index of encoded target state s'")
    col: int = Field(5, description="Column index of encoded target state s'")
    domain: str = Field("neuro", description="'neuro' (Place Field) or 'physics' (Lattice Green's Function)")


class TrackSkewArgs(BaseModel):
    length: int = Field(41, ge=11, le=101, description="Number of states on 1D track")
    target_idx: int = Field(25, ge=2, le=98, description="Index of recorded place cell / state s*")
    p_right: float = Field(0.90, ge=0.05, le=0.99, description="Probability of stepping rightward")
    gamma: float = Field(0.90, ge=0.1, le=0.99, description="Discount factor gamma")
    domain: str = Field("neuro", description="'neuro' or 'physics'")


class RewardRevaluationArgs(BaseModel):
    start_row: int = Field(10, description="Agent starting row")
    start_col: int = Field(2, description="Agent starting col")
    goal_a_row: int = Field(1, description="Initial goal/sink A row")
    goal_a_col: int = Field(10, description="Initial goal/sink A col")
    goal_b_row: int = Field(1, description="New revalued goal/sink B row")
    goal_b_col: int = Field(1, description="New revalued goal/sink B col")
    domain: str = Field("neuro", description="'neuro' or 'physics'")


class GridAndSubgoalsArgs(BaseModel):
    top_n_subgoals: int = Field(2, ge=1, le=6, description="Number of bottleneck subgoals / saddle points to extract")
    mode_indices: List[int] = Field(default_factory=lambda: [2, 4, 7, 12, 18], description="5 eigenvector indices to plot")
    domain: str = Field("neuro", description="'neuro' or 'physics'")


class TdVsAnalyticArgs(BaseModel):
    n_steps: int = Field(8000, ge=500, le=50000, description="Number of online TD(0) transition steps")
    eta: float = Field(0.1, ge=0.01, le=0.5, description="TD learning rate eta")
    k_keep: int = Field(18, ge=2, le=64, description="Number of low-frequency spectral modes for smoothing")


class RunPresetArgs(BaseModel):
    preset_id: str = Field(
        ...,
        description="Preset ID: 'P1_wall_warp', 'P2_track_skew', 'P3_reward_reval', 'P4_two_rooms_subgoal', 'P5_geometry_grids', 'P6_tolman_detour', 'P7_schapiro_graph', 'P8_double_well_saddle', 'P9_advection_diffusion_drift', 'P10_crystal_vacancy_sinks'",
    )
