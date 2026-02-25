from dataclasses import dataclass, field
from typing import Tuple, List, Optional

N_mec = 600
N_hip = 100
gamma = 0.9
length = 4.0
reward_enabled = True

@dataclass
class LinearTrackConfig:
    length: float = length
    reward_info: List[Tuple[float, float, float]] = field(
        # [((報酬の中心地), (報酬の範囲), 報酬の大きさ), ...]
        default_factory=lambda: [(3.66, 0.1, 50.0)]
    )
    reward_enabled: bool = reward_enabled

@dataclass
class SquareArenaConfig:
    arena_size: float = 1.0
    reward_info: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = field(
        # [((報酬の中心地), (報酬の範囲), 報酬の大きさ), ...]
        default_factory=lambda: [((0.5, 0.5), (0.2, 0.2), 100.0)]
    )
    reward_enabled: bool = reward_enabled

@dataclass
class RLConfig:
    dim: int = 2
    max_step: float = 0.1
    actor_mode: str = "MLP"
    gamma: float = gamma
    eta_mlp: float = 5e-4
    eta_R: float = 1e-3
    input_dim: int = N_hip
    N_hip: int = N_hip
    reward_enabled: bool = reward_enabled
    reward_num: int = 0

@dataclass
class GridCellConfig:
    dim: int = 2
    N_lam: int = 4
    N_theta: int = 6
    N_x: int = 5
    N_y: int = 5
    min_lam: float = 0.28

@dataclass
class BoundaryCellConfig:
    arena_size: float = 1.0
    N_cells: int = 300
    sigma_range: tuple = (0.08, 0.12)


@dataclass
class RewardCellConfig:
    dim: int = 2
    env_size: float = length
    reward_positions: Optional[List] = None
    N_reward_cell: int = 600
    reward_center_range: float = 0.07


@dataclass
class SparseCodingConfig:
    beta: float = 0.3
    tau: int = 10
    N_hip: int = N_hip
    N_mec: int = N_mec
    N_lec: int = 0
    n_iter: int = 200
    s_h_max: int = 10
    eta: float = 5e-3

@dataclass
class PmapConfig:
    N_hip: int = N_hip
    gamma: float = gamma
    eta_M: float = 0.05
