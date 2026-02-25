import numpy as np

from config.default_config import (
    SquareArenaConfig, RLConfig, GridCellConfig, BoundaryCellConfig,
    SparseCodingConfig, PmapConfig
)
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.boundary_cell import BoundaryCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from models.wrapper import one_step_grid_border
from visualizer.visualize_map import VisualizeMap
import pickle

if __name__ == "__main__":
    T = int(1_000_000)
    elapsed_time = 1000
    N_h = 169

    sa_cfg = SquareArenaConfig(reward_enabled=False)
    env = SquareArenaEnv(sa_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='random', N_hip=N_h)
    agent = RL(rl_cfg)

    gc_cfg = GridCellConfig(dim=2)
    grid_cell = GridCell(gc_cfg)
    env.set_grid_cell(grid_cell)

    bc_cfg = BoundaryCellConfig()
    boundary_cell = BoundaryCell(bc_cfg)
    env.set_boundary_cell(boundary_cell)

    N_mec = grid_cell.N_e + boundary_cell.N_cells

    object_position = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]

    sc_cfg = SparseCodingConfig(N_hip=N_h, N_mec=N_mec)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, boundary_cell=boundary_cell, resolution=128)

    action = np.array([0, 0])
    s_e, reward_vector, s_h, p_h = one_step_grid_border(action, env, sparse_coding, pmap)
    for t in range(1, T):
        print(t)
        action = agent.act_random_2d(mode='random')

        next_s_e, reward_vector, next_s_h, next_p_h = one_step_grid_border(action, env, sparse_coding, pmap)

        pmap.predictive_map_learning(s_h, p_h, next_p_h)

        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h

        if t % elapsed_time == 0:
            figure7 = {}
            figure7['sparse_coding'] = sparse_coding
            figure7['pmap'] = pmap
            figure7['agent'] = agent
            figure7['visualize_map'] = visualize_map

            with open('../../data/Frontiers/figure7_mec.pkl', 'wb') as f:
                pickle.dump(figure7, f)