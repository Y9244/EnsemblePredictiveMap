import numpy as np

from config.default_config import (
    SquareArenaConfig, RLConfig, GridCellConfig,
    BoundaryCellConfig, SparseCodingConfig, PmapConfig
)
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.boundary_cell import BoundaryCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL

from visualizer.visualize_map import VisualizeMap
from models.wrapper import one_step_grid_border
from utils.logger import Logger
import pickle, copy

if __name__ == "__main__":
    T = int(1_000_000)
    elapsed_time = 1000
    N_h = 100

    sa_cfg = SquareArenaConfig(reward_enabled=True)
    env = SquareArenaEnv(sa_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='random', reward_num=env.reward_num)
    agent = RL(rl_cfg)

    gc_cfg = GridCellConfig(dim=2)
    grid_cell = GridCell(gc_cfg)
    env.set_grid_cell(grid_cell)

    bc_cfg = BoundaryCellConfig()
    boundary_cell = BoundaryCell(bc_cfg)
    env.set_boundary_cell(boundary_cell)

    N_mec = grid_cell.N_e + boundary_cell.N_cells

    sc_cfg = SparseCodingConfig(N_hip=N_h, N_mec=N_mec)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, boundary_cell=boundary_cell, resolution=128)
    logger = Logger()

    action = np.array([0, 0])
    s_e, reward_vector, s_h, p_h = one_step_grid_border(action, env, sparse_coding, pmap)
    for t in range(1, T):
        print(t)
        action = agent.act_random_2d(mode='upper')

        next_s_e, reward_vector, next_s_h, next_p_h = one_step_grid_border(action, env, sparse_coding, pmap)

        pmap.predictive_map_learning(s_h, p_h, next_p_h)
        agent.learning_reward(s_h, reward_vector)

        logger.append(copy.deepcopy(s_h), copy.deepcopy(p_h))
        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h

        if t % elapsed_time == 0:
            figure4 = {}
            figure4['sparse_coding'] = sparse_coding
            figure4['pmap'] = pmap
            figure4['agent'] = agent
            figure4['visualize_map'] = visualize_map
            figure4['logger'] = logger

            with open('../../data/Frontiers/figure4.pkl', 'wb') as f:
                pickle.dump(figure4, f)

