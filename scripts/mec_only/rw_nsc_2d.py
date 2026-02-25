from config.default_config import SquareArenaConfig, RLConfig, GridCellConfig, SparseCodingConfig
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.sparse_coding import SparseCoding
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap
import numpy as np
from models.wrapper import one_step_grid

if __name__ == "__main__":
    T = int(1_000_000)
    elapsed_time = 1000

    sq_cfg = SquareArenaConfig(reward_enabled=False)
    env = SquareArenaEnv(sq_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='random')
    agent = RL(rl_cfg)

    gc_cfg = GridCellConfig(dim=2)
    grid_cell = GridCell(gc_cfg)

    sc_cfg = SparseCodingConfig()
    sparse_coding = SparseCoding(sc_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell)

    action = np.zeros(2)
    for t in range(1, T):
        print(t)
        position, reward = env.step(action)


        s_e = grid_cell.calc_grid_activity(position)
        pre_s_h, s_h = sparse_coding.one_step(s_e)

        action = agent.act_random_2d()
        if t % elapsed_time == 0:
            sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding) # shape: [N_pos, N_place]
            visualize_map.show_place_fields_in_square_arena(sc_place_fields, name='rw_nsc_2d')
