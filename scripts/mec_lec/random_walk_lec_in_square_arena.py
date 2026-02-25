import numpy as np

from config.default_config import SquareArenaConfig, RLConfig, GridCellConfig, RewardCellConfig, SparseCodingConfig, PmapConfig
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.reward_cell import RewardCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap


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

    reward_position = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    reward_cell_config = RewardCellConfig(
        dim=2,
        reward_positions=reward_position
    )
    reward_cell = RewardCell(reward_cell_config)

    sc_cfg = SparseCodingConfig(N_hip=N_h, N_lec=reward_cell.N_reward)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, reward_cell=reward_cell, resolution=128)

    action = agent.act_random_2d()
    for t in range(1, T):
        print(t)
        position, reward = env.step(action)

        s_e = grid_cell.calc_grid_activity(position)
        s_l = reward_cell.calc_reward_cell_activity_in_square_arena(position)
        s_e = np.concatenate([s_e, s_l[None, :]], axis=1)
        pre_s_h, s_h = sparse_coding.one_step(s_e)

        pmap.one_step_from_sc(pre_s_h, s_h)

        action = agent.act_random_2d()
        if t % elapsed_time == 0:
            sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
            pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
            place_centers, place_center_index = visualize_map.compute_place_centers_in_square_arena(sc_place_fields, return_index=True)

            visualize_map.show_place_fields_in_square_arena(sc_place_fields, name='rw_LEC_2d_nsc')
            visualize_map.show_place_fields_in_square_arena(pmap_place_fields, name='rw_LEC_2d_pmap')

            visualize_map.show_place_density_in_square_arena(place_centers, place_center_index, reward_position, name='random_walk_LEC_2d_place_density')
            #visualize_map.show_place_centers_in_square_arena(place_centers, place_center_index, reward_position, name='random_walk_LEC_2d_place_center')
            visualize_map.show_place_center_distance(place_centers, place_center_index, name='random_walk_LEC_2d_place_center_distance')

