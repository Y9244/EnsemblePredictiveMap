import numpy as np
import joblib

from config.default_config import LinearTrackConfig, RLConfig, GridCellConfig, RewardCellConfig, SparseCodingConfig, PmapConfig
from envs.linear_track import LinearTrackEnv
from models.grid_cells import GridCell
from models.reward_cell import RewardCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap
from models.wrapper import one_step_EC

if __name__ == "__main__":
    T = int(500_000)
    elapsed_episode = 1000
    N_h = 169
    N_reward_cell = 400

    env_cfg = LinearTrackConfig(reward_enabled=True)
    env = LinearTrackEnv(env_cfg)

    gc_cfg = GridCellConfig(dim=1)
    grid_cell = GridCell(gc_cfg)
    env.set_grid_cell(grid_cell)

    reward_cell_config = RewardCellConfig(
        dim=1,
        reward_positions=env.reward_position,
        N_reward_cell=N_reward_cell,
    )
    reward_cell = RewardCell(reward_cell_config)
    env.set_reward_cell(reward_cell)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, reward_cell=reward_cell, resolution=512)

    ac_lec_1d = 'ac_lec_1d'
    r_pos_A = 3.66
    sc_place_fields_366 = joblib.load(f'../data/{ac_lec_1d}/sc_place_fields_{r_pos_A}.pkl')
    pmap_place_fields_366 = joblib.load(f'../data/{ac_lec_1d}/pmap_place_fields_{r_pos_A}.pkl')
    agent_366 = joblib.load(f'../data/{ac_lec_1d}/agent_{r_pos_A}.pkl')

    r_pos_B = 1.66
    sc_place_fields_166 = joblib.load(f'../data/{ac_lec_1d}/sc_place_fields_{r_pos_B}.pkl')
    pmap_place_fields_166 = joblib.load(f'../data/{ac_lec_1d}/pmap_place_fields_{r_pos_B}.pkl')
    agent_166 = joblib.load(f'../data/{ac_lec_1d}/agent_{r_pos_B}.pkl')


    place_centers_366, place_center_index_366 = visualize_map.compute_place_centers_in_linear_track(
        sc_place_fields_366, return_index=True)
    place_centers_166, place_center_index_166 = visualize_map.compute_place_centers_in_linear_track(
        sc_place_fields_166, return_index=True)
    policy_map_366 = visualize_map.compute_policy_map_in_linear_track(agent_366, sc_place_fields_366)
    policy_map_166 = visualize_map.compute_policy_map_in_linear_track(agent_166, sc_place_fields_166)

    visualize_map.show_policy_map_in_linear_track_2context(policy_map_366, policy_map_166,
                                                           [r_pos_A, 0.1], [r_pos_B, 0.1],
                                                           base_dir='Frontiers', name="Figure8B")

    visualize_map.show_place_density_in_linear_track(place_centers_366, place_center_index_366,
                                                     r_pos=[r_pos_A, 0.1], base_dir='Frontiers', name="Figure8C")
    visualize_map.show_place_density_in_linear_track(place_centers_166, place_center_index_166,
                                                     r_pos=[r_pos_B, 0.1], base_dir='Frontiers', name="Figure8D")

    visualize_map.show_place_center_scatter_2context(
        place_centers_366, place_center_index_366,
        place_centers_166, place_center_index_166,
        [r_pos_A, 0.1], [r_pos_B, 0.1],
        base_dir='Frontiers', name='Figure8E'
    )



