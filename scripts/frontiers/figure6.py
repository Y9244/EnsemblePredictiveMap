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

def visualize_set(visualize_map, sparse_coding, pmap, agent, name):
    base_dir = "ac_lec_1d"
    sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
    pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
    place_centers, place_center_index = visualize_map.compute_place_centers_in_linear_track(
        sc_place_fields, return_index=True)
    Vmap = visualize_map.compute_Vmap(pmap_place_fields, agent.R, agent.reward_weights)
    policy_map = visualize_map.compute_policy_map_in_linear_track(agent, sc_place_fields)

    visualize_map.show_place_fields_in_linear_track(
        sc_place_fields,
        base_dir=base_dir, name=f'sc_place_fields_{name}')
    visualize_map.show_place_fields_in_linear_track(
        pmap_place_fields,
        base_dir=base_dir, name=f'pmap_place_fields_{name}')

    visualize_map.show_place_density_in_linear_track(
        place_centers, place_center_index,
        base_dir=base_dir, name=f"place_density_{name}")
    visualize_map.show_reward_vector_in_linear_track(
        place_centers, place_center_index, agent.R,
        base_dir=base_dir, name=f'reward_vector_{name}'
    )
    visualize_map.show_value_map_in_linear_track(
        pmap_place_fields, agent.R, place_centers, place_center_index,
        base_dir=base_dir, name=f"value_map_{name}")
    visualize_map.show_policy_map_in_linear_track(
        policy_map, place_centers, place_center_index,
        base_dir=base_dir, name=f'policy_map_{name}')

    joblib.dump(sc_place_fields, f'../../data/{base_dir}/sc_place_fields_{name}.pkl')
    joblib.dump(pmap_place_fields, f'../../data/{base_dir}/pmap_place_fields_{name}.pkl')
    joblib.dump(agent, f'../../data/{base_dir}/agent_{name}.pkl')


if __name__ == "__main__":
    T = int(500_000)
    elapsed_episode = 1000
    N_h = 169
    N_reward_cell = 400

    env_cfg = LinearTrackConfig(reward_enabled=True)
    # , reward_info=[(2.2, 0.1, 50.0)]
    env = LinearTrackEnv(env_cfg)

    rl_cfg = RLConfig(dim=1, actor_mode='MLP', N_hip=N_h, input_dim=N_h, eta_mlp=1e-4, eta_R=1e-3,
                      reward_num=env.reward_num, max_step=0.1)
    agent = RL(rl_cfg)

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

    sc_cfg = SparseCodingConfig(N_hip=N_h, N_lec=N_reward_cell, eta=1e-2)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, reward_cell=reward_cell, resolution=512)

    action = 0.0
    s_e, reward_vector, s_h, p_h = one_step_EC(action, env, sparse_coding, pmap)
    for t in range(1, T):
        action = agent.act_from_pmap(s_h)

        next_s_e, reward_vector, next_s_h, next_p_h = one_step_EC(action, env, sparse_coding, pmap)

        pmap.predictive_map_learning(s_h, p_h, next_p_h)
        agent.learning_policy(reward_vector, s_h, next_s_h, p_h, next_p_h,
                              R_replay=False, save_step=False, episode_i=None, t=t)

        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h

        if t % 1000 == 0:
            visualize_set(visualize_map, sparse_coding, pmap, agent, name=str(env.reward_info[0][0]))

        if t > 60000:
            env.reset_reward_info()
            r_pos = 1.66
            env.add_reward_info([(r_pos, 0.1, 50.0)])
            reward_cell.change_reward_cell(r_pos)
            env.set_reward_cell(reward_cell)
            visualize_map.update_reward_cell(reward_cell)

