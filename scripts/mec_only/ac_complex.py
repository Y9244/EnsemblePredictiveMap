import numpy as np
import joblib

from config.default_config import SquareArenaConfig, RLConfig, GridCellConfig, RewardCellConfig, SparseCodingConfig, PmapConfig
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.reward_cell import RewardCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap

def one_step(action, env, sparse_coding, pmap):
    s_e, reward_vector = env.step_grid(action)
    s_h = sparse_coding.one_step(s_e)
    p_h = pmap.one_step_from_sc(s_h)
    return s_e, reward_vector, s_h, p_h

def visualize_set(visualize_map, sparse_coding, pmap, agent, name):
    base_dir = "ac_complex"
    sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
    pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
    place_centers, place_center_index = visualize_map.compute_place_centers_in_square_arena(
        sc_place_fields, return_index=True)
    Vmap = visualize_map.compute_Vmap(pmap_place_fields, agent.R, agent.reward_weights)
    policy_map = visualize_map.compute_policy_map_in_square_arena(agent, sc_place_fields)

    visualize_map.show_place_fields_in_square_arena(
        sc_place_fields,
        base_dir=base_dir, name=f'sc_place_fields_{name}')
    visualize_map.show_place_fields_in_square_arena(
        pmap_place_fields,
        base_dir=base_dir, name=f'pmap_place_fields_{name}')
    visualize_map.show_place_density_in_square_arena(
        place_centers, place_center_index, env.reward_position,
        base_dir=base_dir, name=f"place_density_{name}")
    visualize_map.show_reward_vector_in_square_arena(
        place_centers, place_center_index, agent.R,
        base_dir=base_dir, name=f'reward_vector_{name}')
    visualize_map.show_policy_map_on_Vmap_in_square_arena(
        policy_map, Vmap,
        base_dir=base_dir, name=f"policy_map_{name}")

if __name__ == "__main__":
    T = int(500_000)
    elapsed_episode = 50
    N_h = 169

    dt, ct, reward_scale = 0.068, 0.5, 100.0
    #reward_info = [((dt * (i + 0.5), 0.5), (dt, 0.1), (-1)**i * reward_scale) for i in range(11)]
    reward_info = [((ct - dt, ct - dt), (dt, dt), +reward_scale),
                   ((ct     , ct - dt), (dt, dt), -reward_scale),
                   ((ct + dt, ct - dt), (dt, dt), +reward_scale),

                   ((ct - dt, ct), (dt, dt), -reward_scale),
                   ((ct     , ct), (dt, dt), -reward_scale),
                   ((ct + dt, ct), (dt, dt), -reward_scale),

                   ((ct - dt, ct + dt), (dt, dt), +reward_scale),
                   ((ct     , ct + dt), (dt, dt), -reward_scale),
                   ((ct + dt, ct + dt), (dt, dt), +reward_scale)
                   ]
    #reward_info += [((ct, 0.975), (1, 0.05), -reward_scale)]
    #reward_info += [((ct, 0.025), (1, 0.05), -reward_scale)]

    env_cfg = SquareArenaConfig(
        reward_enabled=True,
        reward_info=reward_info
    )
    env = SquareArenaEnv(env_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='MLP', N_hip=N_h, input_dim=N_h, eta_mlp=1e-4, eta_R=1e-5,
                      reward_num=env.reward_num, max_step=0.05)
    agent = RL(rl_cfg)

    grid_cell = GridCell(GridCellConfig(dim=2))
    env.set_grid_cell(grid_cell)

    sc_cfg = SparseCodingConfig(N_hip=N_h)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, resolution=64)

    action = np.zeros(2)
    s_e, _, s_h, p_h = one_step(action, env, sparse_coding, pmap)
    for t in range(1, T):
        action = agent.act_random_2d_torch(s_h)

        next_s_e, reward_vector, next_s_h, next_p_h = one_step(action, env, sparse_coding, pmap)

        pmap.predictive_map_learning(s_h, p_h, next_p_h)
        agent.learning_policy(reward_vector, s_h, next_s_h, p_h, next_p_h,
                              R_replay=True, save_step=True, episode_i=None, t=t)

        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h

        if t % 10000 == 0:
            visualize_set(visualize_map, sparse_coding, pmap, agent, "pretraining")
