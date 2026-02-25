import numpy as np
import copy, pickle
from config.default_config import (
    SquareArenaConfig, RLConfig, GridCellConfig, BoundaryCellConfig, SparseCodingConfig, PmapConfig)
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.boundary_cell import BoundaryCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from models.wrapper import one_step_grid_border, one_step_grid
from visualizer.visualize_map import VisualizeMap


def visualize_set(visualize_map, sparse_coding, pmap, agent, name):
    base_dir = "ac_2d"
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
    if name != "pretraining":
        visualize_map.show_reward_vector_in_square_arena(
            place_centers, place_center_index, agent.R,
            base_dir=base_dir, name=f'reward_vector_{name}')
        visualize_map.show_policy_map_on_Vmap_in_square_arena(
            policy_map, Vmap,
            base_dir=base_dir, name=f"policy_map_{name}")

if __name__ == "__main__":
    T = int(20_000)
    elapsed_episode = 50
    N_h = 100

    env_cfg = SquareArenaConfig(reward_enabled=False)
    env = SquareArenaEnv(env_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='MLP', N_hip=N_h, input_dim=N_h, eta_mlp=1e-4, eta_R=1e-5,
                      reward_num=env.reward_num, max_step=0.08)
    agent = RL(rl_cfg)

    grid_cell = GridCell(GridCellConfig(dim=2))
    env.set_grid_cell(grid_cell)

    bc_cfg = BoundaryCellConfig()
    boundary_cell = BoundaryCell(bc_cfg)
    env.set_boundary_cell(boundary_cell)

    N_mec = grid_cell.N_e + boundary_cell.N_cells

    sc_cfg = SparseCodingConfig(N_hip=N_h, N_mec=N_mec)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, boundary_cell=boundary_cell, resolution=50)
    #visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, resolution=50)

    env.reset_reward_info()

    # ---------- 慣らしフェーズ ----------
    action = np.random.random(2)
    s_e, reward_vector, s_h, p_h = one_step_grid_border(action, env, sparse_coding, pmap)
    for t in range(1, T+1):
        print(f"Pretraining phase: t={t}")
        action = agent.act_random_2d(mode='random')
        next_s_e, reward_vector, next_s_h, next_p_h = one_step_grid_border(action, env, sparse_coding, pmap)
        pmap.predictive_map_learning(s_h, p_h, next_p_h)
        # agent.learning_policy(reward_vector, s_h, next_s_h, p_h, next_p_h, R_replay=True, ablation=False, t=t)
        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h
        if t % 5000 == 0:
            visualize_set(visualize_map, sparse_coding, pmap, agent, "pretraining")

    env.reward_enabled = True
    env.add_reward_info([((0.5, 0.5), (0.2, 0.2), 100.0)])

    episode_i = 0
    step_per_episode_list = []
    while True:
        t = 0
        env.reset_pos(mode='random')
        action = np.zeros(2)
        s_e, _, s_h, p_h = one_step_grid_border(action, env, sparse_coding, pmap)
        while True:
            action = agent.act_from_pmap(s_h)

            next_s_e, reward_vector, next_s_h, next_p_h = one_step_grid_border(action, env, sparse_coding, pmap)

            pmap.predictive_map_learning(s_h, p_h, next_p_h)
            agent.learning_policy(reward_vector, s_h, next_s_h, p_h, next_p_h,
                                  R_replay=True, ablation=False, episode_i=episode_i, t=t)

            s_e, s_h, p_h = next_s_e, next_s_h, next_p_h
            t += 1
            if np.sum(reward_vector) > 0:
                break

        episode_i += 1
        step_per_episode_list.append(t)
        if episode_i % 200 == 0:
            print('saving ...')
            visualize_set(visualize_map, sparse_coding, pmap, agent, name=f'episode_{episode_i}')

        if episode_i == 1000:
            break

    figure6_full = {}
    figure6_full['visualize_map'] = visualize_map
    figure6_full['sparse_coding'] = sparse_coding
    figure6_full['pmap'] = pmap
    figure6_full['agent'] = agent
    figure6_full['step_per_episode_list'] = step_per_episode_list

    with open("../../data/Frontiers/figure6_full.pkl", "wb") as f:
        pickle.dump(figure6_full, f)