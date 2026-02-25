from config.default_config import SquareArenaConfig, RLConfig, GridCellConfig, SparseCodingConfig, PmapConfig
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap
import numpy as np


if __name__ == "__main__":
    T = int(1_000_000)
    elapsed_time = 1000
    N_h = 100
    reward_info = [
        ((0.5, 0.5), (0.2, 0.2), 10.0) # (center, range, value)
    ]

    sa_cfg = SquareArenaConfig(
        reward_enabled=True,
        reward_info=reward_info
    )
    env = SquareArenaEnv(sa_cfg)

    rl_cfg = RLConfig(dim=2, actor_mode='MLP', eta_R=0.01)
    agent = RL(rl_cfg)

    gc_cfg = GridCellConfig(dim=2)
    grid_cell = GridCell(gc_cfg)

    sc_cfg = SparseCodingConfig(N_hip=N_h)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell)

    action = agent.act_from_pmap(np.random.random(N_h)) # the first step is random
    for t in range(1, T):
        position, reward = env.step(action)
        if reward > 0.0:  env.reset_pos()

        s_e = grid_cell.calc_grid_activity(position)
        pre_s_h, s_h = sparse_coding.one_step(s_e)

        pre_p_h, p_h = pmap.one_step_from_sc(pre_s_h, s_h)

        R = agent.learning_reward(s_h, reward)
        agent.learning_policy(reward, pre_p_h, p_h, t=t)
        action = agent.act_from_pmap(p_h)
        if t % elapsed_time == 0:
            sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
            pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
            place_centers, place_center_index = visualize_map.compute_place_centers_in_square_arena(sc_place_fields, return_index=True)

            Vmap = visualize_map.compute_Vmap(pmap_place_fields, R)
            policy_map = visualize_map.compute_policy_map(agent, pmap_place_fields)
            """
            visualize_map.show_place_fields_in_square_arena(sc_place_fields)
            visualize_map.show_place_fields_in_square_arena(pmap_place_fields)
            """
            #visualize_map.show_reward_vector_in_square_arena(place_centers, place_center_index, R)
            #visualize_map.show_value_map_in_square_arena(pmap_place_fields, R, place_centers, place_center_index)
            visualize_map.show_policy_map_on_Vmap_in_square_arena(policy_map, Vmap)
