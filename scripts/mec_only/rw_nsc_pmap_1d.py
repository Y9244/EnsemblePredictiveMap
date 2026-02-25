from config.default_config import LinearTrackConfig, RLConfig, GridCellConfig, SparseCodingConfig, PmapConfig
from envs.linear_track import LinearTrackEnv
from models.grid_cells import GridCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap
import numpy as np
from models.wrapper import one_step_grid

if __name__ == "__main__":
    T = int(1_000_000)
    elapsed_time = 1000
    N_h = 36

    lt_cfg = LinearTrackConfig(reward_enabled=False)
    env = LinearTrackEnv(lt_cfg)

    agent_cfg = RLConfig(dim=1, actor_mode='random', max_step=0.05)
    agent = RL(agent_cfg)

    gc_cfg = GridCellConfig(dim=1)
    grid_cell = GridCell(gc_cfg)
    env.set_grid_cell(grid_cell)

    sc_cfg = SparseCodingConfig(N_hip=N_h)
    sparse_coding = SparseCoding(sc_cfg)

    pmap_cfg = PmapConfig(N_hip=N_h)
    pmap = Pmap(pmap_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, resolution=128)

    action = 0
    s_e, reward_vector, s_h, p_h = one_step_grid(action, env, sparse_coding, pmap)
    for t in range(1, T):
        print(t)
        action = agent.act_random_1d()

        next_s_e, reward_vector, next_s_h, next_p_h = one_step_grid(action, env, sparse_coding, pmap)

        pmap.predictive_map_learning(s_h, p_h, next_p_h)

        s_e, s_h, p_h = next_s_e, next_s_h, next_p_h
        if t % elapsed_time == 0:
            sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
            pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
            visualize_map.show_place_fields_in_linear_track(sc_place_fields, name='rw_sc_1d')
            visualize_map.show_place_fields_in_linear_track(pmap_place_fields, name='rw_pmap_1d')
