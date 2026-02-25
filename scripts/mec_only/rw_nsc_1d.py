from config.default_config import LinearTrackConfig, RLConfig, GridCellConfig, SparseCodingConfig
from envs.linear_track import LinearTrackEnv
from models.RL_module import RL
from models.grid_cells import GridCell
from models.sparse_coding import SparseCoding
from visualizer.visualize_map import VisualizeMap


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

    sc_cfg = SparseCodingConfig(N_hip=N_h)
    sparse_coding = SparseCoding(sc_cfg)

    visualize_map = VisualizeMap(env=env, grid_cell=grid_cell, resolution=128)

    action = agent.act_random_1d()
    for t in range(1, T):
        print(t)
        position, reward = env.step(action)

        agent.observe(position, reward)

        s_e = grid_cell.calc_grid_activity(position)
        pre_s_h, s_h = sparse_coding.one_step(s_e)

        action = agent.act_random_1d()
        if t % elapsed_time == 0:
            sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding) # shape: [N_pos, N_place]
            visualize_map.show_place_fields_in_linear_track(sc_place_fields, name='rw_nsc_1d')
