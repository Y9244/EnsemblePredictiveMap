import numpy as np
from open_arena import Agent, Sparse_Coding, Pmap


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 64
    dt = 1.0
    N_h = 100
    on_reward = False

    agent_params = {
        "init_pos": [0.5, 0.5],
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "on_reward": on_reward,
        'policy': False,
        "dt": dt
    }
    sparse_coding_params = {
        "beta": 0.3,
        "N_e": 600,
        "N_h": N_h,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt,
        "grid_files": "../bin/grid2d_{}_{}.npy".format(vec_size, 0.28)
    }
    pmap_params = {
        "N_h": N_h,
        "on_reward": on_reward,
        "vec_size": vec_size,
        "map_size": map_size,
        "eta": 0.05,
        "gamma": 0.9
    }

    agent = Agent(agent_params)
    sparse_coding = Sparse_Coding(sparse_coding_params)
    pmap = Pmap(pmap_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt

        r = agent.one_step_random()

        pre_s_h, s_h = sparse_coding.one_step(r)

        pmap.learning_M(pre_s_h, s_h)

        if nt % 5000 == 0:
            nsc_place_field = sparse_coding.calc_place_field_light()
            sparse_coding.show_place_field(nsc_place_field)
            pmap_place_field = pmap.calc_place_field(nsc_place_field)
            pmap.show_place_field(pmap_place_field)