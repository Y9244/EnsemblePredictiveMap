import matplotlib.pyplot as plt
import numpy as np
from open_arena import Agent, Sparse_Coding


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 64
    dt = 1.0
    N_h = 144

    agent_params = {
        "init_pos": [0.5, 0.5],
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt,
        "policy": False
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

    agent = Agent(agent_params)
    sparse_coding = Sparse_Coding(sparse_coding_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt

        r = agent.one_step_random(nt, T)
        #agent.one_step_show(ax)
        pre_s_h, s_h = sparse_coding.one_step(r)

        if nt % 1000 == 0:
            nsc_place_field = sparse_coding.calc_place_field_light()
            place_center, place_center_index = sparse_coding.calc_place_center(nsc_place_field, return_index=True)

            sparse_coding.show_place_field(nsc_place_field, place_center_index, check=True)
            sparse_coding.show_place_center(place_center, place_center_index)