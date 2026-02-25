import numpy as np
from open_arena import Agent, Sparse_Coding, Pmap, save_variable


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 64
    dt = 1.0
    N_h = 100
    reward_x = [0.6, 0.8]
    reward_y = [0.6, 0.8]

    agent_params = {
        "init_pos": [0.5, 0.5],
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt,
        "policy": False,
        "on_reward": True,
        "reward_x": reward_x,
        "reward_y": reward_y
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
        "vec_size": vec_size,
        "map_size": map_size,
        "eta": 0.05,
        "gamma": 0.9,
        "on_reward": True,
        "reward_x": reward_x,
        "reward_y": reward_y
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

        pmap.learning_reward(s_h, agent.reward)

        if nt % 5000 == 0:
            nsc_place_field = sparse_coding.calc_place_field_light()
            sparse_coding.show_place_field(nsc_place_field, check=False)

            pmap_place_field = pmap.calc_place_field(nsc_place_field)
            save_variable(pmap_place_field, "pmap_place_field.pkl")
            pmap.show_place_field(pmap_place_field)

            pmap_place_center = pmap.calc_place_center(pmap_place_field, return_index=False)
            pmap.show_reward(pmap_place_center)

            V_map = pmap.calc_value_map(pmap_place_field)
            pmap.show_value_map(V_map)