import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from open_arena import Agent, Pmap


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 32
    dt = 1.0
    N_h = 100

    agent_params = {
        "init_pos": np.array([0.5, 0.5]),
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt,
        'policy': False,
        'on_reward': False,
        'periodic': False
    }
    pmap_params = {
        "N_h": N_h,
        "vec_size": vec_size,
        "map_size": map_size,
        "on_reward": False,
        "eta": 0.05,
        "gamma": 0.8
    }

    agent = Agent(agent_params)
    pmap = Pmap(pmap_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt
        agent.one_step_random(periodic=True)

        pmap.learning_M(agent.pre_pos, agent.pos, sparse_coding=False)
        if nt % 5000 == 0:
            pmap_place_field = pmap.calc_place_field_no_nsc()
            pmap.show_place_field(pmap_place_field)