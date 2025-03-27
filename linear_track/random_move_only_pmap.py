import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy
import seaborn as sns
from linear_track import Agent, Pmap

if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 128
    dt = 1.0
    N_h = 25

    agent_params = {
        "init_pos": 0.0,
        "max_velocity": 0.1,
        "vec_size": vec_size,
        "map_size": map_size,
        "T": T,
        "dt": dt
    }
    pmap_params = {
        "vec_size": vec_size,
        "map_size": map_size,
        "N": N_h,
        "eta": 0.01,
        "gamma": 0.8
    }

    agent = Agent(agent_params)
    pmap = Pmap(pmap_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt
        agent.one_step(nt, T)
        pmap.learning_M(agent.pre_pos, agent.pos, sparse_coding=False)
        if nt % 10000 == 0:
            pmap.show_M()
            pmap_place_field = pmap.calc_place_field_no_nsc()
            pmap.show_place_field(pmap_place_field)
            pmap_center = pmap.calc_place_center(pmap_place_field)
            pmap_density = pmap.calc_place_density(pmap_center, periodic=True)
            pmap.show_place_density(pmap_density, pmap_center)

