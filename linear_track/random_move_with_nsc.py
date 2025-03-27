from linear_track import Agent, Sparse_Coding, Pmap
import seaborn as sns
sns.set()

if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 128
    dt = 0.8
    N_h = 36

    agent_params = {
        "init_pos": 0.0,
        "max_velocity": 0.1,
        "vec_size": vec_size,
        "map_size": map_size,
        "T": T,
        "dt": dt
    }
    sparse_coding_params = {
        "N_e": 600,
        "N_h": N_h,
        "map_size": map_size,
        "vec_size": vec_size,
        "on_reward": False,
        "dt": dt,
        "grid_files": "../bin/grid1d_{}_{}.npy".format(vec_size, 0.28)
    }
    pmap_params = {
        "vec_size": vec_size,
        "map_size": map_size,
        "N": N_h,
        "eta": 0.02,
        "gamma": 0.6
    }

    agent = Agent(agent_params)
    sparse_coding = Sparse_Coding(sparse_coding_params)
    pmap = Pmap(pmap_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt
        r = agent.one_step(nt, T)
        pre_s_h, s_h = sparse_coding.one_step(r)
        pmap.learning_M(pre_s_h, s_h)
        if nt % 10000 == 0:
            nsc_place_field = sparse_coding.calc_place_field_light()
            sparse_coding.show_place_field(nsc_place_field, display="multi")
            pmap_place_field = pmap.calc_place_field(nsc_place_field)
            pmap.show_place_field(pmap_place_field)
            pmap_center = pmap.calc_place_center(pmap_place_field)
            pmap_density = pmap.calc_place_density(pmap_center, periodic=False)
            pmap.show_place_density(pmap_density, pmap_center)
