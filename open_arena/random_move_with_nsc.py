import numpy as np
from open_arena import Agent, Sparse_Coding, Pmap


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 64
    dt = 1.0
    N_h = 100

    agent_params = {
        "init_pos": [0.5, 0.5],
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt
    }
    sparse_coding_params = {
        "N_e": 600,
        "N_h": N_h,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt,
        "grid_files": "../bin/grid2d_{}_{}.npy".format(vec_size, 0.28)
    }
    pmap_params = {
        "N": N_h,
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

        r = agent.one_step(nt, T)

        pre_s_h, s_h = sparse_coding.one_step(r)

        pmap.learning_M(pre_s_h, s_h)

        if nt % 10000 == 0:
            nsc_place_field = sparse_coding.calc_place_field_light()
            sparse_coding.show_place_field(nsc_place_field)
            pmap_place_field = pmap.calc_place_field(nsc_place_field)
            pmap.show_place_field(pmap_place_field)


            """place_cell = np.zeros((N_h, vec_size**2))
            for i in range(vec_size**2):
                r = np.zeros(vec_size**2)
                r[i] = 1.0
                pre_s_h, s_h = sparse_coding.sparse_coding(r, sparse_coding.u_h, sparse_coding.s_h, learning=False)
                place_cell[:, i] = s_h / np.linalg.norm(s_h)"""
            """place_cell_sum = place_cell.sum(axis=0)
            n = int(N_h**0.5)
            X, Y = np.linspace(0, map_size, vec_size), np.linspace(0, map_size, vec_size)
            X, Y = np.meshgrid(X, Y)
            fig, ax = plt.subplots(n, n, figsize=(8, 8))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            for i in range(n):
                for j in range(n):
                    ax[i, j].contourf(X, Y, place_cell[n * i + j].reshape(vec_size, vec_size, order='C'), cmap="jet")
                    ax[i, j].text(0.5, 0.5, "{:.3f}".format(np.max(place_cell[n * i + j])), va='center', ha='center', color='white')
                    ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
            plt.show()
            plt.close()"""

            """place_field_recovered = sparse_coding.calc_place_field()
            sparse_coding.imshow_place_field(
                sparse_coding.place_field_recovered,
                'sparse_coding_2d_{}'.format(N_h)
            )
            sparse_coding.show_place_center(place_field_recovered)"""
            #sparse_coding.show_place_field(place_cell.T)
            #sparse_coding.show_place_center(place_cell.T)
            #pmap.place_field(place_cell.T)
