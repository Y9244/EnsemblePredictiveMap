import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

rng = np.random.default_rng()

vec_size = 64

N_lam = 4
N_theta = 6
N_x = 5
N_y = 5
N_e = N_lam * N_theta * N_x * N_y
min_lam = 0.28

grid_spacing = [min_lam * 1.42**(i-0) for i in range(N_lam)]
print(grid_spacing)
grid_orientation = np.pi/3 * rng.random(N_theta)
grid_orientation = [(i/N_theta) * np.pi/3 for i in range(N_theta)]
grid_phase_x = [i/N_x for i in range(N_x)]
grid_phase_y = [i/N_y for i in range(N_y)]

u = np.array([[np.cos(2*np.pi*0 + 0), np.sin(2*np.pi*0 + 0)],
              [np.cos(2*np.pi*1 + 0), np.sin(2*np.pi*1 + 0)],
              [np.cos(2*np.pi*2 + 0), np.sin(2*np.pi*2 + 0)]])

X, Y = np.linspace(0, 1, vec_size), np.linspace(0, 1, vec_size)
X, Y = np.meshgrid(X, Y)
Z = np.zeros((N_e, vec_size, vec_size))

i = 0
for lam in grid_spacing:
    for theta in grid_orientation:
        for x0 in grid_phase_x:
            x0 *= lam
            for y0 in grid_phase_y:
                y0 *= lam
                for j in range(1, 4):
                    uj = ((4*np.pi)/(3**0.5 * lam)) * np.array([np.cos(2*np.pi*j/3 + theta), np.sin(2*np.pi*j/3 + theta)])
                    tmp_Z_j = np.cos(uj[0] * (X-x0) - uj[1] * (Y-y0))
                    Z[i] += tmp_Z_j
                Z[i] = (2/3) * (Z[i]/3 + 0.25)
                i += 1

Z = np.where(Z > 1, 1, Z)
Z = np.where(Z < 0, 0, Z)
np.save("../bin/grid2d_{}_{}.npy".format(vec_size, min_lam), Z)

n = 10
for i in range(24):
    fig, ax = plt.subplots(n, n, figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    index = np.random.randint(0, 600, 100)
    for j in range(n):
        for k in range(n):
            # print(index[j * n + k])
            ax[j, k].contourf(X, Y, Z[index[j * n + k]], cmap="jet", levels=100)
            ax[j, k].tick_params(bottom=False, left=False, right=False, top=False)
            ax[j, k].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    #plt.savefig("grid{}.png".format(i), dpi=500)
    plt.show()
    plt.close()



