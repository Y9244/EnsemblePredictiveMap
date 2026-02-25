import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from models.grid_cells import GridCell
from config.default_config import GridCellConfig
from tqdm import tqdm


grid_config = GridCellConfig()
gc = GridCell(grid_config)

resolution = 256
env_size = 1.0
x = np.linspace(0, env_size, resolution)
y = np.linspace(0, env_size, resolution)
xx, yy = np.meshgrid(x, y)
positions = np.stack([xx.ravel(), yy.ravel()], axis=1)

grid_fields = np.array([gc.calc_grid_activity(pos) for pos in tqdm(positions)], dtype=np.float32).T

N_cell = 10
rng = np.random.default_rng(seed=42)
i_cells = rng.integers(low=0, high=600, size=N_cell)

X, Y = np.linspace(0, env_size, resolution), np.linspace(0, env_size, resolution)
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(15, 3))
gs = gridspec.GridSpec(2, 10, left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
for index in range(2*N_cell):
    i_cell = i_cells[index%N_cell]
    i = index // 10
    j = index % 10
    print(i_cell)
    ax = fig.add_subplot(gs[i, j])
    if i == 0:
        im = ax.contourf(X, Y, grid_fields[i_cell].reshape(resolution, resolution), cmap="jet", levels=100)
    else:
        ax.plot(grid_fields[i_cell].reshape(resolution, resolution)[resolution//4])
    ax.tick_params(bottom=False, left=False, right=False, top=False,
                   labelbottom=False, labelleft=False, labelright=False, labeltop=False)
cbar_ax = fig.add_axes((0.86, 0.1, 0.02, 0.8))  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .2f'))
cbar.set_ticks([0.00, np.max(grid_fields)/2, np.max(grid_fields)])
cbar.ax.tick_params(labelsize=18)
print(np.max(grid_fields))
plt.savefig('../../figs/Frontiers/figure2B.png', dpi=300)


