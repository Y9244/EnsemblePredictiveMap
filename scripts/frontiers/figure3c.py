import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from models.reward_cell import RewardCell
from config.default_config import RewardCellConfig
import seaborn as sns
sns.set()

reward_positions = [(0.25, 0.25),
                    (0.25, 0.75),
                    (0.75, 0.25),
                    (0.75, 0.75)]
resolution = 256
env_size = 1.0
x = np.linspace(0, env_size, resolution)
y = np.linspace(0, env_size, resolution)
X, Y = np.meshgrid(x, y)
positions = np.stack([X.ravel(), Y.ravel()], axis=1)

reward_cell_config = RewardCellConfig(
	dim=2,
	reward_positions=reward_positions
)
rc = RewardCell(reward_cell_config)

reward_cell_activity = rc.calc_reward_cell_activity_in_square_arena(positions)
reward_cell_activity = reward_cell_activity.reshape(resolution, resolution, -1).transpose(2, 0, 1)
print(reward_cell_activity.shape)

n_x, n_y = 3, 1
fig = plt.figure(figsize=(6, 2))
gs = gridspec.GridSpec(n_y, n_x, left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
# x-axis：0.05 - 0.85 (0.80)
# y-axis：0.10 - 0.90 (0.80)
for j in range(n_y):
	for k in range(n_x):
		ax = fig.add_subplot(gs[j, k])
		i_cell = j * n_x + k
		im = ax.contourf(X, Y, reward_cell_activity[i_cell], cmap="jet", levels=100)
		for r_pos in reward_positions:
			ax.add_patch(patches.Circle(xy=r_pos, radius=0.05, fc='white', ec='black'))
		ax.tick_params(bottom=False, left=False, right=False, top=False,
		               labelbottom=False, labelleft=False, labelright=False, labeltop=False)
		ax.set_aspect('equal')
cbar_ax = fig.add_axes((0.86, 0.1, 0.02, 0.8))  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0.00, 0.45, 0.9])
cbar.ax.tick_params(labelsize=18)
plt.savefig(f"../../figs/Frontiers/figure2E.png", dpi=300)
plt.close()