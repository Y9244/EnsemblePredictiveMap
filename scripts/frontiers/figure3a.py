import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from models.reward_cell import RewardCell
from config.default_config import RewardCellConfig
import seaborn as sns
sns.set()

reward_positions = [2.3]
resolution = 256
env_size = 3.0
N_reward_cell = 400
X = np.linspace(0, env_size, resolution)

reward_cell_config = RewardCellConfig(
    dim=1,
    env_size=env_size,
    reward_positions=reward_positions,
    N_reward_cell=N_reward_cell,
)
reward_cell = RewardCell(reward_cell_config)

# (400, 256)
reward_cell_activity = reward_cell.calc_reward_cell_activity_in_linear_track(X).T

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i in range(30):
    ax[0].plot(X, reward_cell_activity[i], color='tab:purple', zorder=10, alpha=0.3)
    ax[1].plot(X, reward_cell_activity[-i-1], color='tab:green', zorder=10, alpha=0.3)
ax[0].axvline(x=reward_positions[0], ymin=0, ymax=1.0, color='skyblue', linewidth=3)
ax[1].axvline(x=reward_positions[0], ymin=0, ymax=1.0, color='skyblue', linewidth=3)
plt.savefig(f"../../figs/Frontiers/figure2C.png", dpi=300)
plt.show()