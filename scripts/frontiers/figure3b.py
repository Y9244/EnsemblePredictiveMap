import numpy as np
import matplotlib.pyplot as plt
from config.default_config import RewardCellConfig
from models.reward_cell import RewardCell
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import seaborn as sns
#sns.set()
length = 4.0

reward_pos = 3.66
reward_cell_cfg = RewardCellConfig(dim=1, env_size=length, reward_positions=[reward_pos], N_reward_cell=400)
reward_cell = RewardCell(reward_cell_cfg)

resolution = 1000
X = np.linspace(0, length, resolution)

#reward_pos = 1.66
#reward_pos = 2.3
#reward_cell.change_reward_cell(reward_pos)
reward_cell_activity = reward_cell.calc_reward_cell_activity_in_linear_track(X)

fig = plt.figure(figsize=(5, 7))

gs = gridspec.GridSpec(1, 1, left=0.15, right=0.85, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[0, 0])

im = ax.imshow(reward_cell_activity.T, cmap='binary', origin='lower', aspect='auto')
ax.axvline(reward_pos / length * resolution, ymin=0, ymax=600, linewidth=3)
ax.set_xticks([i/6 * 1000 for i in range(7)])              # 元の目盛り位置（例）
ax.set_xticklabels(['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])        # 表示したいラベル
ax.set_yticks([0, 400])              # 元の目盛り位置（例）
ax.set_xlabel('position (m)', fontsize=22)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)  # カラーバー軸を追加
cbar = fig.colorbar(im, cax=cax)
fig.set_size_inches(4, 5.5)
#plt.savefig(f"../../figs/Frontiers/figure2D.png", dpi=300)
plt.show()



