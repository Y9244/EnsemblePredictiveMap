import numpy as np
import matplotlib.pyplot as plt
from config.default_config import BoundaryCellConfig


class BoundaryCell:
    def __init__(self, config: BoundaryCellConfig):
        self.N_cells = config.N_cells
        self.sigma_range = config.sigma_range
        self.arena_size = config.arena_size

        self.rng = np.random.default_rng()

        # 各セルの方向を割り当て(0:左右、1:上下)
        self.directions = np.zeros(self.N_cells)
        self.directions[:self.N_cells//2] += 1

        self.sigmas_d = self.rng.uniform(
            self.sigma_range[0], self.sigma_range[1], size=self.N_cells
        )

        """# 壁に沿う方向のσ（s方向）
        scale = 1
        self.sigmas_s = self.rng.uniform(
            self.sigma_range[0]*scale, self.sigma_range[1]*scale, size=self.N_cells
        )"""

        # s方向の中心 (壁に沿った方向の好みの位置)
        self.pref_s = self.rng.uniform(0, self.arena_size, size=self.N_cells)

        # 距離方向に微小なランダムシフト（左右の壁から少しずらす）
        self.shifts = self.rng.uniform(-1, 1, size=self.N_cells) * 0.12
        self.shifts = np.zeros(self.N_cells)


    def calc_border_activity(self, positions):
        dimention = len(positions.shape)
        if dimention == 1:
            positions = positions[None, :] # (1, 2)
        x, y = positions[:, 0], positions[:, 1] # (N_pos,), (N_pos,)

        x = x[:, None] + self.shifts  # (N_pos, N_cells)
        y = y[:, None] + self.shifts  # (N_pos, N_cells)

        # (N_pos, N_cells)
        activity = np.zeros((positions.shape[0], self.N_cells), dtype=np.float32)

        lr_mask = self.directions == 0    # (N_cells,)
        tb_mask = self.directions == 1    # (N_cells,)

        dist_x = np.min([x, self.arena_size - x], axis=0) # (N_pos, N_cells)
        dist_y = np.min([y, self.arena_size - y], axis=0) # (N_pos, N_cells)

        # (N_pos, N_cells//2) = (N_pos, 1) / (1, N_cells//2)
        gauss_d = np.exp(- (dist_x[:, lr_mask] ** 2) / (2 * self.sigmas_d[None, lr_mask] ** 2))
        gauss_s = np.exp(- ((y[:, lr_mask] - self.pref_s[None, lr_mask]) ** 2) / (2 * self.sigmas_d[None, lr_mask] ** 2))
        activity[:, lr_mask] = gauss_d * gauss_s * 0.833333

        gauss_d = np.exp(- (dist_y[:, tb_mask] ** 2) / (2 * self.sigmas_d[None, tb_mask] ** 2))
        gauss_s = np.exp(- ((x[:, tb_mask] - self.pref_s[None, tb_mask]) ** 2) / (2 * self.sigmas_d[None, tb_mask] ** 2))
        activity[:, tb_mask] = gauss_d * gauss_s * 0.833333

        if dimention == 1:
            return activity[0]
        else:
            return activity

if __name__ == "__main__":
    boundary_config = BoundaryCellConfig()
    bc = BoundaryCell(boundary_config)

    resolution = 256
    env_size = 1.0
    x = np.linspace(0, env_size, resolution)
    y = np.linspace(0, env_size, resolution)
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
    print(positions.shape)

    boundary_fields = bc.calc_border_activity(positions).T
    print(boundary_fields.shape)

    N = 10
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.05, hspace=0.05)
    for i in range(N):
        for j in range(N):
            i_cell = N * i + j
            b_field = boundary_fields[i_cell].reshape(resolution, resolution)
            ax[i, j].imshow(b_field, cmap='jet')
            ax[i, j].axis('off')
    plt.show()

    for i_cell in range(100):
        fig, ax = plt.subplots(figsize=(8, 8))
        b_field = boundary_fields[i_cell].reshape(resolution, resolution)
        im = ax.imshow(b_field, cmap='jet')
        cbar = fig.colorbar(im)
        plt.show()