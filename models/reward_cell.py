import numpy as np
from config.default_config import RewardCellConfig

class RewardCell:
    def __init__(self, config: RewardCellConfig):
        self.rng = np.random.default_rng()
        self.dim = config.dim
        self.N_reward_cell = config.N_reward_cell
        self.reward_center_range = config.reward_center_range

        if config.reward_positions == None:
            raise Exception("RewardCellConfigのreward_positionsを設定してください")

        if self.dim == 1:
            self.env_size = config.env_size
            self.reward_position = config.reward_positions[0]
            self.N_pre_reward_cell = self.N_reward_cell // 2
            self.N_post_reward_cell = self.N_reward_cell - self.N_pre_reward_cell
            self.pre_sigma_values, self.pre_delta_values, self.pre_shift_values = self.get_reward_cell_decay(self.N_pre_reward_cell)
            self.post_sigma_values, self.post_delta_values, self.post_shift_values = self.get_reward_cell_decay(self.N_post_reward_cell)

        elif self.dim == 2:
            # [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25)]
            self.reward_positions = np.array(config.reward_positions)
            self.reward_pos_nums = len(self.reward_positions)
            self.reward_cell_centers = self.get_reward_cell_centers() # (200, 3, 2)
            # (200, 3), (200, 3)
            self.reward_cell_scales, self.reward_cell_std = self.get_reward_cell_scales_var()

            self.reward_cell_centers = self.reward_cell_centers.transpose(1, 0, 2)
            self.reward_cell_scales = self.reward_cell_scales.transpose(1, 0)
            self.reward_cell_std = self.reward_cell_std.transpose(1, 0)
        else:
            raise Exception("RewardCell dim must be 1 or 2.")

    def get_reward_cell_centers(self):
        reward_cell_centers = np.stack([self.reward_positions] * self.N_reward_cell, axis=0)
        eps = self.rng.uniform(-self.reward_center_range, self.reward_center_range, reward_cell_centers.shape)
        reward_cell_centers += eps
        return reward_cell_centers

    def get_reward_cell_scales_var(self):
        reward_cell_scales = np.zeros((self.N_reward_cell, self.reward_pos_nums))
        scale_eps = self.rng.uniform(0.5, 1.0, reward_cell_scales.shape)
        reward_cell_scales += scale_eps

        reward_cell_std = np.zeros((self.N_reward_cell, self.reward_pos_nums))
        var_eps = self.rng.uniform(0.03, 0.06, reward_cell_scales.shape)
        reward_cell_std += var_eps
        return reward_cell_scales, reward_cell_std

    def calc_reward_cell_activity_in_square_arena(self, pos):
        if len(pos.shape) == 1:
            pos = pos[None, :]
            single_input = True
        else:
            single_input = False
        reward_cell_activity = np.zeros((pos.shape[0], self.N_reward_cell))  # (B, 200)

        for i in range(self.reward_pos_nums):
            rc_centers = self.reward_cell_centers[i] # (200, 2)
            rc_scale = self.reward_cell_scales[i] # (200,)
            rc_std = self.reward_cell_std[i] # (200,)

            delta = pos[:, None, :] - rc_centers[None, :, :] # (B, 200, 2) = (B, 1, 2) - (1, 200, 2)
            squared_dist = np.sum(delta ** 2, axis=2) # (B, 200)
            scale = rc_scale[None, :]  # (1, 200)
            sigma_sq = rc_std[None, :] ** 2 # (1, 200)
            #print(np.max(sigma_sq))
            reward_cell_activity += scale * np.exp(-squared_dist / (2 * sigma_sq))  # shape: (B, 200)

        if single_input:
            reward_cell_activity = reward_cell_activity.squeeze(0)  # shape: (200,)
        return reward_cell_activity

    def get_reward_cell_decay(self, N_reward_cell):
        """tau_rise = np.random.uniform(0.05, 0.1, N_reward_cell)
        tau_decay = tau_rise + np.random.uniform(0.2, 0.4, N_reward_cell)
        peak = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        B = (np.exp(- peak / tau_decay) - np.exp(-peak / tau_rise)) ** (-1)
        return tau_rise[None, :], tau_decay[None, :], peak[None, :], B[None, :]"""

        sigma_values = self.rng.uniform(0.3, 0.6, size=(1, N_reward_cell))  # 広がり
        delta_values = self.rng.uniform(0.01, 0.05, size=(1, N_reward_cell))
        shift_values = self.rng.uniform(-0.05, 0.05, size=(1, N_reward_cell))
        shift_values = np.sort(shift_values)
        return sigma_values, delta_values, shift_values


    def calc_reward_cell_activity_in_linear_track(self, pos):
        if (type(pos) == float) or (type(pos) == np.float64):
            pos = np.array([pos])
            single_input = True
        else:
            single_input = False

        pre_reward_cell_activity = self.calc_pre_reward_cell_in_linear_track(pos) # (B, N_pre_reward)
        post_reward_cell_activity = self.calc_post_reward_cell_in_linear_track(pos)  # (B, N_post_reward)

        reward_cell_activity = np.concatenate([pre_reward_cell_activity, post_reward_cell_activity], axis=1)
        if single_input:
            reward_cell_activity = reward_cell_activity.squeeze(0)  # shape: (200,)
        return reward_cell_activity

    def calc_pre_reward_cell_in_linear_track(self, pos):
        # pos.shape: [B,]
        # pre_peak, pre_tau_decay, pre_tau_rise, pre_B: [1, N_pre_reward_cell]
        pos = pos[:, None]
        shifted_r_pos = (self.reward_position + self.pre_shift_values)
        gaussian = np.exp(-((pos - shifted_r_pos) ** 2) / (2 * self.pre_sigma_values ** 2))
        gate = 1 - sigmoid((pos - shifted_r_pos) / self.pre_delta_values)
        pre_reward_cell_activity = gaussian * gate

        shifted_r_pos = (self.reward_position + self.pre_shift_values) - self.env_size
        gaussian = np.exp(-((pos - shifted_r_pos) ** 2) / (2 * self.pre_sigma_values ** 2))
        gate = 1 - sigmoid((pos - shifted_r_pos) / self.pre_delta_values)
        pre_reward_cell_activity += gaussian * gate
        return pre_reward_cell_activity

    def calc_post_reward_cell_in_linear_track(self, pos):
        # pos.shape: [B,]
        # post_peak, post_tau_decay, post_tau_rise, post_B: [1, N_post_reward_cell]
        pos = pos[:, None]
        shifted_r_pos = (self.reward_position + self.post_shift_values)
        gaussian = np.exp(-((pos - shifted_r_pos) ** 2) / (2 * self.post_sigma_values ** 2))
        gate = 1 - sigmoid((shifted_r_pos - pos) / self.post_delta_values)
        post_reward_cell_activity = gaussian * gate

        shifted_r_pos = (self.reward_position + self.post_shift_values) - self.env_size
        gaussian = np.exp(-((pos - shifted_r_pos) ** 2) / (2 * self.post_sigma_values ** 2))
        gate = 1 - sigmoid((shifted_r_pos - pos) / self.post_delta_values)
        post_reward_cell_activity += gaussian * gate
        return post_reward_cell_activity

    def change_reward_cell(self, r_pos):
        self.reward_position = r_pos


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
