import numpy as np
from .base_env import BaseEnv
from config.default_config import SquareArenaConfig

class SquareArenaEnv(BaseEnv):
    def __init__(self, config: SquareArenaConfig):
        super().__init__()
        self.dim = 2
        self.rng = np.random.default_rng()
        self.env_size = config.arena_size

        self.reward_enabled = config.reward_enabled
        self.reward_info = config.reward_info
        self.reward_position = [rc for rc, rs, v in self.reward_info]
        self.reward_num = len(self.reward_info)

        self.position = self.reset_edge()

    '''------------------------------- reset methods ------------------------------'''
    def reset_pos(self, mode='random'):
        if mode == 'edge':
            self.position = self.reset_edge()
        elif mode == 'random':
            while True:
                self.position = self.rng.random(2)
                if np.sum(self.get_reward(self.position)) == 0.0:
                    break
        else:
            raise Exception('mode must be edge or random.')

    def reset_edge(self):
        if self.rng.random() < 0.5:
            if self.rng.random() < 0.5:
                x = self.rng.uniform(0.0, 0.1)
            else:
                x = self.rng.uniform(0.9, 1.0)
            y = self.rng.uniform(0.0, 1.0)
        else:
            if self.rng.random() < 0.5:
                y = self.rng.uniform(0.0, 0.1)
            else:
                y = self.rng.uniform(0.9, 1.0)
            x = self.rng.uniform(0.0, 1.0)
        return np.array([x, y])

    '''------------------------------- reward methods -------------------------------'''
    def get_reward(self, position):
        reward_signals = np.zeros(self.reward_num)
        if not self.reward_enabled:
            return reward_signals

        x, y = position
        for i, (center, size, value) in enumerate(self.reward_info):
            cx, cy = center
            sx, sy = size

            x_min, x_max = cx - sx / 2, cx + sx / 2
            y_min, y_max = cy - sy / 2, cy + sy / 2

            if x_min <= x <= x_max and y_min <= y <= y_max:
                reward_signals[i] = value
        return reward_signals

    '''------------------------------- step methods -------------------------------'''

    def set_boundary_cell(self, border_cell):
        self.border_cell = border_cell

    def step_grid_border(self, action):
        self.position += action
        self.position = self.position % self.env_size
        reward = self.get_reward(self.position)
        s_m = self.grid_cell.calc_grid_activity(self.position)
        is_boundary_region = (
            (self.position[0] <= 0.1) | (self.position[0] >= 0.9) |  # 左右の境界
            (self.position[1] <= 0.1) | (self.position[1] >= 0.9)    # 上下の境界
        )
        if is_boundary_region:
            s_m = np.zeros(self.grid_cell.N_e)
        s_b = self.border_cell.calc_border_activity(self.position)

        s_e = np.concatenate([s_m, s_b])

        # s_e = (s_e - np.mean(s_e)) / np.std(s_e)
        return s_e, reward

    def step_EC(self, action):
        self.position += action
        self.position = self.position % self.env_size
        reward = self.get_reward(self.position)
        s_m = self.grid_cell.calc_grid_activity(self.position)
        is_boundary_region = (
            (self.position[0] <= 0.1) | (self.position[0] >= 0.9) |  # 左右の境界
            (self.position[1] <= 0.1) | (self.position[1] >= 0.9)    # 上下の境界
        )
        if is_boundary_region:
            s_m = np.zeros(self.grid_cell.N_e)

        s_b = self.border_cell.calc_border_activity(self.position)

        s_l = self.reward_cell.calc_reward_cell_activity_in_square_arena(self.position)
        s_e = np.concatenate([s_m, s_b, s_l])
        return s_e, reward
