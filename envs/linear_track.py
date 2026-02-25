import numpy as np
from .base_env import BaseEnv
from config.default_config import LinearTrackConfig

class LinearTrackEnv(BaseEnv):
    def __init__(self, config: LinearTrackConfig):
        super().__init__()
        self.dim = 1
        self.rng = np.random.default_rng()
        self.env_size = config.length

        self.reward_enabled = config.reward_enabled
        self.reward_info = config.reward_info
        self.reward_position = [rc for rc, rs, v in self.reward_info]
        self.reward_num = len(self.reward_info)

        self.reset_pos()

    '''------------------------------- reset methods ------------------------------'''
    def reset_pos(self, mode='start'):
        if mode == 'start':
            self.position = 0
        elif mode == 'random':
            self.position = self.rng.random()
        return self.position

    '''------------------------------- reward methods ------------------------------'''
    def get_reward(self, position):
        reward_signals = np.zeros(self.reward_num)
        if not self.reward_enabled:
            return reward_signals

        x = position
        for i, (cx, sx, value) in enumerate(self.reward_info):
            x_min, x_max = cx - sx / 2, cx + sx / 2
            if x_min <= x <= x_max:
                reward_signals[i] = value
        return reward_signals

    '''------------------------------- step methods ------------------------------'''

    def step_EC(self, action):
        self.position += action
        self.position = self.position % self.env_size
        reward = self.get_reward(self.position)
        s_m = self.grid_cell.calc_grid_activity(self.position)
        s_l = self.reward_cell.calc_reward_cell_activity_in_linear_track(self.position)
        s_e = np.concatenate([s_m, s_l])
        return s_e, reward