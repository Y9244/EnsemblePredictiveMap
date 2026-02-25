from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self):
        self.dim = None
        self.env_size = None
        self.position = None
        self.reward_enabled = None

    '''------------------------------- reward methods ------------------------------'''
    @abstractmethod
    def reset_pos(self):
        """環境の初期化"""
        pass

    '''------------------------------- reward methods ------------------------------'''
    @abstractmethod
    def get_reward(self, position):
        pass

    def reset_reward_info(self):
        self.reward_info = []

    def add_reward_info(self, r_info):
        self.reward_info += r_info
        self.reward_num = len(self.reward_info)
        self.reward_position = [rc for rc, rs, v in self.reward_info]

    '''------------------------------- set methods ------------------------------'''
    def set_grid_cell(self, grid_cell):
        self.grid_cell = grid_cell

    def set_reward_cell(self, reward_cell):
        self.reward_cell = reward_cell

    '''------------------------------- step methods ------------------------------'''
    def step_pos(self, action):
        self.position += action
        self.position = self.position % self.env_size
        reward = self.get_reward(self.position)
        return self.position, reward

    def step_grid(self, action):
        self.position += action
        self.position = self.position % self.env_size
        reward = self.get_reward(self.position)
        s_m = self.grid_cell.calc_grid_activity(self.position) # [N_m,]
        return s_m, reward

    @abstractmethod
    def step_EC(self, action):
        """
        エージェントの行動を適用し、次の状態と報酬を返す。
        Returns:
            new_EC_activity, reward
        """
        pass
