import numpy as np
from config.default_config import PmapConfig

class Pmap():
    def __init__(self, config: PmapConfig):
        self.rng = np.random.default_rng()
        self.N_h = config.N_hip # 100
        self.n = int(self.N_h ** 0.5)

        self.M = np.identity(self.N_h)
        self.eta_M = config.eta_M
        self.gamma = config.gamma

    def one_step_from_pos(self, pre_pos, pos):
        x_t1 = self.input_activity(pre_pos)
        x_t2 = self.input_activity(pos)

        x_t1, pre_p_h = self.predictive_map(x_t1)
        x_t2, p_h = self.predictive_map(x_t2)

        self.M += self.eta_M * x_t1.reshape((-1, 1)) * (x_t1 + self.gamma * p_h - pre_p_h)
        return pre_p_h, p_h

    def one_step_from_sc(self, s_h):
        p_h = self.predictive_map(s_h)
        p_h = p_h.squeeze(0)
        return p_h

    def predictive_map_learning(self, s_h, p_h, next_p_h):
        s_h = s_h / np.sum(s_h)
        self.M += self.eta_M * s_h[:, None] * (s_h + self.gamma * next_p_h - p_h)

    def predictive_map(self, s_h): # shape: [N_pos, N_h]
        if s_h.ndim == 1:
            s_h = s_h[None, :]
        elif s_h.ndim == 2:
            pass
        else:
            raise Exception("s_h.ndim must be 1 or 2.")

        s_h = s_h / np.sum(s_h, axis=1, keepdims=True)
        p_h = s_h @ self.M # shape: [N_pos, N_h]

        return p_h


    def input_activity(self, pos):
        input_act = np.zeros((self.n, self.n))  # posにいるときの場所細胞の活動
        for i in range(self.n):
            for j in range(self.n):
                input_act[i, j] = self.norm(pos, (j / self.n, i / self.n))
        # input_act = input_act / np.sum(input_act)
        return input_act.reshape(1, -1)

    def norm(self, pos, loc, scale=0.1):
        dist_x = np.min([(pos[0] - loc[0]) % 1, (loc[0] - pos[0]) % 1])
        dist_y = np.min([(pos[1] - loc[1]) % 1, (loc[1] - pos[1]) % 1])
        return np.exp(- (dist_x ** 2 + dist_y ** 2) / (2 * scale ** 2))