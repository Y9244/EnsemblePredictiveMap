import numpy as np
from config.default_config import SparseCodingConfig

class SparseCoding:
    def __init__(self, config: SparseCodingConfig):
        self.beta = config.beta # 0.3
        self.tau = config.tau # 10
        self.rng = np.random.default_rng()
        self.N_h = config.N_hip # 100
        self.N_e = config.N_mec + config.N_lec # 600 + 0

        self.n_iter = config.n_iter # 200
        self.s_h_max = config.s_h_max # 10

        self.s_h = self.rng.random((1, self.N_h))
        self.next_s_h = self.rng.random((1, self.N_h))
        self.u_h = self.rng.standard_normal(self.N_h)
        self.I_N_h = np.eye(self.N_h)
        self.R = np.zeros(self.N_e)

        self.eta = config.eta # 3e-2
        self.A = self.rng.standard_normal((self.N_e, self.N_h))
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

    def one_step(self, s_e):
        #self.s_h = self.next_s_h.copy()
        self.s_h = self.sparse_coding(s_e)
        self.sparse_coding_learning(s_e)
        return self.s_h.squeeze(0)

    def sparse_coding(self, s_e: np.ndarray) -> np.ndarray:
        """
        s_e: shape [N_pos, N_grid]
        return: s_h: shape [N_pos, N_place]
        """
        if s_e.ndim == 1:
            s_e = s_e[None, :]
        N_pos = s_e.shape[0]
        u_h = np.tile(self.u_h, (N_pos, 1))
        s_h = np.tile(self.s_h, (N_pos, 1))
        U_unit = (self.A.T @ s_e.T).T  # shape: [N_pos, N_h]

        W = self.A.T @ self.A - self.I_N_h  # shape: [N_h, N_h]

        for _ in range(self.n_iter):
            du_h = (-u_h + U_unit - s_h @ W.T) / self.tau
            u_h += du_h
            s_h = np.clip(u_h - self.beta, 0, self.s_h_max)

        #s_h = s_h / np.sum(s_h, axis=1, keepdims=True)
        return s_h # shape: [N_pos, N_h]

    def sparse_coding_learning(self, s_e):
        R = s_e - self.s_h @ self.A.T
        dA = self.eta * (R.reshape(-1, 1) @ self.s_h.reshape(1, -1))
        self.A += dA
        self.A = np.where(self.A >= 0, self.A, 0)
        norms = np.linalg.norm(self.A, axis=0, keepdims=True)
        if np.any(norms == 0):
            # print(norms)
            norms[norms == 0] = 1.0  # ゼロ割りを回避
        self.A = self.A / norms

    def reset(self):
        self.s_h = self.rng.random((1, self.N_h))
        self.pre_s_h = self.rng.random((1, self.N_h))

