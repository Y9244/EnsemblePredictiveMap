import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()

        self.vec_size = params["vec_size"]  # 2次元アリーナを離散化したときのベクトルサイズ
        self.map_size = params["map_size"]  # 2次元アリーナのサイズ
        self.dt = params["dt"]  # 時間刻み幅

        self.velocity = params["velocity"] # 1sで進める最大の距離
        self.theta = params["theta"]
        self.pre_pos = params["init_pos"]  # 初期位置
        self.pos = deepcopy(self.pre_pos)
        delta_pos = self.delta_pos()
        self.pos[0] += delta_pos[0]
        self.pos[1] += delta_pos[1]

    def one_step(self, nt, T, progress=False):
        if progress:
            print("\r{:.2f}%".format(nt / T * 100), end="")
        self.pre_pos = deepcopy(self.pos)
        delta_pos = self.delta_pos()
        self.pos[0] += delta_pos[0]
        self.pos[1] += delta_pos[1]
        self.pos[0] = self.pos[0] % self.map_size
        self.pos[1] = self.pos[1] % self.map_size

        self.r = np.zeros((self.vec_size, self.vec_size))
        self.r[-int(self.pos[1] * self.vec_size)-1, int(self.pos[0] * self.vec_size)] = 1.0
        return self.r

    def delta_pos(self):
        omega = self.rng.normal()
        self.theta += omega * self.dt
        self.theta = self.theta % np.pi - np.pi/2
        delta_pos = [0, 0]
        delta_pos[0] = self.velocity * (np.sin(self.theta + omega * self.dt) - np.sin(self.theta)) / omega
        delta_pos[1] = self.velocity * (- np.cos(self.theta + omega * self.dt) + np.cos(self.theta)) / omega
        return delta_pos

    def one_step_show(self, ax):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        ax.scatter(self.pos[0], self.pos[1], marker="x", color="black")
        plt.pause(0.01)

class Sparse_Coding:
    def __init__(self, params):
        self.beta = 0.3
        self.tau = 10
        self.rng = np.random.default_rng()
        self.N_h = params["N_h"]
        self.N_e = params["N_e"]
        self.vec_size = params["vec_size"]
        self.map_size = params["map_size"]  # 2次元アリーナのサイズ
        self.dt = params["dt"]

        self.E = np.load(params["grid_files"]) # grid activities (600, 128)
        self.E = np.reshape(self.E, [self.N_e, -1]).T # (128, 600)

        self.n_iter = 200
        self.s_h_max = 10
        self.s_e = np.zeros(self.N_e)
        self.s_h = self.rng.random(self.N_h)
        self.pre_s_h = self.rng.random(self.N_h)
        self.u_h = self.rng.standard_normal(self.N_h)
        self.I_N_h = np.eye(self.N_h)
        self.R = np.zeros(self.N_e)

        self.eta = 3e-2
        self.A = self.rng.standard_normal((self.N_e, self.N_h))
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

    def one_step(self, r, learning=True):
        self.pre_s_h = self.s_h.copy()
        self.sparse_coding(r.reshape(-1), self.u_h, self.s_h, learning=learning)
        return self.pre_s_h, self.s_h

    def sparse_coding(self, r, u_h, s_h, learning=True):
        # r: (1024, )
        self.s_e = self.E.T @ r  # (600, 1024) @ (1024,) -> (600,)
        W = (self.A.T @ self.A) - self.I_N_h #
        U_unit = self.A.T @ self.s_e

        for _ in range(self.n_iter):
            du_h = (-u_h + U_unit - np.dot(W, s_h)) / self.tau
            u_h = u_h + du_h * self.dt
            s_h = np.where(u_h - self.beta > 0, u_h - self.beta, 0)
            s_h = np.where(s_h > self.s_h_max, self.s_h_max, s_h)

        if learning:
            self.u_h = u_h
            self.s_h = s_h
            self.sparse_coding_learning()
        else:
            return u_h, s_h

    def sparse_coding_learning(self):
        R = self.s_e - self.A @ self.s_h
        dA = self.eta * (R.reshape(-1, 1) @ self.s_h.reshape(1, -1))
        self.A += dA
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)
        self.A = np.where(self.A >= 0, self.A, 0)

    def calc_place_field_light(self):
        nsc_place_field = np.zeros((self.N_h, self.vec_size ** 2))
        for i in range(self.vec_size ** 2):
            r = np.zeros(self.vec_size ** 2)
            r[i] = 1.0
            pre_s_h, s_h = self.sparse_coding(r, self.u_h, self.s_h, learning=False)
            nsc_place_field[:, i] = s_h / np.linalg.norm(s_h)
        return nsc_place_field.T

    def calc_place_field(self):
        K = int(5e4)
        X_recover = np.zeros((self.vec_size**2, K))
        loc_index = self.rng.integers(0, self.vec_size**2, K)
        X_recover[loc_index, range(K)] = 1
        S, U = self.sparse_coding(
            X_recover,
            self.u_h.reshape([-1, 1]),
            self.s_h.reshape([-1, 1]),
            learning=False)
        """
        X_recover : (1024, 5e4)
        S.T       : (5e4, 100)
        (X_recover @ S.T) : (1024, 100)
        np.tile(np.sum(S, axis=1), (self.vec_size**2, 1)) : (1024, 100)
        """
        nsc_place_field = (X_recover @ S.T)# / np.tile(np.sum(S, axis=1), (self.vec_size**2, 1))
        return nsc_place_field

    def show_place_field(self, nsc_place_field, i_cell=None):
        # nsc_place_field: (self.vec_size ** 2, self.N_h)
        place_cell_sum = np.zeros((self.vec_size, self.vec_size))
        n = int(self.N_h**0.5)
        X, Y = np.linspace(0, self.map_size, self.vec_size), np.linspace(0, self.map_size, self.vec_size)
        X, Y = np.meshgrid(X, Y)

        if i_cell:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.contourf(X, Y, np.abs(nsc_place_field[:, i_cell].reshape(self.vec_size, self.vec_size)), cmap='jet', levels=100)
            ax.tick_params(
                labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                bottom=False, left=False, right=False, top=False
            )
            plt.show()
            plt.close()
        else:
            fig, ax = plt.subplots(n, n, figsize=(8, 8))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.05, hspace=0.05)
            for i in range(n):
                for j in range(n):
                    i_cell = i * n + j
                    ax[i, j].contourf(X, Y, np.abs(nsc_place_field[:, i_cell].reshape(self.vec_size, self.vec_size)), cmap='jet', levels=100)
                    ax[i, j].tick_params(
                        labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                        bottom=False, left=False, right=False, top=False
                    )
            plt.show()
            plt.close()

    def calc_place_center(self, place_field_recovered):
        place_center = np.zeros((self.N_h, 2))
        n = int(self.N_h ** 0.5)
        for i in range(n):
            for j in range(n):
                i_cell = i * n + j
                place_cell = np.abs(place_field_recovered[:, i_cell].reshape(self.vec_size, self.vec_size))
                xc, yc = np.unravel_index(np.argmax(place_cell), place_cell.shape)
                place_center[i_cell, 0] = xc / self.vec_size
                place_center[i_cell, 1] = yc / self.vec_size
        return place_center

    def show_place_center(self, place_center):
        fig, ax = plt.subplots()
        ax.scatter(place_center[:, 0], place_center[:, 1], color='black')
        ax.set_aspect("equal")
        plt.show()
        plt.close()

    def linear_fit(self, X, gamma, xc, yc, sigma):
        x, y = X
        z = gamma * np.exp(-np.log(5) * (((x - xc)**2 + (y - yc)**2) / sigma**2))
        return z.ravel()

class Pmap():
    def __init__(self, params):
        self.N = params["N"] # 100
        self.vec_size = params["vec_size"] # 1024
        self.map_size = params["map_size"] # 1

        self.M = np.identity(self.N)
        self.eta = params['eta']
        self.gamma = params['gamma']
        self.create_fig = False

    def learning_M(self, pre_s_h, s_h, sparse_coding=True):
        if sparse_coding:
            x_t1 = pre_s_h / np.linalg.norm(pre_s_h)
            x_t2 = s_h / np.linalg.norm(s_h)
        else:
            x_t1 = self.input_activity(pre_s_h)
            x_t2 = self.input_activity(s_h)
        #self.M[np.argmax(x_t1), :] += self.eta * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)

        """
        dM = np.zeros(self.M.shape)
        for s in range(self.N):
            dM[s, :] += self.eta * x_t1[s] * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        """
        dM = self.eta * x_t1.reshape((-1, 1)) * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        self.M += dM

    def input_activity(self, pos):
        n = int(self.N ** 0.5)
        input_act = np.zeros((n, n)) # posにいるときの場所細胞の活動
        for i in range(n):
            for j in range(n):
                # i == y, j == x
                input_act[i, j] = self.norm(pos, (j/n, i/n))
        input_act = input_act / np.sum(input_act)
        return input_act.reshape(-1)

    def norm(self, pos, loc, scale=0.1):
        #print(pos, loc)
        dist_x = np.min([(pos[0] - loc[0]) % 1, (loc[0] - pos[0]) % 1])
        dist_y = np.min([(pos[1] - loc[1]) % 1, (loc[1] - pos[1]) % 1])
        return np.exp(- (dist_x**2 + dist_y**2) / (2 * scale ** 2))

    def imshow_M(self):
        if not self.create_fig:
            self.fig, self.ax = plt.subplots()
            self.create_fig = True
            self.ax.set_aspect("equal")
        self.ax.imshow(self.M)
        plt.pause(0.1)

    def calc_place_field(self, nsc_place_field):
        # nsc_place_field: (self.vec_size ** 2, self.N_h)
        pmap_place_field = np.zeros(nsc_place_field.shape)
        for i in range(self.vec_size**2):
            r = np.zeros(self.vec_size**2)
            r[i] = 1
            x = nsc_place_field.T @ r # (100, 1024) x (1024,) = (100,)
            sum_x = np.sum(x)
            x = x / sum_x
            pmap_place_field[i] = self.M.T @ x # (100, 100) x (100,) = (100,)
        return pmap_place_field

    def show_place_field(self, pmap_place_field, i_cell=None):
        n = int(self.N**0.5)
        if i_cell:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(pmap_place_field[:, i_cell].reshape(self.vec_size, self.vec_size), cmap="jet")
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.tick_params(bottom=False, left=False, right=False, top=False)
            ax.invert_yaxis()
            plt.show()
            plt.close()
        else:
            fig, ax = plt.subplots(n, n, figsize=(8, 8))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            for i in range(n):
                for j in range(n):
                    ax[i, j].imshow(pmap_place_field[:, n * i + j].reshape(self.vec_size, self.vec_size), cmap="jet")
                    ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
                    ax[i, j].invert_yaxis()
            plt.show()
            plt.close()

    def calc_place_field_no_nsc(self):
        X, Y = np.linspace(0, self.map_size, self.vec_size), np.linspace(0, self.map_size, self.vec_size)
        pmap_place_field = np.zeros((self.N, self.vec_size, self.vec_size))
        n = int(self.N ** 0.5)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                input_act = self.input_activity([x, y])
                pmap_place_field[:, j, i] = self.M.T @ input_act
        return pmap_place_field.reshape(self.N, self.vec_size**2).T