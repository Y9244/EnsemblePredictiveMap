import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
#sns.set()

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()
        self.dt = params["dt"]  # 時間刻み幅
        self.map_size = params["map_size"]  # 一次元トラックのサイズ
        self.max_velocity = params["max_velocity"] # 1sで進める最大の距離
        self.velocity = self.rng.uniform(0, self.max_velocity)
        self.pre_pos = params["init_pos"]  # 初期位置
        self.pos = (self.pre_pos + self.velocity * self.dt) % self.map_size
        self.vec_size = params["vec_size"]  # 一次元トラックを離散化したときのベクトルサイズ

    def one_step(self, nt, T, progress=False):
        if progress:
            print("\r{:.2f}%".format(nt / T * 100), end="")
        self.pre_pos = self.pos
        self.velocity = self.rng.uniform(0, self.max_velocity)
        self.pos += self.velocity / self.dt
        self.pos = self.pos % self.map_size

        r = np.zeros(self.vec_size)
        r[int(self.pos * self.vec_size)] = 1.0
        return r

class Sparse_Coding:
    def __init__(self, params):
        self.beta = 0.3
        self.tau = 10
        self.rng = np.random.default_rng()
        self.N_h = params["N_h"]
        self.N_e = params["N_e"]
        self.map_size = params["map_size"]
        self.vec_size = params["vec_size"]
        self.X = np.linspace(0, self.map_size, self.vec_size)
        self.dt = params["dt"]

        self.E = np.load(params["grid_files"]) # grid activities (600, 128)
        self.on_reward = params['on_reward']
        if self.on_reward:
            self.N_l_pre = params["N_l_pre"]
            self.N_l_post = params["N_l_post"]
            self.lam_pre = self.rng.normal(params['lam_pre'], 2, (self.N_l_pre, 2))
            self.lam_post = self.rng.normal(params['lam_post'], 2, (self.N_l_post, 2))
            self.lam_low = params['lam_low']
            self.lam_high = params['lam_high']
            self.set_reward_neuron_random(params["reward_pos"])
        else:
            self.E_L = self.E.T
            self.N_l_pre = 0
            self.N_l_post = 0

        self.n_iter = 200
        self.s_h_max = 10
        self.s_e = np.zeros(self.N_e + self.N_l_pre + self.N_l_post)
        self.s_h = self.rng.random(self.N_h)
        self.pre_s_h = self.rng.random(self.N_h)
        self.u_h = self.rng.standard_normal(self.N_h)
        self.I_N_h = np.eye(self.N_h)

        self.eta = 3e-2
        self.A = self.rng.standard_normal((self.N_e + self.N_l_pre + self.N_l_post, self.N_h))
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

    def set_reward_neuron(self, reward_pos):
        if reward_pos == None:
            return None

        post_reward_neuron = np.exp(-self.lam_low * (self.X - reward_pos))# - np.exp(-self.lam_high * (self.X - reward_pos))
        post_reward_neuron = np.where(self.X > reward_pos, post_reward_neuron, 0)
        post_reward_neuron /= np.max(post_reward_neuron)
        L_post = np.tile(post_reward_neuron, reps=(self.N_l_post, 1)) # (self.vec_size, self.N_post_reward)

        pre_reward_neuron = np.exp(self.lam_low * (self.X - reward_pos))# - np.exp(self.lam_high * (self.X - reward_pos))
        pre_reward_neuron = np.where(self.X < reward_pos, pre_reward_neuron, 0)
        pre_reward_neuron /= np.max(pre_reward_neuron)
        L_pre = np.tile(pre_reward_neuron, reps=(self.N_l_pre, 1)) # (self.vec_size, self.N_pre_reward)

        L = np.concatenate([L_post, L_pre], axis=0) # (self.vec_size, self.N_pre_reward+self.N_post_reward)
        self.E_L = np.concatenate([self.E, L], axis=0).T

    def set_reward_neuron_random(self, reward_pos):
        L_post = np.zeros((self.N_l_post, self.vec_size))
        L_pre = np.zeros((self.N_l_pre, self.vec_size))
        for i, l in enumerate(self.lam_post):
            post = np.exp(-np.min(l) * (self.X - reward_pos)) - np.exp(-np.max(l) * (self.X - reward_pos))
            post = np.where(self.X > reward_pos, post, 0)
            post /= np.max(post)
            L_post[i] = post
        for i, l in enumerate(self.lam_pre):
            pre = np.exp(np.min(l) * (self.X - reward_pos)) - np.exp(np.max(l) * (self.X - reward_pos))
            pre = np.where(self.X < reward_pos, pre, 0)
            pre /= np.max(pre)
            L_pre[i] = pre
        L = np.concatenate([L_post, L_pre], axis=0)  # (self.vec_size, self.N_pre_reward+self.N_post_reward)
        self.E_L = np.concatenate([self.E, L], axis=0).T

    def one_step(self, r):
        self.pre_s_h = self.s_h
        self.sparse_coding(r, self.u_h, self.s_h)
        return self.pre_s_h, self.s_h

    def sparse_coding(self, r, u_h, s_h, learning=True):
        # r: (128, )
        self.s_e = self.E_L.T @ r  # (600, 128), (128,) -> (600,)
        W = (self.A.T @ self.A) - self.I_N_h
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
        self.A = np.where(self.A >= 0, self.A, 0)
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

    def calc_place_field(self):
        K = int(5e4)
        X_recover = np.zeros((self.vec_size, K))
        loc_index = self.rng.integers(0, self.vec_size, K)
        X_recover[loc_index, range(K)] = 1
        S, U = self.sparse_coding(
            X_recover,
            self.u_h.reshape([-1, 1]),
            self.s_h.reshape([-1, 1]),
            learning=False)
        nsc_place_field = (X_recover @ S.T) / np.tile(np.sum(S, axis=1), (self.vec_size, 1))
        return nsc_place_field

    def calc_place_field_light(self):
        nsc_place_field = np.zeros((self.N_h, self.vec_size))
        for i in range(self.vec_size):
            r = np.zeros(self.vec_size)
            r[i] = 1.0
            pre_s_h, s_h = self.sparse_coding(r, self.u_h, self.s_h, learning=False)
            nsc_place_field[:, i] = s_h / np.linalg.norm(s_h)
        return nsc_place_field.T

    def show_place_field(self, nsc_place_field, display="single"):
        n = int(self.N_h**0.5)
        assert display == "single" or display == "multi", 'display variable must be "single" or "multi".'
        if display == "single":
            fig, ax = plt.subplots(figsize=(8, 4))
            for i in range(n):
                for j in range(n):
                    i_cell = i * n + j
                    ax.plot(self.X, nsc_place_field[:, i_cell])
            ax.set_ylim([-0.05, np.max(nsc_place_field)+0.05])
            plt.show()
            plt.close()
        elif display == "multi":
            fig, ax = plt.subplots(n, n, figsize=(8, 6))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.05, hspace=0.05)
            for i in range(n):
                for j in range(n):
                    i_cell = i * n + j
                    ax[i, j].plot(self.X, nsc_place_field[:, i_cell])
                    ax[i, j].set_ylim([-0.05, np.max(nsc_place_field)+0.05])
                    ax[i, j].tick_params(
                        labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                        bottom=False, left=False, right=False, top=False
                    )
            #fig.suptitle(name, fontsize=20)
            plt.show()
            plt.close()

    def calc_place_center(self, nsc_place_field):
        place_center = []
        Y = self.X*0
        n = 0
        for i in range(nsc_place_field.shape[1]):
            place_center_index = np.argmax(nsc_place_field[:, i])
            if np.max(nsc_place_field[:, i]) > 0.05:
                place_center.append(self.X[place_center_index])
                Y += norm.pdf(x=self.X, loc=self.X[place_center_index], scale=0.05)
                #Y += self.norm(X, loc=X[place_center_index], scale=0.05)
                n += 1
        return place_center

    def show_place_center(self, place_center):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(place_center, np.zeros(len(place_center)), color='black')
        ax.fill_between(self.X, self.X*0, Y/n, alpha=0.6)
        plt.show()

    def norm(self, X, loc, scale=0.5):
        Y = X*0
        for i, x in enumerate(X):
            dist = np.min([(x - loc) % 1, (loc - x) % 1])
            Y[i] = np.exp(- dist**2 / (2 * scale**2))
        return Y

    def show_input(self):
        fig, ax = plt.subplots()
        ax.imshow(self.E_L.T)
        plt.show()

class Pmap():
    def __init__(self, params):
        self.N = params["N"]
        self.vec_size = params["vec_size"]  # 1024
        self.map_size = params["map_size"]  # 1
        self.X = np.linspace(0, self.map_size, self.vec_size)

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
        """
        dM = np.zeros(self.M.shape)
        for s in range(self.N):
            dM[s, :] += self.eta * x_t1[s] * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        """
        dM = self.eta * x_t1.reshape((-1, 1)) * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        self.M += dM

    def input_activity(self, pos):
        input_act = np.zeros(self.N)
        for j in range(self.N):
            input_act[j] = self.norm(pos, j / self.N)
        input_act = input_act / np.sum(input_act)
        return input_act

    def norm(self, pos, loc, scale=0.1):
        dist = np.min([(pos - loc) % 1, (loc - pos) % 1])
        return np.exp(- dist**2 / (2 * scale**2))

    def show_M(self):
        if not self.create_fig:
            self.fig, self.ax = plt.subplots()
            self.create_fig = True
            self.ax.set_aspect("equal")
        self.ax.imshow(self.M, cmap='viridis')
        plt.pause(0.1)

    def calc_place_field(self, nsc_place_field):
        # nsc_place_field: (self.vec_size, self.N_h)
        pmap_place_field = np.zeros(nsc_place_field.shape)
        for i in range(self.vec_size):
            r = np.zeros(self.vec_size)
            r[i] = 1
            x = nsc_place_field.T @ r # (25, 128) x (128,) = (25,)
            x = x / np.sum(x)
            pmap_place_field[i] = self.M.T @ x # (25, 25) x (25,) = (25,)
        return pmap_place_field

    def show_place_field(self, pmap_place_field):
        sns.set()
        n = int(self.N**0.5)
        fig, ax = plt.subplots(n, n, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                ax[i, j].plot(self.X, pmap_place_field[:, n * i + j])
                ax[i, j].plot(self.X, np.linspace(0, self.map_size, self.vec_size)*0, color='black')
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
                ax[i, j].set_ylim([-0.05, 1.0])
        plt.show()
        plt.close()

    def calc_place_field_no_nsc(self):
        pmap_place_field = np.zeros((self.N, self.vec_size))
        for i, x in enumerate(self.X):
            input_act = self.input_activity(x)
            pmap_place_field[:, i] = self.M.T @ input_act
        return pmap_place_field.T

    def calc_place_center(self, pmap_place_field):
        # pmap_place_field: (self.vec_size, self.N_h)
        place_center = []
        for i in range(self.N):
            if np.max(pmap_place_field[:, i]) < 0.05:
                continue
            index = np.argmax(pmap_place_field[:, i])
            place_center.append(self.X[index])
        return place_center

    def calc_place_density(self, place_center, periodic=True):
        place_density = np.zeros(self.vec_size)
        if periodic:
            for i, x in enumerate(self.X):
                for pc in place_center:
                    place_density[i] += self.norm(x, loc=pc, scale=0.1)
        else:
            for pc in place_center:
                place_density += norm.pdf(self.X, loc=pc, scale=0.1)
        return place_density

    def show_place_density(self, place_density, place_center, reward_pos=None):
        sns.set()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(self.X, place_density*0, place_density/len(place_center), alpha=0.4)
        ax.scatter(place_center, np.zeros(len(place_center)), color='black')
        if reward_pos:
            ax.axvline(x=reward_pos, ymin=0, ymax=np.max(place_density), color='red')
        plt.show()
        plt.close()
