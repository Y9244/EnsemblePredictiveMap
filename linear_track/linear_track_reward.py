import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import seaborn as sns
sns.set()

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()
        self.max_velocity = params["max_velocity"] # 1sで進める最大の距離
        self.velocity = None
        self.pos = params["init_pos"] # 初期位置
        self.vec_size = params["vec_size"] # 一次元トラックを離散化したときのベクトルサイズ
        self.map_size = params["map_size"] # 一次元トラックのサイズ
        self.dt = params["dt"] # 時間刻み幅

    def one_step(self, nt, T, progress=False):
        if progress:
            print("\r{:.2f}%".format(nt / T * 100), end="")
        self.velocity = self.rng.uniform(0, self.max_velocity)
        self.pos += self.velocity / self.dt
        self.pos = self.pos % 1.0

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
        self.vec_size = params["vec_size"]
        self.dt = params["dt"]

        self.E = np.load(params["grid_files"]) # grid activities (600, 128)
        self.N_reward = params["N_reward"]
        self.rewardA1 = np.tile(params['rewardA1'], reps=(self.N_reward // 3, 1))  # (10, 128)
        self.rewardA2 = np.tile(params['rewardA2'], reps=(2 * self.N_reward // 3, 1))  # (10, 128)
        self.rewardB1 = np.tile(params['rewardB1'], reps=(self.N_reward // 3, 1))  # (10, 128)
        self.rewardB2 = np.tile(params['rewardB2'], reps=(2 * self.N_reward // 3, 1))  # (10, 128)
        self.E_A = np.concatenate([self.E, self.rewardA1, self.rewardA2], axis=0).T # (128, 610)
        self.E_B = np.concatenate([self.E, self.rewardB1, self.rewardB2], axis=0).T  # (128, 610)

        self.n_iter = 200
        self.s_h_max = 10
        self.s_e = np.zeros(self.N_e + self.N_reward)
        self.s_h = self.rng.random(self.N_h)
        self.pre_s_h = self.rng.random(self.N_h)
        self.u_h = self.rng.standard_normal(self.N_h)
        self.I_N_h = np.eye(self.N_h)

        self.eta = 3e-2
        self.A = self.rng.standard_normal((self.N_e + self.N_reward, self.N_h))
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

        self.place_fields = np.zeros((self.N_h, self.vec_size**2))

    def one_step(self, r, context):
        self.pre_s_h = self.s_h
        self.sparse_coding(r, self.u_h, self.s_h, context)
        return self.pre_s_h, self.s_h

    def sparse_coding(self, r, u_h, s_h, context, learning=True):
        # r: (128, )
        if context == "A":
            self.s_e = self.E_A.T @ r  # (600, 128), (128,) -> (600,)
        elif context == "B":
            self.s_e = self.E_B.T @ r  # (600, 128), (128,) -> (600,)
        else:
            print("contextにはAかBを入力してください。")
        W = (self.A.T @ self.A) - self.I_N_h
        U_unit = self.A.T @ self.s_e

        for _ in range(self.n_iter):
            du_h = (-u_h + U_unit - np.dot(W, s_h)) / self.tau
            #print(du_h.shape, u_h.shape)
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

    def calc_place_field(self, context, save=True):
        K = int(5e4)
        X_recover = np.zeros((self.vec_size, K))
        loc_index = self.rng.integers(0, self.vec_size, K)
        X_recover[loc_index, range(K)] = 1
        S, U = self.sparse_coding(
            X_recover,
            self.u_h.reshape([-1, 1]),
            self.s_h.reshape([-1, 1]),
            context=context,
            learning=False)
        self.place_field_recovered = (X_recover @ S.T) / np.tile(np.sum(S, axis=1), (self.vec_size, 1))
        return self.place_field_recovered

    def imshow_place_field(self, place_field_recovered, name=None, display="single"):
        n = int(self.N_h**0.5)
        X = np.linspace(0, 1, self.vec_size)
        assert display == "single" or display == "multi", 'display variable must be "single" or "multi".'
        if display == "single":
            fig, ax = plt.subplots(figsize=(8, 4))
            for i in range(n):
                for j in range(n):
                    i_cell = i * n + j
                    ax.plot(X, place_field_recovered.T[i_cell])
            ax.set_ylim([-0.05, np.max(place_field_recovered)+0.05])
            fig.suptitle(name)
            plt.show()
            plt.close()
        elif display == "multi":
            fig, ax = plt.subplots(n, n, figsize=(8, 6))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.05, hspace=0.05)
            for i in range(n):
                for j in range(n):
                    i_cell = i * n + j
                    ax[i, j].plot(X, place_field_recovered.T[i_cell])
                    ax[i, j].set_ylim([-0.05, np.max(place_field_recovered)+0.05])
                    ax[i, j].tick_params(
                        labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                        bottom=False, left=False, right=False, top=False
                    )
            #fig.suptitle(name, fontsize=20)
            plt.show()
            plt.close()

    def show_place_center(self, place_field_recovered, context):
        place_center = []
        X = np.linspace(0, 1, self.vec_size)
        Y = X*0
        n = 0
        for i in range(place_field_recovered.shape[1]):
            place_center_index = np.argmax(place_field_recovered[:, i])
            if np.max(place_field_recovered[:, i]) > 0.05:
                place_center.append(X[place_center_index])
                Y += scipy.stats.norm.pdf(x=X, loc=X[place_center_index], scale=0.05)
                #Y += self.norm(X, loc=X[place_center_index], scale=0.05)
                n += 1
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(place_center, np.zeros(len(place_center)), color='black')
        ax.fill_between(X, X*0, Y/n, alpha=0.6)
        ax.set_title("The density of place center in context {}".format(context), fontsize=24)
        plt.show()

    def norm(self, X, loc, scale=0.5):
        Y = X*0
        for i, x in enumerate(X):
            dist = np.min([(x - loc) % 1, (loc - x) % 1])
            Y[i] = np.exp(- dist**2 / (2 * scale**2))
        return Y

class Pmap():
    def __init__(self, params):
        self.N = params["N"]
        self.vec_size = params["vec_size"]  # 1024
        self.map_size = params["map_size"]  # 1

        self.M = np.identity(self.N)
        self.eta = params['eta']
        self.gamma = params['gamma']
        self.create_fig = False

    def learning_M(self, pre_s_h, s_h):
        # x_t1 = self.input_activity(pre_pos)
        # x_t2 = self.input_activity(pos)
        x_t1 = pre_s_h / np.linalg.norm(pre_s_h)
        x_t2 = s_h / np.linalg.norm(s_h)
        #dM = np.zeros(self.M.shape)
        dM = self.eta * x_t1.reshape((-1, 1)) * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        """for s in range(self.N):
            dM[s, :] += self.eta * x_t1[s] * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
            # self.M[s, ] += self.eta * x_t1[s] * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)"""
        self.M += dM

    def input_activity(self, pos):
        input_act = np.zeros(self.N)
        for j in range(self.N):
            input_act[j] = self.norm(pos, j)
        #input_act[-1] = self.rng.random()
        # input_act = input_act / np.sum(input_act)
        return input_act

    def norm(self, pos, loc, scale=0.5):
        dist = np.min([(pos - loc) % 1, (loc - pos) % 1])
        return np.exp(- dist**2 / (2 * scale**2))

    def show_M(self):
        if not self.create_fig:
            self.fig, self.ax = plt.subplots()
            self.create_fig = True
            self.ax.set_aspect("equal")
        self.ax.imshow(self.M)
        plt.pause(0.1)

    def place_field(self, place_field_recovered):
        vec_size = place_field_recovered.shape[0]
        pmap_place_field = np.zeros(place_field_recovered.shape) # (128, 25)
        for i in range(vec_size):
            r = np.zeros(vec_size)
            r[i] = 1
            x = place_field_recovered.T @ r # (25, 128) x (128,) = (25,)
            x = x / np.sum(x)
            pmap_place_field[i] = self.M.T @ x # (25, 25) x (25,) = (25,)
        n = int(self.N**0.5)
        fig, ax = plt.subplots(n, n, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                #pmap_place_field[:, n * i + j] = (pmap_place_field[:, n * i + j] - np.min(pmap_place_field[:, n * i + j])) / (np.max(pmap_place_field[:, n * i + j]) - np.min(pmap_place_field[:, n * i + j]))
                ax[i, j].plot(np.linspace(0, 1, vec_size), pmap_place_field[:, n * i + j])
                ax[i, j].plot(np.linspace(0, 1, vec_size), np.linspace(0, 1, vec_size)*0, color='black')
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].set_ylim([-0.05, 1.5])
        plt.show()

    def place_field2(self, place_field_recovered):
        X = np.linspace(0, self.map_size, self.vec_size)
        Z = np.zeros((self.N, self.vec_size))
        for i, x in enumerate(X):
            input_act = place_field_recovered[i]
            Z[:, i] = self.M.T @ input_act
        n = int(self.N ** 0.5)
        fig, ax = plt.subplots(n, n, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                ax[i, j].plot(X, Z[n * i + j])
                ax[i, j].plot(np.linspace(0, 1, vec_size), np.linspace(0, 1, vec_size) * 0, color='black')
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
                ax[i, j].set_ylim([-0.05, 0.25])
        plt.show()


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 128
    dt = 0.8
    N_h = 64
    X = np.linspace(0, map_size, vec_size)
    #rewardA = scipy.stats.norm.pdf(X, loc=0.7, scale=0.1) / 40 #((0.7 < X) & (X < 0.8)) * 0.1
    #rewardB = scipy.stats.norm.pdf(X, loc=0.3, scale=0.1) / 40 # ((0.2 < X) & (X < 0.3)) * 0.1
    lam=10
    Y1 = np.exp(-lam * (X - 0.7))
    rewardA1 = np.where(X > 0.7, Y1, 0)
    rewardA1 = rewardA1 / rewardA1.max()
    Y2 = np.exp(lam * (X - 0.7))
    rewardA2 = np.where(X < 0.7, Y2, 0)
    rewardA2 = rewardA2 / rewardA2.max()

    Y1 = np.exp(-lam * (X - 0.3))
    rewardB1 = np.where(X > 0.3, Y1, 0)
    rewardB1 = rewardB1 / rewardB1.max()
    Y2 = np.exp(lam * (X - 0.3))
    rewardB2 = np.where(X < 0.3, Y2, 0)
    rewardB2 = rewardB2 / rewardB2.max()
    N_reward = 300

    agent_params = {
        "init_pos": 0.0,
        "max_velocity": 0.1,
        "vec_size": vec_size,
        "map_size": map_size,
        "T": T,
        "dt": dt
    }
    sparse_coding_params = {
        "N_e": 600,
        "N_h": N_h,
        "vec_size": vec_size,
        "rewardA1": rewardA1,
        "rewardA2": rewardA2,
        "rewardB1": rewardB1,
        "rewardB2": rewardB2,
        "N_reward": N_reward,
        "dt": dt,
        "grid_files": "../bin/grid1d_{}_{}.npy".format(vec_size, 0.28)
    }
    pmap_params = {
        "vec_size": vec_size,
        "map_size": map_size,
        "N": N_h,
        "eta": 0.02,
        "gamma": 0.5
    }


    agent = Agent(agent_params)
    sparse_coding = Sparse_Coding(sparse_coding_params)
    pmap = Pmap(pmap_params)

    #fig, ax = plt.subplots()
    for nt in range(1, T):
        print(nt)
        t = dt * nt
        r = agent.one_step(nt, T)
        if nt <= 10000:
            context = 'A'
        else:
            context = 'B'
        pre_s_h, s_h = sparse_coding.one_step(r, context=context)
        pmap.learning_M(pre_s_h, s_h)
        if nt % 10000 == 0:
            place_field_recovered = sparse_coding.calc_place_field(context=context)
            # place fieldをmultiで表示する関数
            sparse_coding.imshow_place_field(
                sparse_coding.place_field_recovered,
                'sparse_coding_1d_{}'.format(N_h),
                display="multi"
            )
            # 密度表示
            sparse_coding.show_place_center(place_field_recovered, context=context)
            pmap.place_field2(place_field_recovered)
