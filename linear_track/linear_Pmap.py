import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from copy import deepcopy
import seaborn as sns

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()
        self.dt = params["dt"]  # 時間刻み幅
        self.map_size = params["map_size"] # 一次元トラックのサイズ
        self.max_velocity = params["max_velocity"] # 1sで進める最大の距離
        self.velocity = self.rng.uniform(0, self.max_velocity)
        self.pre_pos = params["init_pos"] # 初期位置
        self.pos = (self.pre_pos + self.velocity * self.dt) % self.map_size

    def one_step(self):
        self.pre_pos = self.pos
        self.velocity = self.rng.uniform(0, self.max_velocity)
        self.pos += self.velocity * self.dt
        self.pos = self.pos % self.map_size

class Pmap():
    def __init__(self, params):
        self.N = params["N"]
        self.vec_size = params["vec_size"]  # 1024
        self.map_size = params["map_size"]  # 1

        self.M = np.identity(self.N)
        self.eta = params['eta']
        self.gamma = params['gamma']
        self.create_fig = False

    def learning_M(self, pre_pos, pos):
        x_t1 = self.input_activity(pre_pos)
        x_t2 = self.input_activity(pos)

        dM = self.eta * x_t1.reshape((-1, 1)) * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)
        # 上下同じコード
        """dM = np.zeros(self.M.shape)
        for s in range(self.N):
            dM[s, :] += self.eta * x_t1[s] * (x_t1 + self.gamma * self.M.T @ x_t2 - self.M.T @ x_t1)"""
        self.M += dM

    def input_activity(self, pos):
        input_act = np.zeros(self.N)
        for j in range(self.N):
            input_act[j] = self.norm(pos, j / self.N)
        input_act = input_act / np.sum(input_act)
        return input_act

    def norm(self, pos, loc, scale=0.1):
        dist = np.min([(pos - loc) % 1, (loc - pos) % 1])
        #print(pos, loc, dist)
        return np.exp(- dist**2 / (2 * scale**2))

    def show_M(self):
        if not self.create_fig:
            self.fig, self.ax = plt.subplots()
            self.create_fig = True
            self.ax.set_aspect("equal")
        self.ax.imshow(self.M)
        plt.pause(0.1)

    def place_field(self):
        n = int(self.N**0.5)
        fig, ax = plt.subplots(n, n, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                #pmap_place_field[:, n * i + j] = (pmap_place_field[:, n * i + j] - np.min(pmap_place_field[:, n * i + j])) / (np.max(pmap_place_field[:, n * i + j]) - np.min(pmap_place_field[:, n * i + j]))
                place_field = self.M# - np.identity(self.N)
                ax[i, j].plot(place_field[:, n * i + j])
                #ax[i, j].plot(np.linspace(0, 1, vec_size), np.linspace(0, 1, vec_size)*0, color='black')
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].set_ylim([-0.05, 1.5])
        plt.show()

    def place_field2(self):
        X = np.linspace(0, self.map_size, self.vec_size)
        Z = np.zeros((self.N, self.vec_size))
        for i, x in enumerate(X):
            input_act = self.input_activity(x)
            Z[:, i] = self.M.T @ input_act
        n = int(self.N ** 0.5)
        fig, ax = plt.subplots(n, n, figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                ax[i, j].plot(X, Z[n*i+j])
                ax[i, j].plot(np.linspace(0, 1, vec_size), np.linspace(0, 1, vec_size)*0, color='black')
                ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
                ax[i, j].set_ylim([-0.05, 1.0])
        plt.show()



if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 128
    dt = 1.0
    N_h = 25

    agent_params = {
        "init_pos": 0.0,
        "max_velocity": 0.1,
        "vec_size": vec_size,
        "map_size": map_size,
        "T": T,
        "dt": dt
    }
    pmap_params = {
        "vec_size": vec_size,
        "map_size": map_size,
        "N": N_h,
        "eta": 0.1,
        "gamma": 0.9
    }

    agent = Agent(agent_params)
    pmap = Pmap(pmap_params)

    #fig, ax = plt.subplots()
    for nt in range(1, T):
        print("{} {:.3f} {:.3f}".format(nt, agent.pre_pos, agent.pos))
        t = dt * nt
        agent.one_step()
        pmap.learning_M(agent.pre_pos, agent.pos)
        if nt % 5000 == 0:
            #pmap.show_M()
            pmap.place_field2()

