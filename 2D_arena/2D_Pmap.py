import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from scipy.stats import norm
import seaborn as sns


# sns.set()

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()
        self.velocity = params["velocity"]  # 1sで進める最大の距離
        self.theta = params["theta"]
        self.pre_pos = params["init_pos"]  # 初期位置
        self.pos = deepcopy(self.pre_pos)
        delta_pos = self.delta_pos()
        self.pos[0] += delta_pos[0]
        self.pos[1] += delta_pos[1]
        self.map_size = params["map_size"]  # 2次元アリーナのサイズ
        self.dt = params["dt"]  # 時間刻み幅

    def one_step(self, nt, T, progress=False):
        if progress:
            print("\r{:.2f}%".format(nt / T * 100), end="")
        self.pre_pos = deepcopy(self.pos)
        delta_pos = self.delta_pos()
        self.pos[0] += delta_pos[0]
        self.pos[1] += delta_pos[1]
        self.pos[0] = self.pos[0] % self.map_size
        self.pos[1] = self.pos[1] % self.map_size

    def delta_pos(self):
        omega = self.rng.normal()
        self.theta += omega * dt
        self.theta = self.theta % np.pi - np.pi/2
        delta_pos = [0, 0]
        delta_pos[0] = self.velocity * (np.sin(self.theta + omega * dt) - np.sin(self.theta)) / omega
        delta_pos[1] = self.velocity * (- np.cos(self.theta + omega * dt) + np.cos(self.theta)) / omega
        return delta_pos

    def one_step_show(self, ax, vec_size):
        # plt.cla()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        """for i in range(vec_size):
            ax.axhline(i/vec_size, color='gray')
            ax.axvline(i/vec_size, color='gray')"""
        #ax.scatter(self.pos[0], self.pos[1], marker="x", color="black", zorder=100)

        c = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=0.02, fc='None', ec='black')
        ax.add_patch(c)
        #ax.invert_yaxis()
        plt.draw()
        plt.pause(0.1)

class Pmap():
    def __init__(self, params):
        self.N = params["N"]  # 100
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

    def show_M(self):
        if not self.create_fig:
            self.fig, self.ax = plt.subplots()
            self.create_fig = True
            self.ax.set_aspect("equal")
        self.ax.imshow(self.M)
        self.ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False,
            bottom=False, left=False, right=False, top=False
        )
        plt.show()
        plt.close()
        self.create_fig = False
        #plt.pause(0.1)

    def place_field(self):
        n = int(self.M.shape[0] ** 0.5)
        if not self.create_fig:
            self.fig, self.ax = plt.subplots(n, n, figsize=(8, 8))
            self.create_fig = True
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                place_field = self.M# - np.identity(self.N)
                self.ax[i, j].imshow(place_field[:, n*(9-i) + j].reshape(n, n), cmap='jet')
                #self.ax[i, j].text(n/2, n/2, "({}, {})".format((9-i), j), size=10, horizontalalignment="center", verticalalignment='center', color='white')
                self.ax[i, j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                self.ax[i, j].tick_params(bottom=False, left=False, right=False, top=False)
                self.ax[i, j].invert_yaxis()
        #self.ax[9, 0].tick_params(labelbottom=True, labelleft=True, labelright=True, labeltop=True)
        #self.ax[9, 0].tick_params(bottom=True, left=True, right=True, top=True)
        plt.show()
        plt.close()
        self.create_fig = False
        #plt.draw()
        #plt.pause(0.1)

    def place_field2(self):
        X, Y = np.linspace(0, self.map_size, self.vec_size), np.linspace(0, self.map_size, self.vec_size)
        Z = np.zeros((self.N, self.vec_size, self.vec_size))
        n = int(self.N ** 0.5)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                input_act = pmap.input_activity((x, y))
                Z[:, j, i] = self.M.T @ input_act
        X, Y = np.meshgrid(X, Y)
        if not self.create_fig:
            self.fig, self.ax = plt.subplots(n, n, figsize=(8, 8))
            self.create_fig = True
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(n):
            for j in range(n):
                self.ax[i, j].contourf(X, Y, Z[n * (9 - i) + j], cmap="jet", levels=100)
                self.ax[i, j].set_aspect("equal")
                self.ax[i, j].tick_params(
                    labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, left=False, right=False, top=False
                )
        plt.show()
        plt.close()


if __name__ == "__main__":
    T = int(1_000_000)
    map_size = 1.0
    vec_size = 32
    dt = 1.0
    N_h = 100

    agent_params = {
        "init_pos": np.array([0.5, 0.5]),
        "velocity": 0.1,
        "theta": 0.0,
        "vec_size": vec_size,
        "map_size": map_size,
        "dt": dt
    }
    pmap_params = {
        "N": N_h,
        "vec_size": vec_size,
        "map_size": map_size,
        "eta": 0.05,
        "gamma": 0.8
    }

    agent = Agent(agent_params)
    pmap = Pmap(pmap_params)

    for nt in range(1, T):
        print(nt)
        t = dt * nt
        agent.one_step(nt, T)

        pmap.learning_M(agent.pre_pos, agent.pos)
        if nt % 100000 == 0:
            pmap.show_M()
            pmap.place_field2()