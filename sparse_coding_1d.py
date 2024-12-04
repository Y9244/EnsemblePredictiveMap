import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import pickle, os
import copy
from scipy.optimize import curve_fit
# from lmfit import Parameters, Minimizer, report_fit

class Agent:
    def __init__(self, params):
        self.rng = np.random.default_rng()
        self.max_velocity = params["max_velocity"]
        self.velocity = None
        self.pre_pos = params["init_pos"]
        self.pos = params["init_pos"]
        self.vec_size = params["vec_size"]
        self.map_size = params["map_size"]
        self.T = params["T"]
        self.dt = params["dt"]
        self.pos_history = np.zeros(self.T)

    def one_step(self, nt, progress=False):
        if progress:
            print("\r{:.2f}%".format(nt / self.T * 100), end="")
        self.velocity = self.rng.uniform(0, self.max_velocity)
        #self.pre_pos = self.pos
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
        self.E = np.reshape(self.E, [self.N_e, -1]).T # (128, 600)

        self.n_iter = 200
        self.s_h_max = 10
        self.s_e = np.zeros(self.N_e)
        self.s_h = self.rng.random(self.N_h)
        self.u_h = self.rng.standard_normal(self.N_h)
        self.I_N_h = np.eye(self.N_h)
        self.R = np.zeros(self.N_e)

        self.eta = 3e-2
        self.A = self.rng.standard_normal((self.N_e, self.N_h))
        self.A = self.A / np.linalg.norm(self.A, axis=0, keepdims=True)

        self.place_fields = np.zeros((self.N_h, self.vec_size**2))

    def one_step(self, r, return_s_h=False):
        self.sparse_coding(r, self.u_h, self.s_h)
        if return_s_h:
            return self.s_h


    def sparse_coding(self, r, u_h, s_h, learning=True):
        # r: (128, )
        self.s_e = self.E.T @ r  # (600, 128), (128,) -> (600,)
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

    def calc_place_field(self, nt, save=True):
        K = int(1e5)
        X_recover = np.zeros((self.vec_size, K))
        loc_index = self.rng.integers(0, self.vec_size, K)
        X_recover[loc_index, range(K)] = 1
        S, U = self.sparse_coding(
            X_recover,
            self.u_h.reshape([-1, 1]),
            self.s_h.reshape([-1, 1]),
            learning=False)
        self.place_field_recovered = (X_recover @ S.T) / np.tile(np.sum(S, axis=1), (self.vec_size, 1))
        print(self.place_field_recovered.shape)
        """if save:
            with open("Lian_place_field/place_field_{}_{}.pkl".format(self.N_h, nt), 'wb') as f:
                pickle.dump(self.place_field_recovered, f)"""

    def imshow_place_field(self, place_field_recovered, name):
        n = 6
        X = np.linspace(0, 1, self.vec_size)
        for i in range(int(self.N_h / n ** 2)):
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            for j in range(n):
                for k in range(n):
                    i_cell = i * n ** 2 + j * n + k
                    ax.plot(X, place_field_recovered.T[i_cell])
            #ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.set_ylim([-0.05, 0.3])
            ax.set_aspect("equal")
            fig.suptitle(name)
            plt.show()
            plt.close()

def save_variable(dirname, X, params):
    model_name = params[0]
    step = params[1]
    with open(os.path.join(dirname, "{}_{}.pkl".format(model_name, step)), "wb") as f:
        pickle.dump(X, f)

if __name__ == "__main__":
    T = int(100_000)
    map_size = 1.0
    vec_size = 128
    dt = 0.8

    agent_params = {
        "init_pos": 0.0,
        "max_velocity": 0.05,
        "vec_size": vec_size,
        "map_size": map_size,
        "T": T,
        "dt": dt
    }
    sparse_coding_params = {
        "N_e": 600,
        "N_h": 36,
        "vec_size": vec_size,
        "dt": dt,
        "grid_files": "bin/grid1d_{}_{}.npy".format(vec_size, 0.28)
    }

    agent = Agent(agent_params)
    sparse_coding = Sparse_Coding(sparse_coding_params)

    for nt in range(T):
        print(nt)
        t = dt * nt
        r = agent.one_step(nt)
        sparse_coding.one_step(r)

    save_variable("bin", sparse_coding, ["sc_1d", sparse_coding.N_h])

