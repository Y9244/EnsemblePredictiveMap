import numpy as np
import random, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import deque
from tqdm import tqdm
from typing import Tuple
from entmax import entmax15
from entmax import entmax_bisect
from config.default_config import RLConfig


class RL:
    def __init__(self, config: RLConfig):
        self.dim = config.dim
        self.rng = np.random.default_rng()
        self.max_step = config.max_step
        self.reward_num = config.reward_num
        self.N_h = config.N_hip

        self.reward_enabled = config.reward_enabled
        self.eta_R = config.eta_R
        self.R = np.zeros((config.N_hip, self.reward_num))
        self.V = 0
        self.gamma = config.gamma

        self.reward_weights = np.ones(self.reward_num)
        self.reward_memories = [deque(maxlen=16) for _ in range(self.reward_num)]
        episode_memory_size = 5
        self.one_episode = []
        self.episode_memory = [deque(maxlen=episode_memory_size) for _ in range(self.reward_num)]

        # ------- random -----
        self.theta = self.rng.normal()

        # ------- MLP -------
        if config.actor_mode == 'MLP':
            self.actor = ActorMLP(input_dim=config.input_dim, max_step=self.max_step, dim=self.dim)
            #self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.eta_mlp, weight_decay=1e-6)
            self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.eta_mlp)

    "-------------------- action methods --------------------"
    def act_random_1d(self):
        action = self.rng.uniform(self.max_step/2
                                  , self.max_step)
        return action

    def act_random_2d(self, mode='random'):
        omega = self.rng.normal()
        self.theta += omega
        if mode == 'random':
            self.theta = (self.theta % 2 * np.pi)
        elif mode == 'upper':
            self.theta = (self.theta % np.pi)
        elif mode == 'bottom':
            self.theta = (self.theta % np.pi) - np.pi
        elif mode == 'right':
            self.theta = (self.theta % np.pi) - np.pi / 2
        elif mode == 'left':
            self.theta = (self.theta % np.pi) + np.pi / 2

        action = [0, 0]
        action[0] = self.max_step * (np.sin(self.theta + omega) - np.sin(self.theta)) / omega
        action[1] = self.max_step * (- np.cos(self.theta + omega) + np.cos(self.theta)) / omega
        return np.array(action)

    def act_random_2d_torch(self, x: np.ndarray) -> Tuple[Normal, Normal]:
        state_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        self.dx_dist, self.dy_dist = self.actor(state_tensor)  # MLPを呼び出す（勾配経路あり）

        action = torch.tensor(self.act_random_2d(), dtype=torch.float32)
        self.action = torch.clamp(action, min=-self.max_step, max=self.max_step)
        delta_pos = self.action

        return delta_pos.detach().numpy()

    def act_from_pmap(self, p_h: np.ndarray) -> np.ndarray:
        """
        :param p_h: [N_h,]
        :return: [2,] = dx, dx
        """
        if self.dim == 1:
            self.dx_dist = self.get_action_dist_1d(p_h)
            self.action = self.dx_dist.sample().squeeze(0)
            self.action = torch.clamp(self.action, min=0.01)
        elif self.dim == 2:
            self.dx_dist, self.dy_dist = self.get_action_dist_2d(p_h)
            self.action = torch.stack([self.dx_dist.sample().squeeze(0), self.dy_dist.sample().squeeze(0)], dim=-1).squeeze(
                0)
        self.action = torch.clamp(self.action, min=-2*self.max_step, max=2*self.max_step)

        delta_pos = self.action
        return delta_pos.detach().numpy()

    def get_action_dist_1d(self, x:np.ndarray) -> Normal:
        """
        :param x: [N_h,]
        :return: torch.distributions.normal.Normal
        """
        state_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        dx_dist = self.actor(state_tensor)
        return dx_dist

    def get_action_dist_2d(self, x: np.ndarray) -> Tuple[Normal, Normal]:
        """
        :param x: [N_h,]
        :return: tuple of torch.distributions.normal.Normal
        """
        state_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        dx_dist, dy_dist = self.actor(state_tensor)
        return dx_dist, dy_dist

    "-------------------- learning methods --------------------"

    def learning_reward(self, s_h: np.ndarray, reward_vector: np.ndarray,
                        R_replay=True, n_sample=8) -> np.ndarray:
        """
        s_h: (N_h,)
        reward_vector: (self.reward_num, )  # 報酬エリアごとのスカラー報酬値
        """
        s_h_normalized = s_h / (np.linalg.norm(s_h) + 1e-8)  # 念のためゼロ除算回避

        # 即時更新
        self.R += self.eta_R * (np.outer(s_h_normalized, reward_vector) - self.R)

        if R_replay:
            # 報酬がある要素のインデックス（≠0）
            nonzero_indices = np.where(reward_vector != 0)[0]

            # それぞれのエリアに記録
            for i in nonzero_indices:
                self.reward_memories[i].append((copy.deepcopy(s_h_normalized), copy.deepcopy(reward_vector)))

            # 各エリアからランダムにサンプリング
            for i, mem in enumerate(self.reward_memories):
                sample = random.sample(mem, min(len(mem), n_sample))
                for s_h_normalized, r_vec in sample:
                    self.R += self.eta_R * (np.outer(s_h_normalized, r_vec) - self.R)
        return self.R

    def learning_policy(self, reward_vector: np.ndarray,
                        pre_s_h: np.ndarray, s_h: np.ndarray,
                        pre_p_h: np.ndarray, p_h: np.ndarray,
                        R_replay=True, ablation=False, episode_i=None, t=None) -> None:
        """
        :param reward_vector: [reward_num,]
        :param pre_s_h, s_h, pre_p_h, p_h: [N_h,]
        :return:
        """

        self.learning_reward(s_h, reward_vector, R_replay=R_replay)
        if ablation:
            reward, td_error = self.get_td_error(pre_s_h, s_h, reward_vector, return_reward=True)
        else:
            reward, td_error = self.get_td_error(pre_p_h, p_h, reward_vector, return_reward=True)

        if episode_i is None:
            print(f"t: {t}, ", end='')
        else:
            print(f"(episode_i, t): ({episode_i}, {t}), ", end="")

        if self.dim == 1:
            log_prob = self.dx_dist.log_prob(self.action)
            dx = self.action
            print(f"action: {dx:6.3f}, ", end="")
        elif self.dim == 2:
            log_prob = self.dx_dist.log_prob(self.action[0]) + self.dy_dist.log_prob(self.action[1])
            dx, dy = self.action.squeeze(0).tolist()
            print(f"action: [{dx:6.3f}, {dy:6.3f}], ", end="")
        else:
            raise Exception('RL_module.learning_policy(*): dim must be 1 or 2.')

        loss = self.backward_step(td_error, log_prob)

        print(f'reward: {reward:3.1f}, TD Error: {td_error.item():5.3f}, loss: {loss.item():5.3f}')

        """if save_step:
            self.one_episode.append(
                [pre_p_h.copy(), p_h.copy(), copy.deepcopy(reward_vector), self.action.detach().clone()])"""

    def backward_step(self, td_error: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        loss = -td_error * log_prob
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_td_error(self, pre_p_h: np.ndarray, p_h: np.ndarray,
                     r_vec: np.ndarray, return_reward=False):
        reward = self.reward_weights @ r_vec
        V_t1 = (self.reward_weights @ self.R.T) @ pre_p_h  # ((N_r) @ (N_r, N_h)) @ (N_h,)
        V_t2 = (self.reward_weights @ self.R.T) @ p_h
        td_error = reward + self.gamma * V_t2 - V_t1
        td_error = torch.tensor(td_error, dtype=torch.float32)
        if return_reward:
            return reward, td_error
        else:
            return td_error

    "-------------------- log methods --------------------"

    "-------------------- unused methods --------------------"
    """def PEReplay(self, reward_vector: np.ndarray, n_sample=1) -> None:
        :param reward_vector: [reward_num,]
        :param n_sample: int
        
        print("PEReplay")
        nonzero_indices = np.where(reward_vector != 0)[0]
        for i in nonzero_indices:
            self.episode_memory[i].append(copy.deepcopy(self.one_episode))
        self.one_episode = []

        all_samples = []
        for i, mem in enumerate(self.episode_memory):
            all_samples += random.sample(mem, min(len(mem), n_sample))

        for one_episode in tqdm(all_samples):
            for pre_p_h, p_h, r_vec, action in one_episode[:-20:-1]:
                td_error = self.get_td_error(pre_p_h, p_h, r_vec)
                dx_dist, dy_dist = self.get_action_dist_2d(p_h)
                self.backward_step(td_error, dx_dist, dy_dist, action)"""


class ActorMLP(nn.Module):
    def __init__(self, input_dim, max_step, dim):
        super().__init__()
        hidden_dim = 128
        self.max_step = max_step
        self.dim = dim
        self.flatten = nn.Flatten()

        if dim == 1:
            self.simple_mlp = SimpleMLP(input_dim, hidden_dim, max_step)
        elif dim == 2:
            self.popvec_mlp = PopVectorMLP(input_dim, max_step)
        else:
            raise Exception("ActorMLP must be 1 or 2.")

        # self.resnet = ResNet(input_dim, hidden_dim1, hidden_dim2, max_step)

    def forward(self, state):
        x = self.flatten(state)
        if self.dim == 1:
            return self.simple_mlp(x)
        elif self.dim == 2:
            return self.popvec_mlp(x)

    def forward_mu(self, state):
        with torch.no_grad():
            x = self.flatten(state)
            if self.dim == 1:
                return self.simple_mlp.forward_mu(x)
            elif self.dim == 2:
                return self.popvec_mlp.forward_mu(x)

class PopVectorMLP(nn.Module):
    def __init__(self, input_dim, max_step):
        super().__init__()
        num_direction = 32
        self.max_step = max_step
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_direction),
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x)
        mu = self.direction_to_vector(x) * self.max_step
        dx_dist = torch.distributions.Normal(mu[:, 0], 0.02)
        dy_dist = torch.distributions.Normal(mu[:, 1], 0.02)
        return dx_dist, dy_dist

    def forward_mu(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        mu = self.direction_to_vector(x) * self.max_step
        dx_mu, dy_mu = mu.chunk(2, dim=-1)
        return torch.stack([
            dx_mu.squeeze(-1) * self.max_step,
            dy_mu.squeeze(-1) * self.max_step],
            dim=-1).numpy()

    def direction_to_vector(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Converts 32-directional weights into a 2D movement vector.

        Args:
            weights (torch.Tensor): shape (direction_num,), each entry is the weight for one direction

        Returns:
            torch.Tensor: shape (2,), representing x and y displacement
        """
        direction_num = weights.shape[-1]
        angles = torch.tensor(np.linspace(0, 2 * math.pi, direction_num, endpoint=False), device=weights.device,
                              dtype=torch.float32)  # 0 to 2π in 32 steps
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # (D, 2)
        return torch.matmul(weights, directions)  # (B, D) x (D, 2) → (B, 2)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_step, dim=1):
        super().__init__()
        self.min_step = 0.02
        self.max_step = max_step
        self.dim = dim
        if dim == 1:
            self.fc = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),
            )
            #self.init_weights(self.fc)
        elif dim == 2:
            raise Exception("We do not implement 2-d version of Simple MLP.")

    def init_weights(self, *modules):
        for model in modules:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = ((self.fc(x) + 1) / 2 * (self.max_step - self.min_step)) + self.min_step
        dx_dist = torch.distributions.Normal(x[:, 0], 0.01)
        return dx_dist

    def forward_mu(self, x):
        # [0.01, max_step]
        x = ((self.fc(x) + 1) / 2 * (self.max_step - self.min_step)) + self.min_step
        x = torch.clamp(x, min=self.min_step)
        return x.numpy()


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, max_step):
        super().__init__()
        self.max_step = max_step
        self.shortcut = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim2),
            nn.Tanh()
        )

        # Residual Block
        self.resnet = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh()
        )

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim2, 2),
            nn.Tanh()
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim2, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.resnet(x)

        mu = self.mean_head(x1 + x2) * self.max_step  # 平均にTanh＋スケーリング
        log_std = self.log_std_head(x1 + x2) - 4.0
        std = torch.exp(log_std)

        dx_dist = torch.distributions.Normal(mu[:, 0], std[:, 0])
        dy_dist = torch.distributions.Normal(mu[:, 1], std[:, 1])
        return dx_dist, dy_dist

    def forward_mu(self, x):
        x1 = self.shortcut(x)
        x2 = self.resnet(x)

        mu = self.mean_head(x1 + x2) * self.max_step  # 平均にTanh＋スケーリング

        dx_mu, dy_mu = mu.chunk(2, dim=-1)
        return torch.stack([
            dx_mu.squeeze(-1) * self.max_step,
            dy_mu.squeeze(-1) * self.max_step],
            dim=-1).numpy()

