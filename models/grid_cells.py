import numpy as np
from itertools import product
from config.default_config import GridCellConfig

class GridCell:
	def __init__(self, config: GridCellConfig):
		self.dim = config.dim
		self.N_lam = config.N_lam
		self.N_theta = config.N_theta
		self.N_x = config.N_x
		self.N_y = config.N_y
		self.N_e = self.N_lam * self.N_theta * self.N_x * self.N_y

		min_lam = config.min_lam

		self.grid_spacing = [min_lam * 1.42**(i-0) for i in range(self.N_lam)]
		self.grid_orientation = [(i / self.N_theta) * np.pi / 3.0 for i in range(self.N_theta)]
		self.grid_phase_x = [i / self.N_x for i in range(self.N_x)]
		self.grid_phase_y = [i / self.N_y for i in range(self.N_y)]

		# self.param_grid: (N_e, 4)
		self.param_grid = np.array(list(product(
			self.grid_spacing,
			self.grid_orientation,
			self.grid_phase_x,
			self.grid_phase_y))
		)

		self.u = np.array([[np.cos(2*np.pi*0 + 0), np.sin(2*np.pi*0 + 0)],
		                   [np.cos(2*np.pi*1 + 0), np.sin(2*np.pi*1 + 0)],
		                   [np.cos(2*np.pi*2 + 0), np.sin(2*np.pi*2 + 0)]])

	"""
	def _compute_single_cell_activity(self, position, lam, theta, phase):
		activity = 0
		for j in [1, 2, 3]:
			uj = np.array([
				np.cos(2 * np.pi * j / 3 + theta),
				np.sin(2 * np.pi * j / 3 + theta)
			])
			activity += np.cos((4 * np.pi) / (np.sqrt(3) * lam) * (uj @ (position - phase)))
		activity = (2 / 3) * (activity / 3 + 0.25)
		return np.clip(activity, 0, 1)

	def calc_grid_activity(self, position):
		grid_cell_activity = np.zeros(self.N_e)
		if isinstance(position, float):
			position = np.array([position, 0.25])
		i = 0
		for lam in self.grid_spacing:
			for theta in self.grid_orientation:
				for x0 in self.grid_phase_x:
					for y0 in self.grid_phase_y:
						phase = np.array([x0 * lam, y0 * lam])
						grid_cell_activity[i] = self._compute_single_cell_activity(position, lam, theta, phase)
						i += 1
		grid_cell_activity = np.clip(grid_cell_activity, 0, 1)

		return grid_cell_activity.reshape(-1, self.N_e)
	"""

	def calc_grid_activity(self, position):
		if isinstance(position, float):
			position = np.array([position, 0.25])

		# パラメータ展開
		lam = self.param_grid[:, 0]  # shape: (N_e,)
		theta = self.param_grid[:, 1]
		x0 = self.param_grid[:, 2]
		y0 = self.param_grid[:, 3]
		phase = np.stack([x0 * lam, y0 * lam], axis=1)  # shape: (N_e, 2)

		# 3つの uj ベクトルを一括で生成
		j_vals = np.array([1, 2, 3])  # shape: (3,)
		angles = 2 * np.pi * j_vals[:, None] / 3 + theta[None, :]  # shape: (3, N_e)
		uj = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # shape: (3, N_e, 2)

		# (position - phase) に対する uj の内積
		delta = position[None, :] - phase  # shape: (N_e, 2)
		dot = np.einsum('jne,ne->jn', uj, delta)  # shape: (3, N_e)

		# 発火パターン計算
		factor = (4 * np.pi) / (np.sqrt(3) * lam)  # shape: (N_e,)
		cosine = np.cos(dot * factor[None, :])  # shape: (3, N_e)
		activity = (2 / 3) * (cosine.mean(axis=0) + 0.25)  # shape: (N_e,)
		return np.clip(activity, 0, 1)
