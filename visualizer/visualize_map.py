import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde, ttest_ind
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
import torch
from typing import Union
from tqdm import tqdm

from models.grid_cells import GridCell
from models.boundary_cell import BoundaryCell
from models.reward_cell import RewardCell
from models.sparse_coding import SparseCoding
from envs.square_arena import SquareArenaEnv
from envs.linear_track import LinearTrackEnv
import seaborn as sns
sns.set()
import os

class VisualizeMap:
	def __init__(self, env: Union[LinearTrackEnv, SquareArenaEnv],
	             grid_cell: GridCell,
	             boundary_cell: BoundaryCell = None,
	             reward_cell: RewardCell = None,
	             resolution=32):
		self.env = env
		self.grid_cell = grid_cell
		self.boundary_cell = boundary_cell
		self.resolution = resolution

		self.base_path = '/Users/yk9244/thesis/src/Main/sparse_predictive_map/figs'

		print('--- generate meshgrid position ---')
		self.positions = self._generate_meshgrid_positions()
		print('--- compute MEC fields ---')
		self.grid_fields = self._precompute_grid_activities() # (N_pos, N_e)

		if self.env.dim == 2 and self.boundary_cell is not None:
			self.boundary_fields = self.boundary_cell.calc_border_activity(self.positions) # (N_pos, N_border)
			self.grid_fields = np.concatenate([self.grid_fields, self.boundary_fields], axis=1)

		elif self.env.dim == 2 and self.boundary_cell is None:
			pass
		else:
			raise Exception("The dimension of environment must be 2 if you set boundary_cell.")


		if reward_cell != None:
			print('--- compute LEC fields ---')
			self.reward_cell = reward_cell
			if self.env.dim == 1:
				self.LEC_fields = self.reward_cell.calc_reward_cell_activity_in_linear_track(self.positions)
			elif self.env.dim == 2:
				self.LEC_fields = self.reward_cell.calc_reward_cell_activity_in_square_arena(self.positions)

			self.grid_fields = np.concatenate([self.grid_fields, self.LEC_fields], axis=1)

		# self.grid_fields = self.standard_scalar_grid_fields(self.grid_fields)

	"---------- predefined methods ----------"
	def _generate_meshgrid_positions(self):
		if self.env.dim == 1:
			return np.linspace(0, self.env.env_size, self.resolution) # shape: [N_pos]
		elif self.env.dim == 2:
			x = np.linspace(0, self.env.env_size, self.resolution)
			y = np.linspace(0, self.env.env_size, self.resolution)
			xx, yy = np.meshgrid(x, y)
			return np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape: [N_pos, 2]

	def _precompute_grid_activities(self):
		grid_fields = np.array([self.grid_cell.calc_grid_activity(pos) for pos in tqdm(self.positions)], dtype=np.float32)  # shape: [N_pos, N_grid]
		return grid_fields.reshape(self.resolution**self.env.dim, -1)

	def update_reward_cell(self, reward_cell: RewardCell):
		self.reward_cell = reward_cell
		if self.env.dim == 1:
			self.LEC_fields = self.reward_cell.calc_reward_cell_activity_in_linear_track(self.positions)
		elif self.env.dim == 2:
			self.LEC_fields = self.reward_cell.calc_reward_cell_activity_in_square_arena(self.positions)
		self.grid_fields[:, -self.LEC_fields.shape[1]:] = self.LEC_fields

	def standard_scalar_grid_fields(self, grid_fields):
		for i in range(grid_fields.shape[0]):
			s_m = grid_fields[i, :self.grid_cell.N_e]
			s_b = grid_fields[i, self.grid_cell.N_e:]
			Eg = np.mean(np.sum(s_m ** 2))
			Eb = np.mean(np.sum(s_b ** 2)) + 1e-8
			alpha = np.sqrt(Eg / Eb)  # これで平均エネルギーを一致

			grid_fields[i] = np.concatenate([s_m, alpha * s_b])
		return grid_fields



	"---------- compute methods ----------"

	def compute_sc_place_fields(self, sc: SparseCoding):
		sc_place_fields = sc.sparse_coding(self.grid_fields)  # shape: [N_pos, N_h]
		return sc_place_fields / np.linalg.norm(sc_place_fields, axis=1, keepdims=True)

	def compute_pmap_place_fields(self, pmap, sc_place_fields): # shape: [N_pos, N_h]
		pmap_place_field = pmap.predictive_map(sc_place_fields)
		return pmap_place_field # shape: [N_pos, N_h]

	def compute_place_centers_in_linear_track(self, place_fields, return_index=None):
		"""
		place_fields.shape: [N_pos, N_h]
		"""
		place_center = np.zeros(place_fields.shape[1]) # shape: (N_h,)
		place_center_index = []
		for i_cell in range(place_fields.shape[1]):
			place_cell = place_fields[:, i_cell]
			if np.max(place_cell) > 0.5:
				place_center_index.append(i_cell)
			xc = np.argmax(place_cell)
			place_center[i_cell] = xc / self.resolution * self.env.env_size
		if return_index:
			return place_center, place_center_index
		else:
			return place_center

	def compute_place_centers_in_square_arena(self, place_fields, return_index=None): # shape: [N_pos, N_h]
		place_center = np.zeros((place_fields.shape[1], 2)) # shape: [N_h, 2]
		place_center_index = []
		for i_cell in range(place_fields.shape[1]):
			place_cell = np.abs(place_fields[:, i_cell].reshape(self.resolution, self.resolution))
			if np.max(place_cell) > 0.5:
				place_center_index.append(i_cell)
			yc, xc = np.unravel_index(np.argmax(place_cell), place_cell.shape)
			place_center[i_cell] = xc / self.resolution * self.env.env_size, yc / self.resolution * self.env.env_size
		if return_index:
			return place_center, place_center_index
		else:
			return place_center

	def compute_Vmap(self, pmap_place_field, R, reward_weights):
		return (pmap_place_field @ R) @ reward_weights # shape: [N_pos]

	def compute_policy_map_in_linear_track(self, agent, pmap_place_field):
		# pmap_place_field.shape: [N_pos, N_h]
		state_tensor = torch.tensor(pmap_place_field, dtype=torch.float32)
		policy_map = agent.actor.forward_mu(state_tensor) # [N_pos]
		return policy_map[:, 0]

	def compute_policy_map_in_square_arena(self, agent, pmap_place_field):
		# pmap_place_field.shape: [N_pos, N_place]
		state_tensor = torch.tensor(pmap_place_field, dtype=torch.float32)
		policy_map = agent.actor.forward_mu(state_tensor) # [N_pos, 2]
		policy_map = policy_map / np.max(np.linalg.norm(policy_map, axis=1))
		return policy_map.T

	def compute_peak_mass_diff(self, place_field):
		X = np.linspace(0, self.env.env_size, self.resolution)
		# 最大値の位置（インデックス）
		peak_y_index, peak_x_index = np.unravel_index(np.argmax(place_field), place_field.shape)
		peak = np.array([X[peak_y_index], X[peak_x_index]])
		# 重心の位置（連続値）
		com = center_of_mass(place_field)
		com = np.array([com[0] * self.env.env_size / self.resolution, com[1] * self.env.env_size / self.resolution])  # スケール合わせ

		# 距離
		delta = np.abs(peak - com)  # 各軸の差分
		delta = np.minimum(delta, self.env.env_size - delta)  # 最小周期距離を選ぶ
		return np.linalg.norm(delta)

	def compute_skewness_angle(self, place_field, place_center_index):
		skewness_array = np.zeros(len(place_center_index))
		for i, i_cell in enumerate(place_center_index):
			one_place_field = place_field[:, i_cell].reshape(self.resolution, self.resolution)

			X = np.linspace(0, self.env.env_size, self.resolution)
			# 最大値の位置（インデックス）
			peak_y_index, peak_x_index = np.unravel_index(np.argmax(one_place_field), one_place_field.shape)
			peak = np.array([X[peak_y_index], X[peak_x_index]])
			# 重心の位置（連続値）
			com = center_of_mass(one_place_field)
			com = np.array([com[0] * self.env.env_size / self.resolution, com[1] * self.env.env_size / self.resolution])  # スケール合わせ
			direction = com - peak
			angles = np.arctan2(direction[0], direction[1])  # [-π, π]
			angles = np.mod(angles, 2 * np.pi)
			skewness_array[i] = angles
		return skewness_array

	def compute_peak_points(self, place_field, place_center_index):
		peak_points = np.zeros((len(place_center_index), 2))
		for i, i_cell in enumerate(place_center_index):
			one_place_field = place_field[:, i_cell].reshape(self.resolution, self.resolution)

			X = np.linspace(0, self.env.env_size, self.resolution)
			# 最大値の位置（インデックス）
			peak_y_index, peak_x_index = np.unravel_index(np.argmax(one_place_field), one_place_field.shape)

			peak_points[i] = np.array([X[peak_x_index], X[peak_y_index]])
		return peak_points

	def compute_toroidal_center_of_mass(self, place_fields, place_center_index):
		"""
        Parameters:
            place_fields: np.ndarray of shape (H*W, N)
                Each column is the flattened 2D receptive field of one cell.
            env_size: float
                The size of the environment (e.g., 1.0 means coordinates are in [0, 1)).

        Returns:
            centers: np.ndarray of shape (N, 2)
                The toroidal center of mass (x, y) for each place cell.
        """
		place_fields = place_fields[:, place_center_index]

		# 2D coordinates for each grid cell (flattened)
		x_vals = np.linspace(0, self.env.env_size, self.resolution, endpoint=False)
		y_vals = np.linspace(0, self.env.env_size, self.resolution, endpoint=False)
		X, Y = np.meshgrid(x_vals, y_vals)  # shape: (H, W)

		X_flat, Y_flat = X.ravel(), Y.ravel()  # shape: (resolution**2,)
		N = len(place_center_index)

		centers = np.zeros((N, 2))

		for i in range(N):
			weights = place_fields[:, i]
			if np.sum(weights) == 0:
				centers[i] = np.array([np.nan, np.nan])  # avoid division by zero
				continue

			# X-direction
			theta_x = (X_flat / self.env.env_size) * 2 * np.pi
			sin_x = np.sum(weights * np.sin(theta_x))
			cos_x = np.sum(weights * np.cos(theta_x))
			angle_x = np.arctan2(sin_x, cos_x)
			cx = (angle_x % (2 * np.pi)) / (2 * np.pi) * self.env.env_size

			# Y-direction
			theta_y = (Y_flat / self.env.env_size) * 2 * np.pi
			sin_y = np.sum(weights * np.sin(theta_y))
			cos_y = np.sum(weights * np.cos(theta_y))
			angle_y = np.arctan2(sin_y, cos_y)
			cy = (angle_y % (2 * np.pi)) / (2 * np.pi) * self.env.env_size

			centers[i] = np.array([cx, cy])

		return centers

	def compute_toroidal_vector(self, place_fields, place_center_index):
		"""
        Compute the shortest vector from mass_points to peak_points on a 2D torus.

        Parameters:
            peak_points: np.ndarray of shape (N, 2)
            mass_points: np.ndarray of shape (N, 2)
        Returns:
            delta_vectors: np.ndarray of shape (N, 2)
                Toroidally-corrected vectors from mass_points to peak_points
        """
		peak_points = self.compute_peak_points(place_fields, place_center_index)
		mass_points = self.compute_toroidal_center_of_mass(place_fields, place_center_index)

		delta = mass_points - peak_points  # naive difference
		delta = (delta + self.env.env_size / 2) % self.env.env_size - self.env.env_size / 2
		return delta

	def compute_vec2angle(self, delta_vectors):
		angles = np.arctan2(delta_vectors[:, 1], delta_vectors[:, 0])
		angles = np.mod(angles, 2 * np.pi)
		return angles

	"---------- show methods ----------"

	def show_place_fields_in_linear_track(self, place_fields, base_dir="linear_track", name=None):
		n = int(place_fields.shape[1] ** 0.5)
		X = np.linspace(0, self.env.env_size, self.resolution)
		fig, ax = plt.subplots(n, n, figsize=(8, 6))
		fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.05, hspace=0.05)
		for i in range(n):
			for j in range(n):
				i_cell = i * n + j
				ax[i, j].plot(X, place_fields[:, i_cell])
				ax[i, j].set_ylim([-0.05, np.max(place_fields) + 0.05])
				for rc, rs, rv in self.env.reward_info:
					ax[i, j].fill_between([rc - rs/2, rc + rs/2], y1=[0, 0], y2=np.ones_like(2) * np.max(place_fields),
					                color='red', alpha=0.7)
				#ax[i, j].axvline(x=2.5, color='red')
				ax[i, j].tick_params(
					labelbottom=False, labelleft=False, labelright=False, labeltop=False,
					bottom=False, left=False, right=False, top=False
				)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_place_fields_in_square_arena(self, place_fields, base_dir='square_arena', name=None):
		peak_points = self.compute_peak_points(place_fields, np.arange(place_fields.shape[1]))
		vector = self.compute_toroidal_vector(place_fields, np.arange(place_fields.shape[1]))
		max_cell_index = np.argmax(np.max(place_fields, axis=0))
		place_fields = place_fields.reshape(self.resolution, self.resolution, -1)
		n = int(place_fields.shape[2] ** 0.5)
		x = np.linspace(0, self.env.env_size, self.resolution)
		y = np.linspace(0, self.env.env_size, self.resolution)
		X, Y = np.meshgrid(x, y)
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(n, n, left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
		for i in range(n):
			for j in range(n):
				i_cell = i * n + j
				ax = fig.add_subplot(gs[i, j])
				im = ax.contourf(X, Y, place_fields[:, :, i_cell], cmap="jet",
				                 levels=100, vmin=0.0, vmax=np.max(place_fields))
				#ax.arrow(x=peak_points[i_cell, 0], y=peak_points[i_cell, 1], dx=vector[i_cell, 0], dy=vector[i_cell, 1])
				if max_cell_index == i_cell:
					max_im = im
				ax.tick_params(
					labelbottom=False, labelleft=False, labelright=False, labeltop=False,
					bottom=False, left=False, right=False, top=False
				)
				ax.set_xlim([0, self.env.env_size])
				ax.set_ylim([0, self.env.env_size])
				ax.set_aspect('equal')
		cbar_ax = fig.add_axes((0.87, 0.1, 0.02, 0.8))  # [left, bottom, width, height]
		fig.colorbar(max_im, cax=cbar_ax)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_peak_mass_dist(self, nsc_place_field, pmap_place_field, place_center_index, display='bar', significance=True, base_dir="Frontiers", name=None):
		# pmap_place_field: (vec_size**2, N_h)
		# print(len(place_center_index))
		nsc_peak_mass_vector = self.compute_toroidal_vector(nsc_place_field, place_center_index)
		pmap_peak_mass_vector = self.compute_toroidal_vector(pmap_place_field, place_center_index)
		nsc_peak_mass_dist = np.linalg.norm(nsc_peak_mass_vector, axis=1)
		pmap_peak_mass_dist = np.linalg.norm(pmap_peak_mass_vector, axis=1)
		if display == 'hist':
			bin_width = 0.025
			bins = np.arange(np.min(np.concatenate([nsc_peak_mass_dist, pmap_peak_mass_dist])),
			                 np.max(np.concatenate([pmap_peak_mass_dist, pmap_peak_mass_dist])) + bin_width, bin_width)
			fig, ax = plt.subplots(figsize=(8, 4))
			ax.hist(nsc_peak_mass_dist, bins=bins, color='black', alpha=0.5)
			ax.hist(pmap_peak_mass_dist, bins=bins, color='red', alpha=0.5)
			#plt.savefig("peak_mass_distance.png")
			plt.show()
			plt.close()
		elif display == 'bar':
			skewness_mean = nsc_peak_mass_dist.mean(), pmap_peak_mass_dist.mean()
			skewness_std = nsc_peak_mass_dist.std(), pmap_peak_mass_dist.std()
			t_stat, p_value = ttest_ind(pmap_peak_mass_dist, nsc_peak_mass_dist, equal_var=False, alternative='greater')
			fig, ax = plt.subplots(figsize=(4, 6))
			ax.bar([0, 1], skewness_mean, yerr=skewness_std,
			       width=0.7, edgecolor=['black', 'red'], linewidth=3, facecolor='None',
			       capsize=3, error_kw={'linewidth': 2.0, 'capthick': 2.0})
			if significance:
				t_stat, p_value = ttest_ind(pmap_peak_mass_dist, nsc_peak_mass_dist, equal_var=False, alternative='greater')
				print("skew")
				print(f"s_h: {nsc_peak_mass_dist.mean():.3f}±{nsc_peak_mass_dist.std():.3f}")
				print(f"p_h: {pmap_peak_mass_dist.mean():.3f}±{pmap_peak_mass_dist.std():.3f}")
				print(f"t={t_stat:.3f}, p={p_value:.3g}")
				ax.plot([0, 0, 1, 1], [0.22, 0.24, 0.24, 0.22], linewidth=2.5, color='black')
				if p_value < 0.01:
					ax.scatter(0.5, 0.255, marker=(6, 2), color='black', s=160)
			ax.set_ylabel(r"|peak - mass| (m)", fontfamily="Helvetica Neue", fontsize=24)
			ax.set_xticks([0, 1])
			ax.set_xticklabels([r'$s_h$', r'$p_h$'])
			ax.set_yticks([0.0, 0.1, 0.2, 0.3])
			ax.tick_params(axis='x', labelsize=24, bottom=False)
			ax.tick_params(axis='y', width=2, labelsize=20)
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['left'].set_linewidth(2)
			ax.spines['bottom'].set_linewidth(2)
			plt.tight_layout()
			if name is None:
				plt.show()
			else:
				plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
			plt.close()

	def show_peak_mass_vector(self, nsc_place_field, pmap_place_field, place_center_index, base_dir="Frontiers", name=None):
		nsc_peak_mass_vector = self.compute_toroidal_vector(nsc_place_field, place_center_index)
		pmap_peak_mass_vector = self.compute_toroidal_vector(pmap_place_field, place_center_index)
		nsc_peak_mass_angle = self.compute_vec2angle(nsc_peak_mass_vector)
		pmap_peak_mass_angle = self.compute_vec2angle(pmap_peak_mass_vector)

		num_bins = 36
		bin_edges = np.linspace(0.0, 2 * np.pi, num_bins + 1)
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
		width = 2 * np.pi / num_bins

		hist_nsc, _ = np.histogram(nsc_peak_mass_angle, bins=bin_edges)
		hist_nsc = hist_nsc / hist_nsc.sum()
		hist_pmap, _ = np.histogram(pmap_peak_mass_angle, bins=bin_edges)
		hist_pmap = hist_pmap / hist_pmap.sum()

		fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
		ax.bar(bin_centers, hist_nsc, width=width, bottom=0.0, align='center', alpha=0.75, facecolor='black')
		ax.bar(bin_centers, hist_pmap, width=width, bottom=0.0, align='center', alpha=0.75, facecolor='red')
		ax.set_rgrids([0.0, 0.1, 0.2, 0.3, 0.4])  # 任意の半径値でメモリを設定
		ax.tick_params(labelsize=18, pad=5)
		ax.set_yticklabels([])
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()

	def show_place_center_distance(self, place_center, place_center_index, mode='hist', base_dir='square_arena', name=None):
		# place_center: (N, 2)
		# 全ペア間距離を計算
		place_center = place_center[place_center_index, :]
		dist_matrix = cdist(place_center, place_center)  # shape: (N, N)

		# 自己距離を無限大にして除外
		np.fill_diagonal(dist_matrix, np.inf)

		# 各点の最近傍との距離を取得
		min_distances = np.min(dist_matrix, axis=1)  # shape: (N,)
		# print(len(min_distances))

		fig, ax = plt.subplots(figsize=(8, 6))

		if mode == 'kde':
			kde = gaussian_kde(min_distances)
			x_min, x_max = min_distances.min(), min_distances.max()
			x_min, x_max = 0.06, 0.19
			x_values = np.linspace(x_min, x_max, 100)
			kde_values = kde(x_values)
			# 描画
			ax.plot(x_values, kde_values, color='tab:blue')
			ax.fill_between(x_values, kde_values, alpha=0.3, color='tab:blue')
		elif mode == 'hist':
			ax.hist(min_distances, bins=15)
		else:
			raise Exception('mode must be kde or hist')
		ax.scatter(min_distances, np.zeros_like(min_distances), color='black', s=10)

		ax.set_xlabel('nearest place center distance [m]')
		ax.set_ylabel('Density')
		plt.tight_layout()
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_place_centers_in_square_arena(self, place_centers, place_center_index, reward_positions=None, base_dir="square_arena" ,name=None):
		fig, ax = plt.subplots()
		ax.scatter(place_centers[place_center_index, 0], place_centers[place_center_index, 1], color='black', s=50)
		if reward_positions:
			for r_pos in reward_positions:
				ax.add_patch(patches.Circle(xy=r_pos, radius=0.02, fc='white', ec='black'))
		ax.set_aspect("equal")
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_place_density_in_linear_track(self, place_centers, place_center_index, r_pos=None, base_dir='linear_track', name=None):
		"""
		:param place_centers.shape: (N_h,)
		:param place_center_index: list of index
		:param reward_positions.shape:
		"""
		place_centers = place_centers[place_center_index]
		X = np.linspace(0, self.env.env_size, self.resolution)

		kde = gaussian_kde(place_centers, bw_method=0.1)  # Scott's rule または 'silverman' でも可
		Y = kde(X)

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.fill_between(X, X*0, Y, alpha=0.4)
		ax.scatter(place_centers, np.zeros_like(place_centers), color='black')
		if r_pos is None:
			for rc, rs, rv in self.env.reward_info:
				ax.fill_between([rc - rs/2, rc + rs/2], y1=[0, 0], y2=np.ones(2)*np.max(Y)*1.05, color='red', alpha=0.7)
		else:
			ax.fill_between([r_pos[0] - r_pos[1] / 2, r_pos[0] + r_pos[1] / 2], y1=[0, 0], y2=np.ones(2) * np.max(Y) * 1.05, color='red',
			                alpha=0.7)
		ax.set_xlim([0, self.env.env_size])
		ax.tick_params(labelsize=20)
		ax.set_xticks(np.arange(0, self.env.env_size + 1))
		if name is None:
			plt.show()
		else:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		plt.close()

	def show_place_center_scatter_2context(self, place_centers_A, place_center_index_A,
	                                       place_centers_B, place_center_index_B,
	                                       r_pos_A, r_pos_B, base_dir='Frontiers', name=None):

		place_center_index = list(set(place_center_index_A) & set(place_center_index_B))
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.scatter(place_centers_A[place_center_index], place_centers_B[place_center_index], color='black', s=50, zorder=50)
		ax.plot([0, self.env.env_size], [0, self.env.env_size], color='gray')
		#ax.axvline(x=r_pos_A, ymin=0, ymax=self.env.env_size, color='red')
		#ax.axhline(y=r_pos_B, xmin=0, xmax=self.env.env_size, color='red')
		ax.fill_between([0, self.env.env_size],
		                y1=[r_pos_B[0] - r_pos_B[1] / 2, r_pos_B[0] - r_pos_B[1] / 2],
		                y2=[r_pos_B[0] + r_pos_B[1] / 2, r_pos_B[0] + r_pos_B[1] / 2], color='red')
		ax.fill_between([r_pos_A[0] - r_pos_A[1] / 2, r_pos_A[0] + r_pos_A[1] / 2],
		                y1=[0, 0],
		                y2=[self.env.env_size, self.env.env_size], color='red')
		#ax.fill_between([0, self.env.env_size], y1=[0, 0], y2=[], color='red')
		ax.set_xlim([0.0, self.env.env_size])
		ax.set_ylim([0.0, self.env.env_size])
		ax.set_xticks(np.arange(0, self.env.env_size+1, 1))
		ax.set_yticks(np.arange(0, self.env.env_size+1, 1))
		ax.tick_params(axis='both', labelsize=18)  # 両軸の目盛り文字サイズを14に
		ax.set_xlabel(r'$\mathrm{peak\ activity\ location\ during\ A_{end}\ [m]}$', fontsize=20)
		ax.set_ylabel(r'$\mathrm{peak\ activity\ location\ during\ A_{mid}\ [m]}$', fontsize=20)
		ax.set_aspect('equal')
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_place_density_in_square_arena(self, place_centers, place_center_index,
	                                       reward_positions=None, reward_or_object=None, base_dir='square_arena', name=None):
		L = self.env.env_size
		x, y = place_centers[place_center_index].T

		# --- 周期的拡張データを作る --------------------------------
		shifts = [-L, 0, L]
		xs, ys = [], []
		for dx in shifts:
			for dy in shifts:
				xs.append(x + dx)
				ys.append(y + dy)
		xs = np.concatenate(xs)
		ys = np.concatenate(ys)

		kde = gaussian_kde(np.vstack([xs, ys]), bw_method=0.1)

		# --- 評価グリッド --------------------------------
		x_grid = np.linspace(0, L, self.resolution)
		y_grid = np.linspace(0, L, self.resolution)
		X, Y = np.meshgrid(x_grid, y_grid)
		positions = np.vstack([X.ravel(), Y.ravel()])

		# 密度評価 → shape: (resolution^2,) → reshape
		Z = kde(positions).reshape(X.shape)

		# --- 描画 --------------------------------
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(1, 1, left=0.03, right=0.83, bottom=0.1, top=0.9)
		ax = fig.add_subplot(gs[0, 0])
		vmin = np.min([0, np.min(Z)])
		vmax = np.max(Z)
		levels = np.linspace(vmin, vmax, 100)
		im = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)

		ax.scatter(place_centers[place_center_index, 0],
		           place_centers[place_center_index, 1],
		           color='black', s=50)
		if reward_positions and reward_or_object == 'object':
			for r_pos in reward_positions:
				ax.add_patch(patches.Circle(xy=r_pos, radius=0.02, fc='white', ec='black'))
		elif reward_positions and reward_or_object == 'reward':
			self.draw_reward_regions(ax, self.env.reward_info, alpha=0.4)

		ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
		ax.set_aspect("equal")

		cbar_ax = fig.add_axes((0.85, 0.1, 0.03, 0.8))
		cbar = fig.colorbar(im, cax=cbar_ax)
		# vmin, vmax = np.min(Z), np.max(Z)
		cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		cbar.ax.tick_params(labelsize=18)

		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_reward_vector_in_linear_track(self, place_centers, place_center_index, reward_vector, base_dir='linear_track', name=None):
		fig = plt.figure(figsize=(8, 4))
		gs = gridspec.GridSpec(1, 1, left=0.03, right=0.83, bottom=0.1, top=0.9)
		ax = fig.add_subplot(gs[0, 0])
		im = ax.scatter(place_centers[place_center_index],
		                np.zeros_like(place_center_index),
		                c=np.sum(reward_vector[place_center_index], axis=1),
		                s=50, cmap='plasma', zorder=10
		                )
		for rc, rs, rv in self.env.reward_info:
			ax.fill_between([rc - rs/2, rc + rs/2], y1=[-1, -1], y2=[1, 1], color='red', alpha=0.7)
		cbar_ax = fig.add_axes((0.85, 0.1, 0.03, 0.8))  # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		#plt.tight_layout()
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_reward_vector_in_square_arena(self, place_centers, place_center_index, reward_vector, base_dir='square_arena', name=None):
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(1, 1, left=0.03, right=0.83, bottom=0.1, top=0.9)
		ax = fig.add_subplot(gs[0, 0])
		im = ax.scatter(place_centers[place_center_index, 0],
		                place_centers[place_center_index, 1],
		                c=np.sum(reward_vector[place_center_index], axis=1),
		                s=50, cmap='plasma', zorder=10
		                )
		self.draw_reward_regions(ax, self.env.reward_info)
		ax.set_aspect("equal")
		ax.tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False
		)
		ax.set_xlim([0, self.env.env_size])
		ax.set_ylim([0, self.env.env_size])
		cbar_ax = fig.add_axes((0.85, 0.1, 0.03, 0.8))  # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def draw_reward_regions(self, ax, reward_info, alpha=0.6):
		"""
        報酬・罰エリアを矩形としてax上に描画する。

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            描画対象のaxes
        reward_info : List[Tuple[Tuple[float, float], Tuple[float, float], float]]
            [(中心座標), (x幅, y幅), 報酬値] のリスト
        """
		for center, size, reward_value in reward_info:
			cx, cy = center
			sx, sy = size

			# 左下の座標へ変換
			x0 = cx - sx / 2
			y0 = cy - sy / 2

			# 色を報酬値で決定
			if reward_value > 0:
				color = 'red'
			elif reward_value < 0:
				color = 'blue'
			else:
				color = 'gray'

			# 矩形を追加
			rect = patches.Rectangle(
				(x0, y0), sx, sy,
				linewidth=0,
				edgecolor=color,
				facecolor=color,
				alpha=alpha,
				zorder = 5
			)
			ax.add_patch(rect)

	def show_value_map_in_linear_track(self, pmap_place_field, R, place_centers, place_center_index, base_dir='square_arena', name=None, alpha=0.7):
		V_map = pmap_place_field @ R  # shape: [N_pos]
		X = np.linspace(0, self.env.env_size, self.resolution)
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.scatter(
			place_centers[place_center_index],
			np.zeros_like(place_center_index),
			s=50, color='black', zorder=10)
		ax.plot(X, V_map, color="red", zorder=1)
		for rc, rs, rv in self.env.reward_info:
			ax.fill_between([rc - rs/2, rc + rs/2], y1=[0, 0], y2=np.ones_like(2)*np.max(V_map)*1.05, color='red', alpha=0.7)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_value_map_in_square_arena(self, pmap_place_field, R, place_centers, place_center_index, base_dir='square_arena', name=None, alpha=0.4):
		V_map = pmap_place_field @ R # shape: [N_pos]
		V_map = V_map.reshape(self.resolution, self.resolution)
		x = np.linspace(0, self.env.env_size, self.resolution)
		y = np.linspace(0, self.env.env_size, self.resolution)
		X, Y = np.meshgrid(x, y)
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(1, 1, left=0.03, right=0.83, bottom=0.1, top=0.9)
		ax = fig.add_subplot(gs[0, 0])
		ax.scatter(place_centers[place_center_index, 0], place_centers[place_center_index, 1], s=50, color='black', zorder=10)
		im = ax.contourf(X, Y, V_map, cmap="plasma", levels=100, vmin=np.min(V_map), vmax=np.max(V_map), zorder=1)
		self.draw_reward_regions(ax, self.env.reward_info, alpha=alpha)
		ax.set_aspect('equal')
		ax.tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False
		)
		cbar_ax = fig.add_axes((0.85, 0.1, 0.03, 0.8)) # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_policy_map_in_linear_track(self, policy_map, place_centers, place_center_index, base_dir='square_arena', name=None):
		X = np.linspace(0, self.env.env_size, self.resolution)
		fig, ax = plt.subplots(figsize=(9, 3))
		ax.scatter(
			place_centers[place_center_index],
			np.zeros_like(place_center_index),
			s=50, color='black', zorder=10)
		ax.plot(X, policy_map, color="red", zorder=1)
		for rc, rs, rv in self.env.reward_info:
			ax.fill_between([rc - rs/2, rc + rs/2], y1=[0, 0], y2=np.ones_like(2)*np.max(policy_map)*1.05, color='red', alpha=0.7)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_policy_map_in_linear_track_2context(self, policy_map_A, policy_map_B, r_pos_A, r_pos_B,
	                                             base_dir='square_arena', name=None):
		X = np.linspace(0, self.env.env_size, self.resolution)
		fig, ax = plt.subplots(2, 1, figsize=(9, 6))
		ax[0].plot(X, policy_map_A, color="black", linewidth=3, zorder=1)
		ax[1].plot(X, policy_map_B, color="black", linewidth=3, zorder=1)
		ax[0].fill_between([r_pos_A[0] - r_pos_A[1]/2, r_pos_A[0] + r_pos_A[1]/2],
		                   y1=[0, 0], y2=[0.105, 0.105], facecolor='red', edgecolor='red', alpha=0.7)
		ax[1].fill_between([r_pos_B[0] - r_pos_B[1] / 2, r_pos_B[0] + r_pos_B[1] / 2],
		                   y1=[0, 0], y2=[0.105, 0.105], facecolor='red', edgecolor='red', alpha=0.7)
		ax[0].plot([0, self.env.env_size], [0, 0], color='black')
		ax[1].plot([0, self.env.env_size], [0, 0], color='black')
		#ax[0].axis('off')
		#ax[1].axis('off')
		ax[0].tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False,
		)
		ax[1].tick_params(
			labelbottom=True, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False,
			labelsize=18
		)
		ax[0].set_xticks(np.arange(0, self.env.env_size+1))
		ax[1].set_xticks(np.arange(0, self.env.env_size + 1))
		for spine in ['top', 'left', 'right', 'bottom']:
			ax[0].spines[spine].set_visible(False)
			ax[1].spines[spine].set_visible(False)
		ax[1].set_xlabel('track position [m]', fontsize=24)
		ax[0].set_ylabel(r'$\mathrm{A_{end}}$', fontsize=32)
		ax[1].set_ylabel(r'$\mathrm{A_{mid}}$', fontsize=32)
		plt.tight_layout()
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()


	def show_policy_map_on_Vmap_in_square_arena(self, policy_map, value_map, base_dir='square_arena', name=None):
		X, Y = np.meshgrid(np.linspace(0, self.env.env_size, self.resolution),
		                   np.linspace(0, self.env.env_size, self.resolution))

		# 各位置の進行方向ベクトルを計算（単位ベクトル）
		H = policy_map[0].reshape(self.resolution, self.resolution) / 2
		V = policy_map[1].reshape(self.resolution, self.resolution) / 2

		# ベクトル場を描画
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(1, 1, left=0.03, right=0.83, bottom=0.1, top=0.9)
		ax = fig.add_subplot(gs[0, 0])
		vmin = np.min([0, np.min(value_map)])
		vmax = np.max(value_map)
		print(vmin, vmax)
		levels = np.linspace(vmin, vmax, 100)
		im = ax.contourf(X, Y, value_map.reshape(self.resolution, self.resolution), cmap='plasma', vmin=vmin, vmax=vmax, levels=levels, zorder=1)
		ax.quiver(X, Y, H, V, scale=20, color='deepskyblue', zorder=2)
		self.draw_reward_regions(ax, self.env.reward_info)
		ax.tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False
		)
		ax.set_aspect('equal')
		ax.invert_yaxis()  # imshowなどと整合をとるなら
		ax.set_xlim([0, self.env.env_size])
		ax.set_ylim([0, self.env.env_size])
		cbar_ax = fig.add_axes((0.85, 0.1, 0.03, 0.8)) # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		cbar.ax.tick_params(labelsize=18)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_activity(self, sc_place_field, pmap_place_field, base_dir='square_arena', name=None): # shape: [N_pos, N_place]

		fig, ax = plt.subplots(1, 2, figsize=(16, 8))
		ax[0].imshow(sc_place_field, cmap='jet')
		ax[1].imshow(pmap_place_field, cmap='jet')
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_figure4ac(self, place_fields, base_dir='Frontiers', name=None):
		# i_cell_list = [40 + i for i in range(10)]
		i_cell_list = [40, 41, 42, 43, 44, 45,
		               66, 52, 53, 54]

		relative_index = np.argmax(np.max(place_fields[:, i_cell_list], axis=0))
		max_cell_index = i_cell_list[relative_index]

		place_fields = place_fields.reshape(self.resolution, self.resolution, -1)

		x = np.linspace(0, self.env.env_size, self.resolution)
		y = np.linspace(0, self.env.env_size, self.resolution)
		X, Y = np.meshgrid(x, y)
		fig = plt.figure(figsize=(10, 4))
		nx, ny = 5, 2
		gs = gridspec.GridSpec(nrows=ny, ncols=nx, left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
		vmin = np.min(place_fields[:, :, i_cell_list])
		vmax = np.max(place_fields[:, :, i_cell_list])
		print(vmin, vmax)
		levels = np.linspace(0, vmax, 100)
		for i in range(ny):
			for j in range(nx):
				i_cell = i_cell_list[i * nx + j]
				ax = fig.add_subplot(gs[i, j])
				im = ax.contourf(X, Y, place_fields[:, :, i_cell], cmap="jet",
				                 levels=levels, vmin=0, vmax=vmax)
				if max_cell_index == i_cell:
					max_im = im
				ax.tick_params(
					labelbottom=False, labelleft=False, labelright=False, labeltop=False,
					bottom=False, left=False, right=False, top=False
				)
				ax.set_aspect('equal')
		cbar_ax = fig.add_axes((0.87, 0.1, 0.02, 0.8))  # [left, bottom, width, height]
		cbar = fig.colorbar(max_im, cax=cbar_ax)
		cbar.set_ticks([0.0, vmax/2, vmax])
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		cbar.ax.tick_params(labelsize=18)
		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_figure5ab(self, place_centers, place_center_index, R, V_map, base_dir='Frontiers', name=None):
		fig = plt.figure(figsize=(16, 8))  # 横長

		# Reward vector
		left, bottom, width, height = 0.02, 0.12, 0.38, 0.76
		cbar_width = 0.02
		cbar_wspace = 0.01
		ax_left = fig.add_axes([left, bottom, width, height]) # (left, bottom, width, height)

		im = ax_left.scatter(place_centers[place_center_index, 0],
		                place_centers[place_center_index, 1],
		                c=np.sum(R[place_center_index], axis=1),
		                s=50, cmap='plasma', zorder=10
		                )
		self.draw_reward_regions(ax_left, self.env.reward_info)
		ax_left.set_aspect("equal")
		ax_left.set_xlim([0, self.env.env_size])
		ax_left.set_ylim([0, self.env.env_size])
		ax_left.tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False
		)
		cbar_ax = fig.add_axes((left+width+cbar_wspace, bottom, cbar_width, height))  # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.set_ticks([i/2 * np.max(R[place_center_index]) for i in range(3)])
		#cbar.set_ticks([f"{i * np.max(V_map): .3f}" for i in range(3)])
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		cbar.ax.tick_params(labelsize=18)

		# Value map
		ax_right = fig.add_axes([left+0.5, bottom, width, height]) # (left, bottom, width, height)
		V_map = V_map.reshape(self.resolution, self.resolution)
		x = np.linspace(0, self.env.env_size, self.resolution)
		y = np.linspace(0, self.env.env_size, self.resolution)
		X, Y = np.meshgrid(x, y)
		levels = np.linspace(0.0, np.max(V_map), 100)
		im = ax_right.contourf(X, Y, V_map, cmap="plasma", levels=levels, vmin=0, vmax=np.max(V_map), zorder=1)
		self.draw_reward_regions(ax_right, self.env.reward_info, alpha=0.4)
		ax_right.set_aspect('equal')
		ax_right.tick_params(
			labelbottom=False, labelleft=False, labelright=False, labeltop=False,
			bottom=False, left=False, right=False, top=False
		)
		cbar_ax = fig.add_axes((left+0.5+width+cbar_wspace, bottom, cbar_width, height)) # (left, bottom, width, height)
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.set_ticks([i/2 * np.max(V_map) for i in range(3)])
		cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('% .3f'))
		cbar.ax.tick_params(labelsize=18)

		if name is not None:
			plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
		else:
			plt.show()
		plt.close()

	def show_predictive_accuracy(self, s_h, p_h, display='bar', significance=True, base_dir='Frontiers', name=None):
		s_h = np.array(s_h[10000:]).T # (T, N_h)
		p_h = np.array(p_h[10000:]).T # (T, N_h)
		N_h, T = s_h.shape
		autocorrs_s_h = np.zeros(N_h)
		autocorrs_p_h = np.zeros(N_h)
		for i in range(N_h):
			x_s_h = s_h[i, :-1]
			y_s_h = s_h[i, 1:]
			if np.std(x_s_h) > 1e-8 and np.std(y_s_h) > 1e-8:
				autocorrs_s_h[i] = np.corrcoef(x_s_h, y_s_h)[0, 1]
			else:
				autocorrs_s_h[i] = 0.0
			x_p_h = p_h[i, :-1]
			y_p_h = p_h[i, 1:]
			autocorrs_p_h[i] = np.corrcoef(x_p_h, y_p_h)[0, 1]
		if display == 'hist':
			bin_width = 0.05
			bins = np.arange(np.min(np.concatenate([autocorrs_s_h, autocorrs_p_h])),
			                 np.max(np.concatenate([autocorrs_s_h, autocorrs_p_h])) + bin_width, bin_width)
			fig, ax = plt.subplots(figsize=(8, 4))
			ax.hist(autocorrs_s_h, bins=bins, color='black', alpha=0.5)
			ax.hist(autocorrs_p_h, bins=bins, color='red', alpha=0.5)
			#plt.savefig('predictive_accuracy.png')
			plt.show()
			plt.close()
		elif display == 'bar':
			autocorrs_mean = autocorrs_s_h.mean(), autocorrs_p_h.mean()
			autocorrs_std = autocorrs_s_h.std(), autocorrs_p_h.std()
			fig, ax = plt.subplots(figsize=(4, 6))
			ax.bar([0, 1], autocorrs_mean, yerr=autocorrs_std,
			       width=0.7, edgecolor=['black', 'red'], linewidth=3, facecolor='None',
			       capsize=3, error_kw={'linewidth': 2.0, 'capthick': 2.0})
			if significance:
				t_stat, p_value = ttest_ind(autocorrs_p_h, autocorrs_s_h, equal_var=False, alternative='greater')
				print("autocorrelation")
				print(f"s_h: {autocorrs_s_h.mean():.3f}±{autocorrs_s_h.std():.3f}")
				print(f"p_h: {autocorrs_p_h.mean():.3f}±{autocorrs_p_h.std():.3f}")
				print(f"t={t_stat:.3f}, p={p_value:.3g}")
				ax.plot([0, 0, 1, 1], [0.75, 0.8, 0.8, 0.75], linewidth=2.5, color='black')
				if p_value < 0.01:
					ax.scatter(0.5, 0.85, marker=(6, 2), color='black', s=160)

			ax.set_ylabel("autocorrelation", fontfamily="Helvetica Neue", fontsize=24)
			ax.set_xticks([0, 1])
			ax.set_xticklabels([r'$s_h$', r'$p_h$'])
			ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
			ax.tick_params(axis='x', labelsize=24, bottom=False)
			ax.tick_params(axis='y', width=2, labelsize=20)
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['left'].set_linewidth(2)
			ax.spines['bottom'].set_linewidth(2)
			plt.tight_layout()
			if name is None:
				plt.show()
			else:
				plt.savefig(os.path.join(self.base_path, f"{base_dir}/{name}.png"), dpi=300)
			plt.close()

