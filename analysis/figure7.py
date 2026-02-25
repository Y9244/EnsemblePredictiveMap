import numpy as np
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def show_place_center_distance_mec_lec(
    place_centers_mec, place_center_index_mec,
    place_centers_lec, place_center_index_lec,
    base_dir='Frontiers', name="Figure7D"): # place_center: (N, 2)

    # 全ペア間距離を計算
    place_centers_mec = place_centers_mec[place_center_index_mec, :]
    dist_matrix_mec = cdist(place_centers_mec, place_centers_mec)  # shape: (N, N)

    place_centers_lec = place_centers_lec[place_center_index_lec, :]
    dist_matrix_lec = cdist(place_centers_lec, place_centers_lec)  # shape: (N, N)

    # 自己距離を無限大にして除外
    np.fill_diagonal(dist_matrix_mec, np.inf)
    np.fill_diagonal(dist_matrix_lec, np.inf)

    # 各点の最近傍との距離を取得
    min_distances_mec = np.min(dist_matrix_mec, axis=1)  # shape: (N,)
    min_distances_lec = np.min(dist_matrix_lec, axis=1)  # shape: (N,)

    kde_mec = gaussian_kde(min_distances_mec)
    kde_lec = gaussian_kde(min_distances_lec)


    fig, ax = plt.subplots(figsize=(10, 6))

    x_min = np.min([min_distances_mec.min(), min_distances_lec.min()])
    x_max = np.max([min_distances_mec.max(), min_distances_lec.max()])
    x_values = np.linspace(x_min, x_max, 100)
    kde_mec_values = kde_mec(x_values)
    kde_lec_values = kde_lec(x_values)
    # 描画
    ax.plot(x_values, kde_mec_values, color='tab:blue')
    ax.plot(x_values, kde_lec_values, color='tab:orange')
    ax.fill_between(x_values, kde_mec_values, alpha=0.3, color='tab:blue')
    ax.fill_between(x_values, kde_lec_values, alpha=0.3, color='tab:orange')

    ax.scatter(min_distances_mec, np.zeros_like(min_distances_mec), color='tab:blue', s=10)
    ax.scatter(min_distances_lec, np.zeros_like(min_distances_lec), color='tab:orange', s=10)

    ax.set_xlabel('nearest place center distance [m]', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    plt.tight_layout()
    if name is not None:
        plt.savefig(os.path.join("../figs", f"{base_dir}/{name}.png"), dpi=300)
    else:
        plt.show()
    plt.close()

def figure7a(object_position):
    fig, ax = plt.subplots(figsize=(6, 6))
    for o_pos in object_position:
        ax.add_patch(patches.Circle(xy=o_pos, radius=0.02, fc='white', ec='black'))
    ax.set_aspect("equal")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.savefig(os.path.join("../figs", f"Frontiers/Figure7A.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    with open('../data/Frontiers/figure7_mec.pkl', 'rb') as f:
        figure7_mec = pickle.load(f)
    with open('../data/Frontiers/figure7_lec.pkl', 'rb') as f:
        figure7_lec = pickle.load(f)

    object_position = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    figure7a(object_position)

    sparse_coding_mec = figure7_mec['sparse_coding']
    pmap_mec = figure7_mec['pmap']
    agent_mec = figure7_mec['agent']
    visualize_map_mec = figure7_mec['visualize_map']

    sparse_coding_lec = figure7_lec['sparse_coding']
    pmap_lec = figure7_lec['pmap']
    agent_lec = figure7_lec['agent']
    visualize_map_lec = figure7_lec['visualize_map']

    sc_place_fields_mec = visualize_map_mec.compute_sc_place_fields(sparse_coding_mec)
    place_centers_mec, place_center_index_mec = visualize_map_mec.compute_place_centers_in_square_arena(
        sc_place_fields_mec, return_index=True
    )

    visualize_map_mec.show_place_density_in_square_arena(
        place_centers_mec, place_center_index_mec,
        reward_positions=object_position,
        reward_or_object='object', base_dir='Frontiers', name='Figure7B')

    sc_place_fields_lec = visualize_map_lec.compute_sc_place_fields(sparse_coding_lec)
    place_centers_lec, place_center_index_lec = visualize_map_lec.compute_place_centers_in_square_arena(
        sc_place_fields_lec, return_index=True
    )
    visualize_map_lec.show_place_density_in_square_arena(
        place_centers_lec, place_center_index_lec,
        reward_positions=object_position,
        reward_or_object='object', base_dir='Frontiers', name='Figure7C')

    show_place_center_distance_mec_lec(
        place_centers_mec, place_center_index_mec,
        place_centers_lec, place_center_index_lec,
        base_dir='Frontiers', name="Figure7D"
    )


