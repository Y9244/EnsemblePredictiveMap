import joblib
import matplotlib.pyplot as plt
import numpy as np


def compute_place_centers_in_linear_track(place_fields, return_index=None):
    """
    place_fields.shape: [N_pos, N_h]
    """
    resolution = 512
    env_size = 4.0
    place_center = np.zeros(place_fields.shape[1])  # shape: (N_h,)
    place_center_index = []
    for i_cell in range(place_fields.shape[1]):
        place_cell = place_fields[:, i_cell]
        if np.max(place_cell) > 0.5:
            place_center_index.append(i_cell)
        xc = np.argmax(place_cell)
        place_center[i_cell] = xc / resolution * env_size
    if return_index:
        return place_center, place_center_index
    else:
        return place_center

r_pos_A = 3.66
r_pos_B = 1.66
sc_place_fields_A = joblib.load(f'../data/sc_place_fields_{r_pos_A}.pkl') # shape: [N_pos, N_place]
sc_place_fields_B = joblib.load(f'../data/sc_place_fields_{r_pos_B}.pkl') # shape: [N_pos, N_place]

place_center_A, place_center_index_A = compute_place_centers_in_linear_track(sc_place_fields_A, return_index=True)
place_center_B, place_center_index_B = compute_place_centers_in_linear_track(sc_place_fields_B, return_index=True)

place_center_A = place_center_A
place_center_B = place_center_B

print(place_center_index_A)
print(place_center_index_B)


fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(place_center_A, place_center_B, color='black', s=50, zorder=50)
ax.plot([0, 4], [0, 4], color='gray')
ax.axvline(x=r_pos_A, ymin=0, ymax=3.0, color='red')
ax.axhline(y=r_pos_B, xmin=0, xmax=3.0, color='red')
ax.set_xlim([0.0, 4.0])
ax.set_ylim([0.0, 4.0])
ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=18)  # 両軸の目盛り文字サイズを14に
ax.set_xlabel("COM during A[m]", fontsize=20)
ax.set_ylabel("COM during B[m]", fontsize=20)
ax.set_aspect('equal')
plt.show()
