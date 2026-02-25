import joblib
import matplotlib.pyplot as plt
import numpy as np

sc_place_fields = joblib.load('../data/sc_place_fields.pkl') # shape: [N_pos, N_place]
resolution = 64

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i_place, one_place_activity in enumerate(sc_place_fields):
    ax[0].cla()
    ax[1].cla()
    ax[0].plot(one_place_activity)
    ax[0].set_box_aspect(1)
    ax[0].set_ylim([0, 1])
    i = i_place // resolution
    j = i_place % resolution
    X = np.zeros((resolution, resolution))
    X[i, j] = 1
    ax[1].imshow(X, cmap='plasma')
    ax[1].set_box_aspect(1)
    plt.pause(0.01)
    plt.draw()
    #plt.show()
    #plt.close()
