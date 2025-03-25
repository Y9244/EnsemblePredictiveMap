import numpy as np
import matplotlib.pyplot as plt
from linear_Pmap import Pmap

N = 6
vec_size = 1000
map_size = 1

pmap_params = {
        "vec_size": vec_size,
        "map_size": map_size,
        "N": N,
        "eta": 0.01,
        "gamma": 0.8
    }
pmap = Pmap(pmap_params)

X = np.linspace(0, map_size, vec_size)
norm_input = np.zeros((vec_size, N))
for i, x in enumerate(X):
    norm_input[i] = pmap.input_activity(x)

fig, ax = plt.subplots()
for i in range(N):
    ax.plot(X, norm_input[:, i], label='place cell {}'.format(i+1))
ax.set_xticks([i*0.1 for i in range(11)])
ax.set_xlim([0, map_size])
plt.legend(loc='upper right')
plt.show()