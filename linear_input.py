import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 10
"""def norm(pos, loc):
    dist = np.min([(pos - loc) % N, (loc - pos) % N])
    return dist
for i in np.linspace(0, 10, 20):
    for j in np.linspace(0, 10, 20):
        print(i, j, norm(i, j))
exit()"""

fig, ax = plt.subplots()
X = np.linspace(0, 10, 1000)
norm_input = []
for i in range(10):
    norm_input.append(norm.pdf(X, loc=i, scale=0.5))
norm_input[0] = (norm_input[0] + norm.pdf(X, loc=10, scale=0.5))

for i in range(10):
    ax.plot(X, norm_input[i], label='place cell {}'.format(i))
ax.set_xticks([0, 1, 2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper right')
plt.show()