import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

rng = np.random.default_rng()

N = 1000
lam = rng.normal(15, 5, (N, 2))
reward_pos = 0.7
vec_size = 1000

X = np.linspace(0, 1, vec_size)

L = np.zeros((2*N, vec_size))
for i, l in enumerate(lam):
    post = np.exp(-np.min(l) * (X - reward_pos)) - np.exp(-np.max(l) * (X - reward_pos))
    post = np.where(X > reward_pos, post, 0)
    post /= np.max(post)
    L[i] = post

    pre = np.exp(np.min(l) * (X - reward_pos)) - np.exp(np.max(l) * (X - reward_pos))
    pre = np.where(X < reward_pos, pre, 0)
    pre /= np.max(pre)
    L[N+i] = pre

fig, ax = plt.subplots()
ax.imshow(L)
plt.show()