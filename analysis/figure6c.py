import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
sns.set()

base_dir = Path('../data/Frontiers/figure6_full')
step_per_episode_list_full = []
for file in sorted(base_dir.glob("*.pkl")):
    print(file)
    with open(file, 'rb') as f:
        figure6_full = pickle.load(f)
    step_per_episode_list_full.append(figure6_full['step_per_episode_list'])

base_dir = Path('../data/Frontiers/figure6_ablation')
step_per_episode_list_ablation = []
for file in sorted(base_dir.glob("*.pkl")):
    print(file)
    with open(file, 'rb') as f:
        figure6_full = pickle.load(f)
    step_per_episode_list_ablation.append(figure6_full['step_per_episode_list'])

step_per_episode_list_full = np.array(step_per_episode_list_full)
print(step_per_episode_list_full.shape)
step_per_episode_list_ablation = np.array(step_per_episode_list_ablation)
print(step_per_episode_list_ablation.shape)


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(step_per_episode_list_full.mean(axis=0))
ax.plot(step_per_episode_list_ablation.mean(axis=0))
ax.set_xlabel("episodes", fontfamily="Helvetica Neue", fontsize=24)
ax.set_ylabel("average number of steps", fontfamily="Helvetica Neue", fontsize=24)
plt.tight_layout()
plt.show()

print(step_per_episode_list_full[:, 900:].mean(axis=0).mean())
print(step_per_episode_list_full[:, 900:].mean(axis=0).std())

print(step_per_episode_list_ablation[:, 900:].mean(axis=0).mean())
print(step_per_episode_list_ablation[:, 900:].mean(axis=0).std())

step_per_episode_mean = [
    step_per_episode_list_full[:, 900:].mean(axis=0).mean(),
    step_per_episode_list_ablation[:, 900:].mean(axis=0).mean()
]
step_per_episode_std = [
    step_per_episode_list_full[:, 900:].mean(axis=0).std(),
    step_per_episode_list_ablation[:, 900:].mean(axis=0).std()
]

fig, ax = plt.subplots(figsize=(4, 6))
ax.bar([0, 1], step_per_episode_mean, yerr=step_per_episode_std,
       width=0.7, edgecolor=['black', 'red'], linewidth=3, facecolor='None',
       capsize=3, error_kw={'linewidth': 2.0, 'capthick': 2.0})
ax.set_xticks([0, 1])
ax.set_xticklabels(['full', 'ablation'])
ax.set_ylabel("step per episode after learning", fontfamily="Helvetica Neue", fontsize=24)
ax.tick_params(axis='x', labelsize=24, bottom=False)
ax.tick_params(axis='y', width=2, labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.tight_layout()
plt.show()