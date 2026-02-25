import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def show_predictive_accuracy(s_h, p_h, display='bar', significance=True, base_dir='Frontiers', ):
    s_h, p_h = np.array(s_h[20000:]).T, np.array(p_h[20000:]).T
    N_h, T = s_h.shape
    autocorrs_s_h = np.zeros(N_h)
    autocorrs_p_h = np.zeros(N_h)
    for i in range(N_h):
        x_s_h = s_h[i, :-1]
        y_s_h = s_h[i, 1:]
        autocorrs_s_h[i] = np.corrcoef(x_s_h, y_s_h)[0, 1]
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
        plt.savefig('predictive_accuracy.png')
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
        plt.savefig("../figs/Frontiers/Figure4E.png", dpi=300)
        plt.show()
        plt.close()