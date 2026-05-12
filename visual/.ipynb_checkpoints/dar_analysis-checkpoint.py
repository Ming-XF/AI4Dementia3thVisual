import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from scipy.integrate import simpson
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def calculate_band_power_efficient(time_series, hz, freq_bands,
                                   window_sec=4, return_relative=False):
    N, C, L = time_series.shape
    band_powers = {}

    nperseg = min(L, int(hz * window_sec))
    noverlap = nperseg // 2

    data_reshaped = time_series.reshape(-1, L)

    freqs, psd_matrix = signal.welch(
        data_reshaped,
        fs=hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        axis=-1
    )

    if return_relative:
        total_mask = (freqs >= 0.5) & (freqs <= 45)
        total_power = simpson(psd_matrix[:, total_mask],
                              freqs[total_mask], axis=-1)
        band_powers['total_power'] = total_power.reshape(N, C)

    for band_name, (f_low, f_high) in freq_bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        freq_range = freqs[freq_mask]
        psd_band = psd_matrix[:, freq_mask]

        abs_power = simpson(psd_band, freq_range, axis=-1)
        band_powers[band_name] = abs_power.reshape(N, C)

        if return_relative:
            rel_power = abs_power / total_power
            band_powers[band_name + '_relative'] = rel_power.reshape(N, C)

    return band_powers


def load_and_preprocess_data():
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    _, _, labels, _, _, r1, r2, r3, ts = data
    return labels, r1, r2, r3, ts


def compute_dar(band_powers):
    """Compute Delta/Alpha Ratio: DAR = delta_power / alpha_power."""
    delta = band_powers['delta']      # (N, C)
    alpha = band_powers['alpha']      # (N, C)
    dar = delta / (alpha + 1e-12)     # avoid div by zero
    return dar


def plot_dar_bar(dar_denoised, dar_original, labels, group_order,
                 label_map, output_dir="./"):
    """Plot grouped bar chart of DAR for denoised and original data."""
    group_data_denoised = {}
    group_data_original = {}

    for group_name in group_order:
        group_idx = label_map[group_name]
        mask = labels == group_idx

        # Average DAR across channels for each subject
        subj_dar_denoised = np.mean(dar_denoised[mask], axis=1)
        subj_dar_original = np.mean(dar_original[mask], axis=1)

        group_data_denoised[group_name] = subj_dar_denoised
        group_data_original[group_name] = subj_dar_original

    means_denoised = [np.mean(group_data_denoised[g]) for g in group_order]
    sems_denoised = [stats.sem(group_data_denoised[g]) for g in group_order]
    means_original = [np.mean(group_data_original[g]) for g in group_order]
    sems_original = [stats.sem(group_data_original[g]) for g in group_order]

    x = np.arange(len(group_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, means_denoised, width,
                   yerr=sems_denoised, capsize=5,
                   color='steelblue', edgecolor='black', linewidth=0.8,
                   label='Denoised EEG (r1)')

    bars2 = ax.bar(x + width / 2, means_original, width,
                   yerr=sems_original, capsize=5,
                   color='darkorange', edgecolor='black', linewidth=0.8,
                   label='Original EEG (ts)')

    ax.set_xlabel('Group', fontsize=13)
    ax.set_ylabel('DAR (Delta / Alpha)', fontsize=13)
    ax.set_title('Delta/Alpha Ratio (DAR) by Group: Denoised vs Original', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bar values
    for bar, mean, sem in zip(bars1, means_denoised, sems_denoised):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, mean, sem in zip(bars2, means_original, sems_original):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    # ANOVA for denoised
    f_val_d, p_val_d = stats.f_oneway(*[group_data_denoised[g] for g in group_order])
    # ANOVA for original
    f_val_o, p_val_o = stats.f_oneway(*[group_data_original[g] for g in group_order])

    ax.text(0.98, 0.95,
            f'Denoised ANOVA: F={f_val_d:.3f}, p={p_val_d:.2e}\n'
            f'Original ANOVA: F={f_val_o:.3f}, p={p_val_o:.2e}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/dar_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to {output_dir}/dar_bar_chart.png")


def plot_dar_comparison(dar_denoised, dar_original, labels, group_order,
                        label_map, output_dir="./"):
    """Plot denoised vs original DAR scatter for each subject, colored by group."""
    subj_dar_d = np.mean(dar_denoised, axis=1)
    subj_dar_o = np.mean(dar_original, axis=1)

    group_names_sorted = sorted(label_map.keys(), key=lambda k: label_map[k])
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names_sorted)))

    fig, ax = plt.subplots(figsize=(8, 8))

    for group_name, color in zip(group_names_sorted, colors):
        idx = label_map[group_name]
        mask = labels == idx
        ax.scatter(subj_dar_o[mask], subj_dar_d[mask],
                   c=[color], label=group_name, alpha=0.6, edgecolors='black',
                   linewidth=0.5, s=60)

    all_vals = np.concatenate([subj_dar_o, subj_dar_d])
    lim_min = all_vals.min() * 0.9
    lim_max = all_vals.max() * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, alpha=0.5,
            label='y = x')

    ax.set_xlabel('Original DAR (ts)', fontsize=13)
    ax.set_ylabel('Denoised DAR (r1)', fontsize=13)
    ax.set_title('DAR: Denoised vs Original by Subject', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(subj_dar_d, subj_dar_o)
    ax.text(0.98, 0.05,
            f'Paired t-test: t={t_stat:.3f}, p={p_val:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    fig.savefig(f'{output_dir}/dar_scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {output_dir}/dar_scatter_comparison.png")


def print_dar_statistics(group_data_denoised, group_data_original, group_order):
    """Print DAR statistics table and pairwise t-tests."""
    print("\n" + "=" * 70)
    print("DAR Statistics Summary")
    print("=" * 70)
    print(f"{'Group':<8} {'Denoised Mean':>14} {'Denoised SEM':>14} "
          f"{'Original Mean':>14} {'Original SEM':>14}")
    print("-" * 70)

    for g in group_order:
        d_mean = np.mean(group_data_denoised[g])
        d_sem = stats.sem(group_data_denoised[g])
        o_mean = np.mean(group_data_original[g])
        o_sem = stats.sem(group_data_original[g])
        print(f"{g:<8} {d_mean:14.4f} {d_sem:14.4f} {o_mean:14.4f} {o_sem:14.4f}")

    # Pairwise t-tests between groups (denoised)
    print("\n" + "-" * 70)
    print("Pairwise t-tests (Denoised DAR):")
    for i, g1 in enumerate(group_order):
        for g2 in group_order[i + 1:]:
            t, p = stats.ttest_ind(group_data_denoised[g1],
                                   group_data_denoised[g2])
            print(f"  {g1} vs {g2}: t={t:.3f}, p={p:.4f}")

    # Pairwise t-tests between groups (original)
    print("\nPairwise t-tests (Original DAR):")
    for i, g1 in enumerate(group_order):
        for g2 in group_order[i + 1:]:
            t, p = stats.ttest_ind(group_data_original[g1],
                                   group_data_original[g2])
            print(f"  {g1} vs {g2}: t={t:.3f}, p={p:.4f}")

    # Denoised vs Original paired within each group
    print("\nPaired t-tests (Denoised vs Original within group):")
    for g in group_order:
        t, p = stats.ttest_rel(group_data_denoised[g],
                               group_data_original[g])
        print(f"  {g}: t={t:.3f}, p={p:.4f}")


if __name__ == "__main__":

    output_path = "./output_dar"
    os.makedirs(output_path, exist_ok=True)

    # Load data following the same pattern as rmsTtest.py
    labels, r1, r2, r3, ts = load_and_preprocess_data()
    print(f"Data loaded: r1 shape={r1.shape}, ts shape={ts.shape}")
    print(f"Labels distribution: {np.bincount(labels.astype(int).flatten())}")

    freq_bands = {
        'delta': (0.5, 4),
        'alpha': (8, 13),
    }

    # Label mapping: 0=AD, 1=SCD/DSC, 2=MCI, 3=NC
    label_map = {'AD': 0, 'SCD': 1, 'MCI': 2, 'NC': 3}
    group_order = ['NC', 'SCD', 'MCI', 'AD']  # display order

    # Calculate band power for denoised EEG (r1) and original EEG (ts)
    print("\nCalculating band power for denoised EEG (r1)...")
    band_power_r1 = calculate_band_power_efficient(r1, 250, freq_bands)

    print("Calculating band power for original EEG (ts)...")
    band_power_ts = calculate_band_power_efficient(ts, 250, freq_bands)

    # Compute DAR
    dar_denoised = compute_dar(band_power_r1)    # shape (N, C)
    dar_original = compute_dar(band_power_ts)    # shape (N, C)

    print(f"DAR denoised range: [{dar_denoised.min():.4f}, {dar_denoised.max():.4f}]")
    print(f"DAR original range: [{dar_original.min():.4f}, {dar_original.max():.4f}]")

    # Plot bar chart
    plot_dar_bar(dar_denoised, dar_original, labels, group_order,
                 label_map, output_dir=output_path)

    # Plot scatter comparison
    plot_dar_comparison(dar_denoised, dar_original, labels, group_order,
                        label_map, output_dir=output_path)

    # Print statistics
    group_data_denoised = {}
    group_data_original = {}
    for g in group_order:
        mask = labels == label_map[g]
        group_data_denoised[g] = np.mean(dar_denoised[mask], axis=1)
        group_data_original[g] = np.mean(dar_original[mask], axis=1)

    print_dar_statistics(group_data_denoised, group_data_original, group_order)

    print("\nDone!")
