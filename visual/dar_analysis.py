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
                                   window_sec=4, return_relative=True):
    N, C, L = time_series.shape
    band_powers = {}

    nperseg = min(L, int(hz * window_sec))
    noverlap = nperseg // 2
    data_reshaped = time_series.reshape(-1, L)

    freqs, psd_matrix = signal.welch(
        data_reshaped, fs=hz, nperseg=nperseg, noverlap=noverlap,
        detrend='constant', axis=-1
    )

    total_mask = (freqs >= 0.5) & (freqs <= 45)
    total_power = simpson(psd_matrix[:, total_mask], freqs[total_mask], axis=-1)
    band_powers['total_power'] = total_power.reshape(N, C)

    for band_name, (f_low, f_high) in freq_bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        freq_range = freqs[freq_mask]
        psd_band = psd_matrix[:, freq_mask]
        abs_power = simpson(psd_band, freq_range, axis=-1)
        band_powers[band_name] = abs_power.reshape(N, C)

        if return_relative:
            band_powers[band_name + '_rel'] = (abs_power / total_power).reshape(N, C)

    return band_powers


def load_and_preprocess_data():
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)
    _, _, labels, _, _, r1, r2, r3, ts = data
    return labels, r1, r2, r3, ts


if __name__ == "__main__":

    output_path = "./output_cv"
    os.makedirs(output_path, exist_ok=True)

    labels, r1, r2, r3, ts = load_and_preprocess_data()
    n_channels = r1.shape[1]

    print(f"Data loaded: r1 shape={r1.shape}, ts={ts.shape}, {n_channels} channels")
    print(f"Labels: {np.bincount(labels.astype(int).flatten())}")

    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }
    label_map = {'AD': 0, 'SCD': 1, 'MCI': 2, 'NC': 3}
    group_order = ['NC', 'SCD', 'MCI', 'AD']
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    group_colors = {'NC': '#2ca02c', 'SCD': '#ff7f0e', 'MCI': '#1f77b4', 'AD': '#d62728'}

    print("Computing band power...")
    bp_r1 = calculate_band_power_efficient(r1, 250, freq_bands, return_relative=True)
    bp_ts = calculate_band_power_efficient(ts, 250, freq_bands, return_relative=True)

    # --- Compute per-band per-group per-channel CV ---
    # CV = std / mean, computed per channel then averaged across channels
    cv_data = {'r1': {}, 'ts': {}}
    for band in band_names:
        cv_data['r1'][band] = {}
        cv_data['ts'][band] = {}
        for g in group_order:
            mask = (labels == label_map[g]).flatten()
            # CV per channel, then mean across channels
            cv_r1 = np.mean(np.std(bp_r1[band][mask], axis=0, ddof=1) /
                            (np.mean(bp_r1[band][mask], axis=0) + 1e-12))
            cv_ts = np.mean(np.std(bp_ts[band][mask], axis=0, ddof=1) /
                            (np.mean(bp_ts[band][mask], axis=0) + 1e-12))
            cv_data['r1'][band][g] = cv_r1
            cv_data['ts'][band][g] = cv_ts

    # ===== Figure 1: Grouped Bar — CV per band per group, r1 vs ts =====
    fig, axes = plt.subplots(1, len(band_names), figsize=(24, 5))

    for ax, band in zip(axes, band_names):
        x = np.arange(len(group_order))
        width = 0.35

        cv_r1_vals = [cv_data['r1'][band][g] for g in group_order]
        cv_ts_vals = [cv_data['ts'][band][g] for g in group_order]

        bars1 = ax.bar(x - width / 2, cv_r1_vals, width,
                       color='steelblue', edgecolor='black', linewidth=0.8,
                       label='r1 (denoised)')
        bars2 = ax.bar(x + width / 2, cv_ts_vals, width,
                       color='darkorange', edgecolor='black', linewidth=0.8,
                       label='ts (original)')

        # Annotate reduction ratio
        for i, (v_r1, v_ts) in enumerate(zip(cv_r1_vals, cv_ts_vals)):
            ratio = v_r1 / v_ts
            ax.text(x[i], max(v_r1, v_ts) + 0.02,
                    f'{ratio:.2f}x', ha='center', fontsize=8,
                    color='green' if ratio < 1 else 'red', fontweight='bold')

        ax.set_title(band, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_order, fontsize=11)
        ax.set_ylabel('CV (std / mean)', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Within-Group Coefficient of Variation: Denoised (r1) vs Original (ts)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{output_path}/cv_bar_by_band.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved: cv_bar_by_band.png")

    # ===== Figure 2: Summary — mean CV reduction ratio across groups per band =====
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(band_names))
    reduction_ratios = []
    for band in band_names:
        ratios = [cv_data['r1'][band][g] / cv_data['ts'][band][g]
                  for g in group_order]
        reduction_ratios.append(np.mean(ratios))

    bars = ax.bar(x, reduction_ratios, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'],
                  edgecolor='black', linewidth=0.8)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
               label='No change (ratio=1)')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5,
               label='50% reduction')

    for bar, ratio in zip(bars, reduction_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{ratio:.3f}', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=13)
    ax.set_ylabel('CV Ratio (r1 / ts)', fontsize=13)
    ax.set_title('Mean CV Reduction Across Groups\n(< 1 = Denoising Reduces Variance)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(reduction_ratios) + 0.15)

    plt.tight_layout()
    fig.savefig(f'{output_path}/cv_reduction_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: cv_reduction_summary.png")

    # ===== Figure 3: Per-group line — CV across bands, r1 vs ts =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, g in zip(axes, group_order):
        cv_r1_vals = [cv_data['r1'][band][g] for band in band_names]
        cv_ts_vals = [cv_data['ts'][band][g] for band in band_names]

        x = np.arange(len(band_names))
        ax.plot(x, cv_r1_vals, 'o-', color='steelblue', linewidth=2, markersize=8,
                label='r1 (denoised)')
        ax.plot(x, cv_ts_vals, 's--', color='darkorange', linewidth=2, markersize=8,
                label='ts (original)')

        # Fill between
        ax.fill_between(x, cv_r1_vals, cv_ts_vals, alpha=0.15,
                        color='steelblue' if np.mean(cv_r1_vals) < np.mean(cv_ts_vals) else 'darkorange')

        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=11)
        ax.set_ylabel('CV (std / mean)', fontsize=11)
        ax.set_title(f'{g}', fontsize=14, fontweight='bold',
                     color=group_colors[g])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Within-Group CV by Frequency Band: Denoised vs Original',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_path}/cv_per_group_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: cv_per_group_line.png")

    # ===== Print summary =====
    print("\n" + "=" * 75)
    print("Within-Group Coefficient of Variation Analysis")
    print("=" * 75)
    print(f"{'Band':<10}", end="")
    for g in group_order:
        print(f"{'':>6}{g}{'':>6}", end=" ")
    print(f"{'Mean r1/ts':>12}")
    print("-" * 75)

    for band in band_names:
        print(f"{band:<10}", end="")
        ratios = []
        for g in group_order:
            cv_r = cv_data['r1'][band][g]
            cv_t = cv_data['ts'][band][g]
            ratio = cv_r / cv_t
            ratios.append(ratio)
            print(f"  {cv_r:.3f}/{cv_t:.3f} ", end=" ")
        print(f"{np.mean(ratios):>10.3f}")

    print("-" * 75)
    print(f"{'Mean':<10}", end="")
    for g in group_order:
        mean_ratio = np.mean([cv_data['r1'][b][g] / cv_data['ts'][b][g]
                              for b in band_names])
        print(f"{'':>7}{mean_ratio:.3f}{'':>6}", end=" ")
    overall_mean = np.mean([cv_data['r1'][b][g] / cv_data['ts'][b][g]
                            for b in band_names for g in group_order])
    print(f"{overall_mean:>10.3f}")

    print(f"\nAll outputs saved to {output_path}/")
    print("Done!")
