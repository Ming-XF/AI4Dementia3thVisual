import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import coordinates_data

# Paths
csv_path = os.path.join(os.path.dirname(__file__),
                        'model_2_data_result', 'cvib0_NC vs AD_high_quality_connections.csv')
output_dir = os.path.join(os.path.dirname(__file__), 'model_2_data_result')

ALL_REGIONS = list(coordinates_data.keys())
ALL_COORDS = np.array(list(coordinates_data.values()))

# Read connections
df = pd.read_csv(csv_path)

# Separate by T sign
pos_df = df[df['t_statistic'] > 0].copy()
neg_df = df[df['t_statistic'] < 0].copy()


def region_name(idx):
    return ALL_REGIONS[idx]


def top3_connections(group_df):
    """Return top 3 connections by abs_cohens_d."""
    top = group_df.nlargest(3, 'abs_cohens_d')
    result = []
    for _, row in top.iterrows():
        r1 = region_name(int(row['Region1']))
        r2 = region_name(int(row['Region2']))
        d = row['abs_cohens_d']
        result.append(f"  {r1} -- {r2}  (|d|={d:.3f})")
    return result


def print_stats(group_df, label):
    n = len(group_df)
    mean_t = group_df['t_statistic'].mean()
    var_t = group_df['t_statistic'].var(ddof=1)
    mean_d = group_df['abs_cohens_d'].mean()
    var_d = group_df['abs_cohens_d'].var(ddof=1)

    print(f"\n{'='*60}")
    print(f"  {label} (n={n})")
    print(f"{'='*60}")
    print(f"  T value:       {mean_t:.4f} ± {var_t:.4f} (mean ± var)")
    print(f"  |Cohen's d|:   {mean_d:.4f} ± {var_d:.4f} (mean ± var)")
    print(f"  Top 3 |d| connections:")
    for line in top3_connections(group_df):
        print(line)


print_stats(pos_df, "Positive (T > 0)")
print_stats(neg_df, "Negative (T < 0)")


# ============================================================
# Nilearn connectome plots (only nodes with connections)
# ============================================================
def filtered_connectome(group_df, title, cmap, output_name):
    """Build matrix and coords for only connected nodes, then plot."""
    # Find connected ROI indices
    connected = set()
    for _, row in group_df.iterrows():
        connected.add(int(row['Region1']))
        connected.add(int(row['Region2']))
    connected = sorted(connected)

    idx_map = {old: new for new, old in enumerate(connected)}
    n = len(connected)

    # Build reduced matrix
    mat = np.zeros((n, n))
    for _, row in group_df.iterrows():
        i = idx_map[int(row['Region1'])]
        j = idx_map[int(row['Region2'])]
        val = abs(row['t_statistic'])
        mat[i, j] = val
        mat[j, i] = val

    coords = ALL_COORDS[connected]

    print(f"\nPlotting {title}: {n} nodes, {len(group_df)} edges")

    fig = plot_connectome(
        mat, coords,
        edge_threshold=0.001,
        title=title,
        node_size=30,
        edge_cmap=cmap,
        edge_vmin=mat[mat > 0].min() * 0.9,
        edge_vmax=mat[mat > 0].max(),
        colorbar=True,
    )
    fig.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_name}")


from nilearn.plotting import plot_connectome
from matplotlib.colors import LinearSegmentedColormap

red_cmap = LinearSegmentedColormap.from_list('solid_red', ['#d73027', '#67000d'])
blue_cmap = LinearSegmentedColormap.from_list('solid_blue', ['#2171b5', '#08306b'])

filtered_connectome(pos_df, "NC vs AD — Positive Connections (T > 0)",
                    red_cmap, 'connectome_positive.png')
filtered_connectome(neg_df, "NC vs AD — Negative Connections (T < 0)",
                    blue_cmap, 'connectome_negative.png')

# Combined (all connections, signed)
connected_all = set()
for _, row in df.iterrows():
    connected_all.add(int(row['Region1']))
    connected_all.add(int(row['Region2']))
connected_all = sorted(connected_all)
idx_map_all = {old: new for new, old in enumerate(connected_all)}
n_all = len(connected_all)
mat_all = np.zeros((n_all, n_all))
for _, row in df.iterrows():
    i = idx_map_all[int(row['Region1'])]
    j = idx_map_all[int(row['Region2'])]
    mat_all[i, j] = row['t_statistic']
    mat_all[j, i] = row['t_statistic']
coords_all = ALL_COORDS[connected_all]

print(f"\nPlotting Combined: {n_all} nodes, {len(df)} edges")
fig_all = plot_connectome(
    mat_all, coords_all,
    edge_threshold=0.001,
    title="NC vs AD — All Connections",
    node_size=30,
    edge_cmap='coolwarm',
    edge_vmin=df['t_statistic'].min(),
    edge_vmax=df['t_statistic'].max(),
    colorbar=True,
)
fig_all.savefig(os.path.join(output_dir, 'connectome_combined.png'), dpi=300, bbox_inches='tight')
print("  Saved to: connectome_combined.png")

print("\nDone. All outputs saved to:", output_dir)
