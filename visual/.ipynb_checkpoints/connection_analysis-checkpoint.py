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
    std_t = group_df['t_statistic'].std(ddof=1)
    mean_d = group_df['abs_cohens_d'].mean()
    std_d = group_df['abs_cohens_d'].std(ddof=1)

    print(f"\n{'='*60}")
    print(f"  {label} (n={n})")
    print(f"{'='*60}")
    print(f"  T value:       {mean_t:.4f} ± {std_t:.4f} (M ± SD)")
    print(f"  |Cohen's d|:   {mean_d:.4f} ± {std_d:.4f} (M ± SD)")
    print(f"  Top 3 |d| connections:")
    for line in top3_connections(group_df):
        print(line)


print_stats(pos_df, "Positive (T > 0)")
print_stats(neg_df, "Negative (T < 0)")


# ============================================================
# Nilearn connectome: Top 3 |d| from each group in a single call
#    Positive (T > 0) → blue (+|d|),  Negative (T < 0) → red (-|d|)
# ============================================================
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Gather top 3 from each group
pos_top = pos_df.nlargest(3, 'abs_cohens_d')
neg_top = neg_df.nlargest(3, 'abs_cohens_d')
selected = pd.concat([pos_top, neg_top])

# Collect all nodes involved
all_nodes = set()
for _, row in selected.iterrows():
    all_nodes.add(int(row['Region1']))
    all_nodes.add(int(row['Region2']))
all_nodes = sorted(all_nodes)

idx_map = {old: new for new, old in enumerate(all_nodes)}
n = len(all_nodes)
coords = ALL_COORDS[all_nodes]

# Build signed matrix: +|d| for positive group, -|d| for negative group
mat = np.zeros((n, n))
for _, row in pos_top.iterrows():
    i, j = idx_map[int(row['Region1'])], idx_map[int(row['Region2'])]
    mat[i, j] = row['abs_cohens_d']
    mat[j, i] = row['abs_cohens_d']
for _, row in neg_top.iterrows():
    i, j = idx_map[int(row['Region1'])], idx_map[int(row['Region2'])]
    mat[i, j] = -row['abs_cohens_d']
    mat[j, i] = -row['abs_cohens_d']

d_max = max(row['abs_cohens_d'] for _, row in selected.iterrows())

print(f"\nPlotting top-3 connectome: {n} nodes, 6 edges  (|d| max: {d_max:.3f})")

# Diverging colormap: negative → red, zero → white, positive → blue
div_cmap = LinearSegmentedColormap.from_list('div_rb', ['#d73027', '#f7f7f7', '#2171b5'])

fig = plot_connectome(
    mat, coords,
    edge_threshold=0.001,
    title=None,
    node_size=30,
    edge_cmap=div_cmap,
    edge_vmin=-d_max,
    edge_vmax=d_max,
    colorbar=False,
)

LEGEND_X = 0.98   # 0~1, 相对于 axes 右边界
LEGEND_Y = 1.1   # 0~1, 相对于 axes 上边界
LEGEND_FONTSIZE = 9

ax = plt.gca()
legend_patches = [
    mpatches.Patch(color='#2171b5', label='Positive (T > 0)'),
    mpatches.Patch(color='#d73027', label='Negative (T < 0)'),
]
ax.legend(handles=legend_patches, fontsize=LEGEND_FONTSIZE, framealpha=0.9,
          bbox_to_anchor=(LEGEND_X, LEGEND_Y), loc='upper right',
          bbox_transform=ax.transAxes)

fig.savefig(os.path.join(output_dir, 'connectome_top3_pos_neg.png'),
            dpi=300, bbox_inches='tight')
print("  Saved to: connectome_top3_pos_neg.png")

print("\nDone. All outputs saved to:", output_dir)
