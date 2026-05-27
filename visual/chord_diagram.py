import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
import sys

# 从brain.py导入脑区名称
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import coordinates_data

# 68个脑区的原始名称列表（按brain.py字典顺序）
ALL_REGION_NAMES = list(coordinates_data.keys())

# 功能分组定义（按脑区名称关键词）
FUNCTIONAL_GROUPS = [
    ('Frontal', ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal',
                 'precentral', 'paracentral', 'parsopercularis', 'parstriangularis',
                 'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole']),
    ('Insula',  ['insula']),
    ('Temporal', ['superiortemporal', 'middletemporal', 'inferiortemporal',
                  'transversetemporal', 'bankssts', 'fusiform', 'entorhinal',
                  'parahippocampal', 'temporalpole']),
    ('Parietal', ['superiorparietal', 'inferiorparietal', 'supramarginal',
                  'postcentral', 'precuneus']),
    ('Occipital', ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine']),
    ('Cingulate', ['rostralanteriorcingulate', 'caudalanteriorcingulate',
                   'posteriorcingulate', 'isthmuscingulate']),
]

GROUP_COLORS = {
    'Frontal':   '#ff9999',
    'Insula':    '#ffcc99',
    'Temporal':  '#99cc99',
    'Parietal':  '#99ccff',
    'Occipital': '#cccc99',
    'Cingulate': '#cc99cc',
}

# 脑区名称缩写映射
NAME_ABBR = {
    'bankssts': 'BanksSTS', 'caudalanteriorcingulate': 'CauAntCing',
    'caudalmiddlefrontal': 'CauMidFront', 'cuneus': 'Cuneus',
    'entorhinal': 'Entorhinal', 'fusiform': 'Fusiform',
    'inferiorparietal': 'InfParietal', 'inferiortemporal': 'InfTemp',
    'isthmuscingulate': 'IsthCing', 'lateraloccipital': 'LatOcc',
    'lateralorbitofrontal': 'LatOrbFront', 'lingual': 'Lingual',
    'medialorbitofrontal': 'MedOrbFront', 'middletemporal': 'MidTemp',
    'parahippocampal': 'ParaHipp', 'paracentral': 'ParaCentral',
    'parsopercularis': 'ParsOperc', 'parsorbitalis': 'ParsOrb',
    'parstriangularis': 'ParsTri', 'pericalcarine': 'PeriCalc',
    'postcentral': 'PostCentral', 'posteriorcingulate': 'PostCing',
    'precentral': 'PreCentral', 'precuneus': 'Precuneus',
    'rostralanteriorcingulate': 'RosAntCing', 'rostralmiddlefrontal': 'RosMidFront',
    'superiorfrontal': 'SupFrontal', 'superiorparietal': 'SupParietal',
    'superiortemporal': 'SupTemp', 'supramarginal': 'SupraMarg',
    'frontalpole': 'FrontPole', 'temporalpole': 'TempPole',
    'transversetemporal': 'TransvTemp', 'insula': 'Insula',
}

def build_functional_order():
    """按功能分区排序：先左半脑(0-33)再右半脑(34-67)，区内按额叶-岛叶-颞叶-顶叶-枕叶-扣带排列"""
    new_order = []

    for hemi_prefix in ['l.', 'r.']:
        hemi_indices = []
        for group_name, keywords in FUNCTIONAL_GROUPS:
            for kw in keywords:
                full_name = hemi_prefix + kw
                if full_name in ALL_REGION_NAMES:
                    orig_idx = ALL_REGION_NAMES.index(full_name)
                    hemi_indices.append((orig_idx, group_name))
        new_order.extend(hemi_indices)

    return new_order  # list of (original_index, group_name)

def get_label(full_name):
    """从完整名称提取缩写"""
    # 去掉 l./r. 前缀
    region = full_name[2:] if full_name.startswith(('l.', 'r.')) else full_name
    hemi = full_name[:2]
    abbr = NAME_ABBR.get(region, region[:10])
    return f'{hemi}{abbr}'

def read_connections(csv_path):
    df = pd.read_csv(csv_path)
    return df

def draw_chord_figure(valid_connections, functional_order, new_to_orig, orig_to_new,
                      n_rois, angles, radius, ring_width, segment_angle,
                      group_boundaries, output_path, color, title_label):
    """绘制单张和弦图（正连接或负连接）"""
    fig, ax = plt.subplots(figsize=(30, 30), facecolor='white')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')

    FONT_SIZE = 18

    # --- 绘制外围功能分组色带 ---
    for new_pos in range(n_rois):
        center_angle = angles[new_pos]
        half_seg = segment_angle / 2
        start_angle = center_angle - half_seg
        end_angle = center_angle + half_seg

        _, group_name = functional_order[new_pos]
        seg_color = GROUP_COLORS[group_name]

        theta = np.linspace(start_angle, end_angle, 30)
        x_outer = (radius + ring_width) * np.cos(theta)
        y_outer = (radius + ring_width) * np.sin(theta)
        x_inner = (radius - ring_width) * np.cos(theta[::-1])
        y_inner = (radius - ring_width) * np.sin(theta[::-1])

        verts = list(zip(np.concatenate([x_outer, x_inner]),
                         np.concatenate([y_outer, y_inner])))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        path = Path(verts, codes)

        # 高亮有连接的ROI
        orig_idx = new_to_orig[new_pos]
        has_conn = ((valid_connections['R1_new'] == new_pos) |
                    (valid_connections['R2_new'] == new_pos)).any()
        edge_color = '#333333' if has_conn else '#999999'
        edge_lw = 1.5 if has_conn else 0.5
        patch = patches.PathPatch(path, facecolor=seg_color, edgecolor=edge_color, lw=edge_lw, zorder=2)
        ax.add_patch(patch)

    # --- 分组标签 ---
    for gs, ge, gname in group_boundaries:
        mid_pos = (gs + ge - 1) / 2
        mid_angle = angles[int(mid_pos)]
        lr = radius + ring_width + 0.14
        lx, ly = lr * np.cos(mid_angle), lr * np.sin(mid_angle)
        cos_a = np.cos(mid_angle)
        if cos_a >= 0.03:
            ha = 'left'
        elif cos_a <= -0.03:
            ha = 'right'
        else:
            ha = 'center'
        rot = np.degrees(mid_angle)
        if rot > 90 and rot < 270:
            rot += 180
        ax.text(lx, ly, gname, fontsize=FONT_SIZE, ha=ha, va='center', fontweight='bold',
               rotation=rot, rotation_mode='anchor', zorder=5, color='#333333')

    # --- 绘制连接线 ---
    abs_d = valid_connections['abs_cohens_d'].values
    if len(abs_d) > 0 and abs_d.max() > abs_d.min():
        linewidths = 1.5 + 6.5 * (abs_d - abs_d.min()) / (abs_d.max() - abs_d.min())
    else:
        linewidths = np.full(len(abs_d), 4.0)
    alphas = 0.65 + 0.30 * (abs_d - abs_d.min()) / (abs_d.max() - abs_d.min()) if abs_d.max() > abs_d.min() else np.full(len(abs_d), 0.80)

    sorted_indices = np.argsort(linewidths)

    for ii in sorted_indices:
        row = valid_connections.iloc[ii]
        r1_new = int(row['R1_new'])
        r2_new = int(row['R2_new'])

        angle1 = angles[r1_new]
        angle2 = angles[r2_new]

        r_start = radius - ring_width

        x1 = r_start * np.cos(angle1)
        y1 = r_start * np.sin(angle1)
        x2 = r_start * np.cos(angle2)
        y2 = r_start * np.sin(angle2)

        # Control point on the angle bisector
        angle_diff = angle2 - angle1
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        mid_angle = angle1 + angle_diff / 2
        cp_radius = r_start * 0.4
        cx = cp_radius * np.cos(mid_angle)
        cy = cp_radius * np.sin(mid_angle)

        t = np.linspace(0, 1, 80)
        bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
        by = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

        lw = linewidths[ii]

        ax.plot(bx, by, color=color, linewidth=lw, alpha=alphas[ii],
               solid_capstyle='round', zorder=1)

    # --- ROI标签 ---
    label_radius = radius + 0.09
    for new_pos in range(n_rois):
        orig_idx = new_to_orig[new_pos]
        full_name = ALL_REGION_NAMES[orig_idx]
        label = get_label(full_name)

        angle = angles[new_pos]
        x = label_radius * np.cos(angle)
        y = label_radius * np.sin(angle)

        cos_a = np.cos(angle)
        if cos_a >= 0.03:
            ha = 'left'
        elif cos_a <= -0.03:
            ha = 'right'
        else:
            ha = 'center'

        rot = np.degrees(angle)
        if rot > 90 and rot < 270:
            rot += 180

        ax.text(x, y, label, fontsize=FONT_SIZE, ha=ha, va='center', fontweight='bold',
               rotation=rot, rotation_mode='anchor', zorder=4)

    # --- 图例 ---
    max_d = abs_d.max() if len(abs_d) > 0 else 1.0
    legend_elements = [
        Line2D([0], [0], color='#555555', lw=2, linestyle='--', label=f'Min |d|=0.80'),
        Line2D([0], [0], color='#555555', lw=8, linestyle='--', label=f'Max |d|={max_d:.2f}'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=FONT_SIZE,
             framealpha=0.9, bbox_to_anchor=(0.02, 1.02),
             title=f'{title_label} ({len(valid_connections)} connections)',
             title_fontsize=FONT_SIZE)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Chord diagram saved to: {output_path}")


def chord_diagram(df, n_rois=68, output_dir='./model_2_data_result', pair_name='NC-AD'):
    """绘制正负连接分开的两张和弦图"""
    valid_connections = df[(df['Region1'] < n_rois) & (df['Region2'] < n_rois)].copy()

    if len(valid_connections) == 0:
        print("No valid connections found.")
        return

    # 构建功能排序
    functional_order = build_functional_order()
    orig_to_new = {orig_idx: new_pos for new_pos, (orig_idx, _) in enumerate(functional_order)}
    new_to_orig = [orig_idx for orig_idx, _ in functional_order]

    # 重映射连接中的Region索引
    valid_connections['R1_new'] = valid_connections['Region1'].map(orig_to_new)
    valid_connections['R2_new'] = valid_connections['Region2'].map(orig_to_new)

    radius = 1.0
    ring_width = 0.06
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)
    gap = 0.18
    segment_angle = 2 * np.pi / n_rois * (1 - gap)

    # 分组边界
    group_boundaries = []
    current_group = None
    group_start = 0
    for new_pos in range(n_rois):
        _, gname = functional_order[new_pos]
        if gname != current_group:
            if current_group is not None:
                group_boundaries.append((group_start, new_pos, current_group))
            current_group = gname
            group_start = new_pos
    group_boundaries.append((group_start, n_rois, current_group))
    print(f"Functional groups: {[(g, ge-gs) for gs,ge,g in group_boundaries]}")

    # 分离正负连接
    pos_conn = valid_connections[valid_connections['t_statistic'] > 0].copy()
    neg_conn = valid_connections[valid_connections['t_statistic'] < 0].copy()

    safe_name = pair_name.replace(' vs ', '-').replace(' ', '_')

    # 正连接图
    if len(pos_conn) > 0:
        draw_chord_figure(
            pos_conn, functional_order, new_to_orig, orig_to_new,
            n_rois, angles, radius, ring_width, segment_angle,
            group_boundaries,
            f'{output_dir}/chord_{safe_name}_positive.png',
            color='#2166ac', title_label=f'{pair_name} T > 0')
    else:
        print("No positive connections to plot.")

    # 负连接图
    if len(neg_conn) > 0:
        draw_chord_figure(
            neg_conn, functional_order, new_to_orig, orig_to_new,
            n_rois, angles, radius, ring_width, segment_angle,
            group_boundaries,
            f'{output_dir}/chord_{safe_name}_negative.png',
            color='#b2182b', title_label=f'{pair_name} T < 0')
    else:
        print("No negative connections to plot.")

if __name__ == "__main__":
    csv_path = './model_2_data_result/cvib0_NC vs AD_high_quality_connections.csv'

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        exit(1)

    df = read_connections(csv_path)
    print(f"Loaded {len(df)} connections")
    print(f"Positive T: {(df['t_statistic'] > 0).sum()}, Negative T: {(df['t_statistic'] < 0).sum()}")
    print(f"Effect size range: {df['abs_cohens_d'].min():.3f} ~ {df['abs_cohens_d'].max():.3f}")

    chord_diagram(df, output_dir='./model_2_data_result', pair_name='NC vs AD')
