import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from math import pi
import os

# ==================== 模式切换 ====================
MODE = 'split'  # 'combined': 两张雷达图并排; 'split': 分开保存

# ==================== 样式设置 ====================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',
})

# ==================== 数据准备 ====================

metrics = ['AUC', 'Acc.', 'Pre.', 'Rec.', 'FS.']
n_metrics = len(metrics)

angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
angles += angles[:1]

# 所有方法按类别排列
all_methods_ordered = [
    'SteadyNet', 'TACNet', 'TCACNet', 'DCN',
    'CEEDNet', 'EEGNet', 'LMDA', 'SCN',
    'AlzNetV3', 'FBNetGen', 'ALTER', 'BNT', 'BNC',
    'VIB', 'GCDGCN',
]

# 每个方法的颜色
_method_colors_tab = [
    '#1f77b4', '#2c9fc3', '#38a5d6', '#48b2db',
    '#5bbcd6', '#74c4d8', '#8ecfde', '#a7dbe8',
    '#2ca02c', '#4caf50', '#66bb6a', '#81c784', '#a5d6a7',
    '#ff7f0e', '#e6ab02',
]

method_colors = {}
for i, m in enumerate(all_methods_ordered):
    method_colors[m] = _method_colors_tab[i]
method_colors['CVIB'] = '#E74C3C'

# --- Motor Imagery 数据 ---
mi_data = {
    'SteadyNet':  [71.04, 63.26, 64.62, 77.11, 67.89],
    'TACNet':     [78.42, 71.26, 76.17, 66.78, 69.29],
    'TCACNet':    [78.93, 72.01, 73.56, 73.10, 71.98],
    'DCN':        [82.72, 69.38, 64.13, 92.09, 75.25],
    'CEEDNet':    [83.81, 71.12, 77.40, 67.65, 67.79],
    'EEGNet':     [84.15, 74.87, 77.45, 70.42, 73.69],
    'LMDA':       [85.18, 75.29, 77.28, 74.20, 74.76],
    'SCN':        [85.37, 75.70, 78.02, 71.57, 74.65],
    'AlzNetV3':   [53.69, 50.00, 33.33, 66.67, 44.44],
    'FBNetGen':   [63.26, 55.50, 45.84, 46.17, 40.05],
    'ALTER':      [53.18, 50.00, 16.67, 33.33, 22.22],
    'BNT':        [53.76, 51.46, 50.95, 88.52, 64.42],
    'BNC':        [53.18, 52.70, 52.95, 60.73, 54.49],
    'VIB':        [52.60, 52.16, 52.17, 55.87, 53.22],
}
mi_cvib = [87.06, 76.74, 81.19, 70.56, 74.89]

# --- Epilepsy 数据 ---
ep_data = {
    'TACNet':     [77.73, 74.82, 71.34, 87.86, 76.20],
    'TCACNet':    [77.98, 73.79, 74.18, 86.79, 76.17],
    'SteadyNet':  [95.30, 72.59, 74.92, 91.67, 78.00],
    'SCN':        [99.54, 89.22, 87.74, 95.77, 90.73],
    'EEGNet':     [99.84, 99.01, 98.59, 99.52, 99.05],
    'LMDA':       [99.87, 97.95, 96.26, 99.52, 97.86],
    'DCN':        [99.93, 92.45, 99.19, 85.95, 91.67],
    'CEEDNet':    [99.93, 97.98, 97.35, 98.57, 97.93],
    'AlzNetV3':   [64.19, 55.77, 60.70, 38.27, 40.49],
    'FBNetGen':   [94.88, 84.28, 98.95, 69.82, 78.22],
    'BNC':        [98.39, 84.07, 98.10, 70.48, 78.67],
    'ALTER':      [98.77, 80.02, 96.40, 64.05, 67.50],
    'BNT':        [99.17, 86.25, 97.83, 75.24, 81.44],
    'VIB':        [60.50, 44.57, 36.04, 35.24, 20.96],
    'GCDGCN':     [99.56, 85.31, 98.12, 72.86, 80.38],
}
ep_cvib = [99.99, 98.02, 98.04, 97.62, 97.77]

# ==================== 绘制函数 ====================

def draw_radar(ax, data, cvib):
    for method, vals in data.items():
        vals_c = vals + vals[:1]
        color = method_colors.get(method, '#888888')
        ax.plot(angles, vals_c, '-', linewidth=0.8, color=color, alpha=0.55)

    cvib_c = cvib + cvib[:1]
    ax.plot(angles, cvib_c, 'o-', linewidth=3.0, color='#E74C3C',
            markersize=8, zorder=10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=20)
    ax.tick_params(axis='x', pad=30)
    ax.set_ylim(0, 100)
    ax.set_yticks([])


def build_legend_handles():
    handles = []
    handles.append(mlines.Line2D([], [], color='#333333', linewidth=0, marker='',
                                  label='Temporal-Spectral'))
    for m in ['SteadyNet', 'TACNet', 'TCACNet', 'DCN',
              'CEEDNet', 'EEGNet', 'LMDA', 'SCN']:
        handles.append(mlines.Line2D([], [], color=method_colors[m], linewidth=2,
                                      label='  ' + m))
    handles.append(mlines.Line2D([], [], color='#333333', linewidth=0, marker='',
                                  label='Brain Network'))
    for m in ['AlzNetV3', 'FBNetGen', 'ALTER', 'BNT', 'BNC']:
        handles.append(mlines.Line2D([], [], color=method_colors[m], linewidth=2,
                                      label='  ' + m))
    handles.append(mlines.Line2D([], [], color='#333333', linewidth=0, marker='',
                                  label='Denoising'))
    for m in ['VIB', 'GCDGCN']:
        handles.append(mlines.Line2D([], [], color=method_colors[m], linewidth=2,
                                      label='  ' + m))
    handles.append(mlines.Line2D([], [], color='white', linewidth=0, label=''))
    handles.append(mlines.Line2D([], [], color='#E74C3C', linewidth=3, marker='o',
                                  markersize=7, label='CVIB (Ours)'))
    return handles


output_path = os.path.join(os.path.dirname(__file__), "output_cross_task")
os.makedirs(output_path, exist_ok=True)

# ==================== 绘图 ====================

if MODE == 'combined':
    fig, axes = plt.subplots(1, 2, figsize=(22, 8),
                             subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.40, left=0.05, right=0.58)

    titles = ['Motor Imagery', 'Epilepsy']
    for ax, title, (data, cvib) in zip(axes, titles,
                                       [(mi_data, mi_cvib), (ep_data, ep_cvib)]):
        draw_radar(ax, data, cvib)
        ax.set_title(title, fontsize=20, pad=25)

    fig.legend(
        handles=build_legend_handles(),
        loc='center left',
        bbox_to_anchor=(0.62, 0.50),
        fontsize=16,
        frameon=True,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.5,
    )

    plt.savefig(output_path + '/cross_task_radar.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path + '/cross_task_radar.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: cross_task_radar.png / .pdf")

else:
    # 图 1: Motor Imagery (无图例)
    fig1, ax1 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    draw_radar(ax1, mi_data, mi_cvib)
    plt.savefig(output_path + '/radar_motor_imagery.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path + '/radar_motor_imagery.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)

    # 图 2: Epilepsy (带图例)
    fig2, ax2 = plt.subplots(figsize=(14, 7), subplot_kw=dict(polar=True))
    fig2.subplots_adjust(right=0.32)
    draw_radar(ax2, ep_data, ep_cvib)

    fig2.legend(
        handles=build_legend_handles(),
        loc='center left',
        bbox_to_anchor=(0.38, 0.50),
        fontsize=14,
        frameon=True,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.5,
    )

    plt.savefig(output_path + '/radar_epilepsy.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path + '/radar_epilepsy.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)

    print("Saved:")
    print("  radar_motor_imagery.png / .pdf")
    print("  radar_epilepsy.png / .pdf")
