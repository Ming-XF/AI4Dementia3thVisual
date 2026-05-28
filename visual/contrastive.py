import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os

# 设置样式
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 13
sns.set_style("ticks")

# F-Score 数据
fs_data = {
    ('XWEEG', '2-class'): [88.88, 96.42],
    ('CAUEEG', '2-class'): [96.62, 94.77],
    ('XWEEG', '4-class'): [81.12, 83.67],
    ('CAUEEG', '4-class'): [61.96, 88.88],
}

scenarios = list(fs_data.keys())
scenario_labels = [f'{d}\n{t}' for (d, t) in scenarios]

# 颜色
color_cvib_wn = '#E74C3C'
color_cvib = '#2ECC71'

# 单张图
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(scenarios))
bar_width = 0.30

for i, scenario in enumerate(scenarios):
    cvib_wn_val, cvib_val = fs_data[scenario]
    improvement = cvib_val - cvib_wn_val

    # 并排柱状图
    bars = ax.bar(
        [x[i] - bar_width / 2, x[i] + bar_width / 2],
        [cvib_wn_val, cvib_val],
        bar_width,
        color=[color_cvib_wn, color_cvib],
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
        zorder=2,
    )
    if i == 0:
        bars[0].set_label('CVIB w. N.')
        bars[1].set_label('CVIB')

    # 垂直箭头标注差异
    if abs(improvement) > 0.5:
        arrow_color = '#27AE60' if improvement > 0 else '#C0392B'
        symbol = '+' if improvement > 0 else ''
        y_bottom = max(cvib_wn_val, cvib_val) + 2
        y_top = y_bottom + 3

        ax.annotate('',
                    xy=(x[i], y_top),
                    xytext=(x[i], y_bottom),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.0))

        ax.text(x[i], y_top + 1.5, f'{symbol}{improvement:.2f}%',
                ha='center', va='bottom',
                fontsize=22, color=arrow_color)

# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=22)
ax.set_ylabel('F-Score (%)', fontsize=22)
ax.set_ylim(0, 112)

ax.tick_params(labelsize=22)

# 图例
ax.legend(loc='upper right', fontsize=20, frameon=True, ncol=2)

# 坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(width=1.5)

plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "output_contrastive")
os.makedirs(output_path, exist_ok=True)
plt.savefig(output_path + '/CVIB_contrastive.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_path + '/CVIB_contrastive.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# 终端统计输出
print("=" * 80)
print("CONFLICT TRANSFER STRATEGY - F-SCORE COMPARISON")
print("=" * 80)
print(f"{'Dataset':<10} {'Task':<10} {'CVIB w. N.':>12} {'CVIB':>12} {'Δ':>10}")
print("-" * 55)

for (dataset, task), (cvib_wn, cvib) in fs_data.items():
    imp = cvib - cvib_wn
    label = "▲ CRITICAL" if imp > 20 else ("▲ MAJOR" if imp > 5 else ("▲" if imp > 0 else "▼"))
    print(f"{dataset:<10} {task:<10} {cvib_wn:>10.2f}% {cvib:>10.2f}% {imp:>+9.2f}  {label}")
print("=" * 80)
