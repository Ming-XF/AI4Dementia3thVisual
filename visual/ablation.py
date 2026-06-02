import matplotlib.pyplot as plt
import numpy as np

import os

# ==================== 数据准备 ====================
# Full CVIB 基准
full_metrics = [91.49, 83.86, 84.09, 83.63, 83.67]

# 消融变体
variants = [
    "CVIB\nw.o. R.",
    "CVIB\nw.o. C.",
    "CVIB\nw.o. F.",
    "CVIB\nw.o. P.",
    "CVIB\nw.o. T.",
]

# 各消融变体的指标值 (AUC, Acc., Pre., Rec., FS.)
ablated = np.array([
    [53.02, 32.77, 33.62, 32.67, 31.75],  # w.o. R.
    [88.55, 81.20, 82.04, 81.02, 81.19],  # w.o. C.
    [56.13, 32.98, 35.30, 31.94, 29.84],  # w.o. F.
    [83.11, 68.03, 69.35, 67.92, 68.12],  # w.o. P.
    [84.29, 74.95, 74.76, 74.41, 73.58],  # w.o. T.
])

metric_names = ['AUC', 'Acc.', 'Pre.', 'Rec.', 'FS.']
n_metrics = len(metric_names)
n_variants = len(variants)

# Delta = 消融 - 完整模型
delta_matrix = ablated - full_metrics

# ==================== 绘图设置 ====================
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.default': 'regular',
})

# 单张大图，瀑布柱状图
fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

# 各指标使用不同颜色
metric_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# 紧凑分组参数
x = np.arange(n_variants) * 0.65
bar_width = 0.09

for i, (metric, color) in enumerate(zip(metric_names, metric_colors)):
    deltas = delta_matrix[:, i]
    offset = (i - (n_metrics - 1) / 2) * bar_width

    bars = ax.bar(
        x + offset,
        deltas,
        bar_width,
        color=color,
        edgecolor='white',
        linewidth=0.5,
        label=metric,
        zorder=2,
    )

    # 柱上标值（柱内底部，白色文字）
    for j, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            d - 1.5,
            f'{d:.1f}',
            ha='center',
            va='top',
            fontsize=22,
            color='white',
            rotation=90,
        )

# 零线
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

# X轴设置
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=22)
ax.set_xlim(-0.5, x[-1] + 0.5)

# Y轴设置
y_min = np.min(delta_matrix) - 6
y_max = 6
ax.set_ylim([y_min, y_max])
ax.set_ylabel('Δ from Full CVIB (%)', fontsize=22)

# variant 组间分隔线
for i in range(n_variants - 1):
    mid = (x[i] + x[i + 1]) / 2
    ax.axvline(x=mid, color='gray', linewidth=0.5, linestyle=':', alpha=0.35)

# 类别标注
y_top = y_max - 2.5
opt_center = (x[0] + x[1]) / 2
view_center = (x[2] + x[4]) / 2
ax.annotate(
    'Opt. terms', xy=(opt_center, y_top), fontsize=22,
    ha='center', color='#8B0000',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9, edgecolor='#E74C3C')
)
ax.annotate(
    'Views', xy=(view_center, y_top), fontsize=22,
    ha='center', color='#8B0000',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9, edgecolor='#E74C3C')
)

# 隐藏顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 水平网格线
ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
ax.set_axisbelow(True)

# 图例（右下角，x轴内侧）
ax.legend(
    loc='lower right',
    frameon=True,
    fancybox=True,
    fontsize=22,
    title='Metrics',
    title_fontsize=22,
    ncol=1,
)

ax.set_xlabel('Model Variants', fontsize=22)

output_path = os.path.join(os.path.dirname(__file__), "output_ablation")
os.makedirs(output_path, exist_ok=True)
# 显示图表
plt.savefig(output_path+'/CVIB_ablation.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(output_path+'/CVIB_ablation.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# ==================== 终端打印差值数据 ====================
print("\n" + "=" * 80)
print("ABLATION STUDY — Full CVIB vs. Variants  (Δ from Full CVIB)")
print("=" * 80)
print(f"Full CVIB:  AUC={full_metrics[0]:.2f}  Acc={full_metrics[1]:.2f}  "
      f"Pre={full_metrics[2]:.2f}  Rec={full_metrics[3]:.2f}  FS={full_metrics[4]:.2f}")
print("-" * 80)
print(f"{'Variant':<20} {'AUC':>8} {'Acc.':>8} {'Pre.':>8} {'Rec.':>8} {'FS.':>8}")
print("-" * 80)
variant_labels = ["w.o. R.", "w.o. C.", "w.o. F.", "w.o. P.", "w.o. T."]
for i, label in enumerate(variant_labels):
    d = delta_matrix[i]
    print(f"{label:<20} {d[0]:>+8.2f} {d[1]:>+8.2f} {d[2]:>+8.2f} {d[3]:>+8.2f} {d[4]:>+8.2f}")
print("-" * 80)
print(f"{'Mean Δ':<20} {np.mean(delta_matrix[:,0]):>+8.2f} {np.mean(delta_matrix[:,1]):>+8.2f} "
      f"{np.mean(delta_matrix[:,2]):>+8.2f} {np.mean(delta_matrix[:,3]):>+8.2f} "
      f"{np.mean(delta_matrix[:,4]):>+8.2f}")
print("=" * 80)