import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

import os

# 设置中文字体
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据准备 ====================

# 定义方法和类别
methods = [
    'TCACNet', 'TACNet', 'LMDA', 'DCN', 'EEGNet', 'SteadyNet', 'CEEDNet', 'SCN',
    'AlzNetV3', 'FBNetGen', 'ALTER', 'BNT', 'BNC',
    'VIB', 'GCDGCN', 'CVIB'
]

categories = {
    'Temporal-Spectral': ['TCACNet', 'TACNet', 'LMDA', 'DCN', 'EEGNet', 'SteadyNet', 'CEEDNet', 'SCN'],
    'Brain Network': ['AlzNetV3', 'FBNetGen', 'ALTER', 'BNT', 'BNC'],
    'Denoising': ['VIB', 'GCDGCN', 'CVIB']
}

method_to_category = {}
for cat, mlist in categories.items():
    for m in mlist:
        method_to_category[m] = cat

# 定义所有指标行 (指标×数据集×任务)
metrics_rows = []
for task in ['2cls', '4cls']:
    for dataset in ['XWEEG', 'CAUEEG']:
        for metric in ['AUC', 'Acc', 'Pre', 'Rec', 'FS']:
            metrics_rows.append(f"{dataset}-{task}-{metric}")

print(f"总共有 {len(metrics_rows)} 个指标行")

# 从表格中提取数据
# 二分类任务数据
binary_data = {
    'XWEEG': {
        'TCACNet': {'AUC': 52.55, 'Acc': 52.56, 'Pre': 36.89, 'Rec': 63.10, 'FS': 46.48},
        'TACNet': {'AUC': 53.27, 'Acc': 52.56, 'Pre': 69.58, 'Rec': 67.86, 'FS': 49.26},
        'LMDA': {'AUC': 56.42, 'Acc': 54.49, 'Pre': 39.24, 'Rec': 51.19, 'FS': 44.42},
        'DCN': {'AUC': 59.57, 'Acc': 53.21, 'Pre': 39.17, 'Rec': 53.57, 'FS': 44.44},
        'EEGNet': {'AUC': 62.00, 'Acc': 60.26, 'Pre': 60.20, 'Rec': 77.38, 'FS': 67.61},
        'SteadyNet': {'AUC': np.nan, 'Acc': np.nan, 'Pre': np.nan, 'Rec': np.nan, 'FS': np.nan},
        'CEEDNet': {'AUC': 67.31, 'Acc': 61.54, 'Pre': 62.89, 'Rec': 66.67, 'FS': 63.93},
        'SCN': {'AUC': 73.96, 'Acc': 64.74, 'Pre': 66.62, 'Rec': 75.00, 'FS': 68.61},
        'AlzNetV3': {'AUC': 63.59, 'Acc': 55.13, 'Pre': 60.19, 'Rec': 59.52, 'FS': 57.35},
        'FBNetGen': {'AUC': 75.69, 'Acc': 66.03, 'Pre': 66.39, 'Rec': 78.57, 'FS': 71.32},
        'ALTER': {'AUC': 85.22, 'Acc': 67.31, 'Pre': 86.11, 'Rec': 51.19, 'FS': 60.07},
        'BNT': {'AUC': 87.00, 'Acc': 72.44, 'Pre': 81.67, 'Rec': 64.29, 'FS': 70.80},
        'BNC': {'AUC': 88.72, 'Acc': 78.21, 'Pre': 75.54, 'Rec': 88.10, 'FS': 81.23},
        'VIB': {'AUC': 69.64, 'Acc': 66.03, 'Pre': 63.51, 'Rec': 88.10, 'FS': 73.65},
        'GCDGCN': {'AUC': 86.19, 'Acc': 60.26, 'Pre': 44.62, 'Rec': 61.90, 'FS': 50.92},
        'CVIB': {'AUC': 99.95, 'Acc': 96.15, 'Pre': 96.77, 'Rec': 96.43, 'FS': 96.42}
    },
    'CAUEEG': {
        'TCACNet': {'AUC': 57.64, 'Acc': 57.56, 'Pre': 56.55, 'Rec': 54.07, 'FS': 43.68},
        'TACNet': {'AUC': 62.11, 'Acc': 60.48, 'Pre': 55.60, 'Rec': 77.70, 'FS': 63.82},
        'LMDA': {'AUC': 86.51, 'Acc': 80.07, 'Pre': 76.94, 'Rec': 79.54, 'FS': 78.22},
        'DCN': {'AUC': 86.10, 'Acc': 77.90, 'Pre': 75.71, 'Rec': 74.86, 'FS': 75.28},
        'EEGNet': {'AUC': 67.72, 'Acc': 67.45, 'Pre': 63.18, 'Rec': 62.46, 'FS': 62.74},
        'SteadyNet': {'AUC': np.nan, 'Acc': np.nan, 'Pre': np.nan, 'Rec': np.nan, 'FS': np.nan},
        'CEEDNet': {'AUC': 90.29, 'Acc': 80.14, 'Pre': 84.07, 'Rec': 70.47, 'FS': 75.81},
        'SCN': {'AUC': 86.02, 'Acc': 69.59, 'Pre': 86.85, 'Rec': 42.14, 'FS': 50.75},
        'AlzNetV3': {'AUC': np.nan, 'Acc': np.nan, 'Pre': np.nan, 'Rec': np.nan, 'FS': np.nan},
        'FBNetGen': {'AUC': 81.63, 'Acc': 70.94, 'Pre': 51.72, 'Rec': 49.83, 'FS': 50.75},
        'ALTER': {'AUC': 91.83, 'Acc': 84.51, 'Pre': 82.61, 'Rec': 83.02, 'FS': 82.82},
        'BNT': {'AUC': 92.74, 'Acc': 85.57, 'Pre': 83.27, 'Rec': 84.99, 'FS': 84.11},
        'BNC': {'AUC': 94.01, 'Acc': 87.06, 'Pre': 83.76, 'Rec': 88.52, 'FS': 86.02},
        'VIB': {'AUC': np.nan, 'Acc': np.nan, 'Pre': np.nan, 'Rec': np.nan, 'FS': np.nan},
        'GCDGCN': {'AUC': 94.47, 'Acc': 74.23, 'Pre': 74.10, 'Rec': 90.88, 'FS': 78.98},
        'CVIB': {'AUC': 98.88, 'Acc': 95.27, 'Pre': 94.26, 'Rec': 95.28, 'FS': 94.77}
    }
}

# 四分类任务数据
fourcls_data = {
    'XWEEG': {
        'TCACNet': {'AUC': 50.69, 'Acc': 25.74, 'Pre': 26.70, 'Rec': 25.72, 'FS': 22.17},
        'TACNet': {'AUC': 50.32, 'Acc': 25.32, 'Pre': 28.07, 'Rec': 25.16, 'FS': 21.43},
        'LMDA': {'AUC': 52.29, 'Acc': 32.12, 'Pre': 32.52, 'Rec': 31.87, 'FS': 31.45},
        'DCN': {'AUC': 51.87, 'Acc': 31.49, 'Pre': 31.33, 'Rec': 31.60, 'FS': 31.18},
        'EEGNet': {'AUC': 52.00, 'Acc': 29.36, 'Pre': 29.46, 'Rec': 29.10, 'FS': 28.95},
        'SteadyNet': {'AUC': 51.83, 'Acc': 28.08, 'Pre': 7.02, 'Rec': 25.00, 'FS': 10.96},
        'CEEDNet': {'AUC': 53.88, 'Acc': 32.77, 'Pre': 34.73, 'Rec': 32.82, 'FS': 31.39},
        'SCN': {'AUC': 55.44, 'Acc': 31.49, 'Pre': 31.02, 'Rec': 31.24, 'FS': 30.88},
        'AlzNetV3': {'AUC': 55.20, 'Acc': 30.64, 'Pre': 28.17, 'Rec': 30.09, 'FS': 27.77},
        'FBNetGen': {'AUC': 57.47, 'Acc': 37.44, 'Pre': 37.29, 'Rec': 37.32, 'FS': 36.91},
        'ALTER': {'AUC': 69.93, 'Acc': 46.16, 'Pre': 48.08, 'Rec': 45.45, 'FS': 45.02},
        'BNT': {'AUC': 69.47, 'Acc': 45.96, 'Pre': 47.88, 'Rec': 45.20, 'FS': 44.29},
        'BNC': {'AUC': 66.26, 'Acc': 45.73, 'Pre': 46.12, 'Rec': 45.31, 'FS': 45.10},
        'VIB': {'AUC': 54.64, 'Acc': 30.21, 'Pre': 30.23, 'Rec': 29.50, 'FS': 28.15},
        'GCDGCN': {'AUC': 84.11, 'Acc': 71.42, 'Pre': 72.28, 'Rec': 71.22, 'FS': 71.23},
        'CVIB': {'AUC': 91.49, 'Acc': 83.86, 'Pre': 84.09, 'Rec': 83.63, 'FS': 83.67}
    },
    'CAUEEG': {
        'TCACNet': {'AUC': 49.80, 'Acc': 27.67, 'Pre': 21.45, 'Rec': 24.71, 'FS': 17.38},
        'TACNet': {'AUC': 50.81, 'Acc': 30.96, 'Pre': 28.36, 'Rec': 26.23, 'FS': 21.41},
        'LMDA': {'AUC': 62.10, 'Acc': 40.97, 'Pre': 39.21, 'Rec': 38.74, 'FS': 38.53},
        'DCN': {'AUC': 63.56, 'Acc': 38.89, 'Pre': 37.92, 'Rec': 38.41, 'FS': 37.78},
        'EEGNet': {'AUC': 50.04, 'Acc': 32.19, 'Pre': 23.51, 'Rec': 25.06, 'FS': 19.42},
        'SteadyNet': {'AUC': 54.28, 'Acc': 33.68, 'Pre': 23.91, 'Rec': 29.26, 'FS': 25.03},
        'CEEDNet': {'AUC': 65.25, 'Acc': 40.38, 'Pre': 39.30, 'Rec': 38.80, 'FS': 38.29},
        'SCN': {'AUC': 61.35, 'Acc': 37.09, 'Pre': 35.89, 'Rec': 35.28, 'FS': 35.39},
        'AlzNetV3': {'AUC': 49.67, 'Acc': 31.21, 'Pre': 15.06, 'Rec': 25.01, 'FS': 18.53},
        'FBNetGen': {'AUC': 65.38, 'Acc': 47.20, 'Pre': 29.59, 'Rec': 40.20, 'FS': 32.30},
        'ALTER': {'AUC': 68.12, 'Acc': 47.90, 'Pre': 39.05, 'Rec': 43.83, 'FS': 40.10},
        'BNT': {'AUC': 80.20, 'Acc': 58.04, 'Pre': 57.60, 'Rec': 57.51, 'FS': 57.53},
        'BNC': {'AUC': 81.90, 'Acc': 61.72, 'Pre': 61.78, 'Rec': 61.15, 'FS': 61.42},
        'VIB': {'AUC': 49.91, 'Acc': 35.13, 'Pre': 18.85, 'Rec': 25.03, 'FS': 13.61},
        'GCDGCN': {'AUC': 93.02, 'Acc': 80.08, 'Pre': 80.54, 'Rec': 79.65, 'FS': 79.73},
        'CVIB': {'AUC': 97.83, 'Acc': 88.73, 'Pre': 89.13, 'Rec': 88.69, 'FS': 88.88}
    }
}

# 构建数据矩阵
data_matrix = np.zeros((len(metrics_rows), len(methods)))
data_matrix.fill(np.nan)

for i, metric_row in enumerate(metrics_rows):
    parts = metric_row.split('-')
    dataset = parts[0]
    task = parts[1]
    metric = parts[2]
    
    if task == '2cls':
        source_data = binary_data
    else:
        source_data = fourcls_data
    
    for j, method in enumerate(methods):
        if method in source_data[dataset]:
            data_matrix[i, j] = source_data[dataset][method][metric]

# 创建DataFrame
df = pd.DataFrame(data_matrix, index=metrics_rows, columns=methods)

# 计算CVIB相对于第二名的提升幅度
improvement_matrix = np.zeros((len(metrics_rows), len(methods)))
improvement_matrix.fill(np.nan)

for i, metric_row in enumerate(metrics_rows):
    # 获取该行的所有值
    row_values = df.iloc[i].values.copy()
    cvib_value = row_values[-1]  # CVIB是最后一列
    
    if np.isnan(cvib_value):
        continue
    
    # 将NaN替换为-inf以便排序
    row_values_sorted = row_values.copy()
    row_values_sorted[np.isnan(row_values_sorted)] = -np.inf
    
    # 找到第二好的值（不包括CVIB）
    sorted_indices = np.argsort(row_values_sorted)[::-1]
    second_best_idx = None
    for idx in sorted_indices:
        if idx != len(methods) - 1 and not np.isnan(row_values[idx]):  # 不是CVIB且不是NaN
            second_best_idx = idx
            break
    
    if second_best_idx is not None:
        second_best_value = row_values[second_best_idx]
        improvement = cvib_value - second_best_value
        improvement_matrix[i, :] = improvement

# 创建第二个矩阵用于标注（非CVIB方法显示提升幅度，CVIB显示原始值）
annotation_matrix = np.empty((len(metrics_rows), len(methods)), dtype=object)
for i in range(len(metrics_rows)):
    for j in range(len(methods)):
        if j == len(methods) - 1:  # CVIB列
            if not np.isnan(data_matrix[i, j]):
                annotation_matrix[i, j] = f'{data_matrix[i, j]:.1f}'
            else:
                annotation_matrix[i, j] = 'N/A'
        else:  # 非CVIB列
            if not np.isnan(improvement_matrix[i, j]):
                annotation_matrix[i, j] = f'+{improvement_matrix[i, j]:.2f}'
            else:
                annotation_matrix[i, j] = ''

# ==================== 绘图 ====================

# 创建掩码
mask = pd.DataFrame(np.isnan(data_matrix), index=metrics_rows, columns=methods)
data_for_plot = pd.DataFrame(data_matrix, index=metrics_rows, columns=methods)

# 简化行标签 & 分组信息
metric_names = ['AUC', 'Acc', 'Pre', 'Rec', 'FS']
group_info = [
    ('XWEEG\n2cls', '#E8F5E9'),
    ('CAUEEG\n2cls', '#E3F2FD'),
    ('XWEEG\n4cls', '#FFF3E0'),
    ('CAUEEG\n4cls', '#F3E5F5'),
]
short_labels = metric_names * len(group_info)
group_boundaries = [i * 5 for i in range(len(group_info) + 1)]

# 方法类别在x轴上的颜色条
cat_colors = {'Temporal-Spectral': '#5DADE2', 'Brain Network': '#58D68D', 'Denoising': '#EC7063'}

fig, ax = plt.subplots(figsize=(26, 14))

# 绘制热图 —— 使用 RdYlBu_r 色阶（禁用内置colorbar，手动放在更右侧）
sns.heatmap(data_for_plot, mask=mask, cmap='RdYlBu_r', vmin=0, vmax=100,
            annot=False, fmt='', linewidths=0.3, linecolor='#EEEEEE',
            cbar=False,
            xticklabels=True, yticklabels=short_labels, ax=ax)

# 字体
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
ax.xaxis.tick_bottom()

# 手动添加 colorbar，放在右侧分组标签之后
cbar_ax = fig.add_axes([0.9, 0.12, 0.015, 0.76])
cbar = fig.colorbar(ax.collections[0], cax=cbar_ax)
cbar.set_label('Score (%)', fontsize=22)
cbar.ax.tick_params(labelsize=18)

# 行分组 —— 粗分隔线 + 右侧彩色条
for i, (label, color) in enumerate(group_info):
    y_start, y_end = group_boundaries[i], group_boundaries[i + 1]
    if i > 0:
        ax.axhline(y=y_start, color='black', linewidth=2.0, linestyle='-')
    rect = mpatches.Rectangle((len(methods) + 0.15, y_start), 0.55, 5,
                               facecolor=color, edgecolor='#BBBBBB', linewidth=0.5,
                               clip_on=False)
    ax.add_patch(rect)
    ax.text(len(methods) + 0.425, (y_start + y_end) / 2, label,
            ha='center', va='center', fontsize=19, rotation=0)

# 方法类别 —— 顶部彩色条
cat_boundaries = {'Temporal-Spectral': (0, 8), 'Brain Network': (8, 13), 'Denoising': (13, 16)}
for cat, (x0, x1) in cat_boundaries.items():
    rect = mpatches.Rectangle((x0, -0.65), x1 - x0, 0.55,
                               facecolor=cat_colors[cat], edgecolor='#999999', linewidth=0.5,
                               clip_on=False)
    ax.add_patch(rect)
    ax.text((x0 + x1) / 2, -0.375, cat, ha='center', va='center', fontsize=20)

# 高亮CVIB列
ax.axvline(x=len(methods) - 1, color='#FFD700', linewidth=2.5, linestyle='-')
ax.axvline(x=len(methods), color='#FFD700', linewidth=2.5, linestyle='-')

# CVIB列提升标注
cvib_col = len(methods) - 1
for i in range(len(metrics_rows)):
    imp = improvement_matrix[i, 0]
    if not np.isnan(imp):
        ax.text(cvib_col + 0.5, i + 0.5, f'+{imp:.2f}',
                ha='center', va='center', color='black', fontsize=14)

fig.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.10)

output_path = os.path.join(os.path.dirname(__file__), "output_baseline")
os.makedirs(output_path, exist_ok=True)
plt.savefig(output_path + '/CVIB_heatmap_comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_path + '/CVIB_heatmap_comparison.pdf', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("热图已保存为 CVIB_heatmap_comparison.png 和 CVIB_heatmap_comparison.pdf")
print("\n数据统计：")
print(f"- 总方法数: {len(methods)}")
print(f"- 总指标行数: {len(metrics_rows)}")
print(f"- 包含缺失数据的方法: SteadyNet (2cls), AlzNetV3 (CAUEEG 2cls), VIB (CAUEEG 2cls)")
print(f"- CVIB在所有任务和数据集上均取得最佳性能")