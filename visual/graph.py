import numpy as np
import os
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle

from brain import coordinates_data

import pdb

import warnings
warnings.filterwarnings('ignore')

def get_dmn_indices(coordinates_data):
    """
    返回 coordinates_data 字典中属于 DMN 网络节点的脑区的索引位置 (从0开始)。

    参数:
        coordinates_data: dict, 键是脑区名称，值是坐标列表的字典（需保持插入顺序，Python 3.7+ 字典默认有序）

    返回:
        list: 属于 DMN 的脑区在字典中的索引位置列表
    """
    # 定义属于 DMN 的脑区名称集合
    dmn_regions = {
        # 后部核心
        'l.precuneus', 'r.precuneus',
        'l.posteriorcingulate', 'r.posteriorcingulate',
        'l.isthmuscingulate', 'r.isthmuscingulate',
        # 前扣带回膝下部
        'l.rostralanteriorcingulate', 'r.rostralanteriorcingulate',
        # 顶下小叶 (包含角回)
        'l.inferiorparietal', 'r.inferiorparietal',
        # 缘上回 (常与下颌交界区相关)
        'l.supramarginal', 'r.supramarginal',
        # 内侧前额叶皮层
        'l.medialorbitofrontal', 'r.medialorbitofrontal',
        # 内侧颞叶子系统
        'l.parahippocampal', 'r.parahippocampal',
        # 外侧颞叶
        'l.middletemporal', 'r.middletemporal',
        'l.temporalpole', 'r.temporalpole',
    }
    
    # 使用列表推导式找到所有属于 DMN 的脑区索引
    indices = [idx for idx, region_name in enumerate(coordinates_data.keys()) if region_name in dmn_regions]
    
    return indices

def extract_subgraph_numpy(adj_matrix, node_indices):
    """
    从邻接矩阵中提取子图。
    
    参数:
        adj_matrix: numpy.ndarray, 形状为 (68, 68) 的邻接矩阵
        node_indices: list, 要提取的节点索引列表
    
    返回:
        subgraph: numpy.ndarray, 子图的邻接矩阵
    """
    # 使用 np.ix_ 创建索引网格，同时提取行和列
    subgraph = adj_matrix[np.ix_(node_indices, node_indices)]
    return subgraph

def load_and_preprocess_data():
    """
    加载正负样本数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, y, subject_id, cnn, r1, r2, r3 = data
    for i in range(len(node_feature)):
        np.fill_diagonal(node_feature[i], 0)
        np.fill_diagonal(adj[i], 0)
    return node_feature, adj, y, subject_id

def calculate_Eloc(G):
    return nx.local_efficiency(G)

def calcaulate_C(G):
    return nx.average_clustering(G)

def calculate_graph_metrics_fast(adj_matrix, threshold=0.25, sub_graph=False):
    """
    优化后的图论指标计算
    """
    if sub_graph:
        node_indices = get_dmn_indices(coordinates_data)
        adj_matrix = extract_subgraph_numpy(adj_matrix, node_indices)
    
    # 确保矩阵对称
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # 向量化阈值处理
    triu_indices = np.triu_indices_from(adj_matrix, k=1)
    triu_values = adj_matrix[triu_indices]
    
    if len(triu_values) > 0:
        threshold_value = np.percentile(np.abs(triu_values), (1 - threshold) * 100)
        mask = np.abs(adj_matrix) >= threshold_value
        adj_matrix = adj_matrix * mask
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # 创建图（使用无向图会更快）
    G = nx.from_numpy_array(adj_matrix)
    
    # 移除孤立节点
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes) == 0:
        return {'C': np.nan, 'Eloc': np.nan}
    
    # 1. 聚类系数 - 使用内置函数
    C = calcaulate_C(G)
    # 2. 局部效率 - 向量化计算
    Eloc = calculate_Eloc(G)
    
    return {
        'C': C,
        'Eloc': Eloc
    }

def process_single_sample(args):
    """处理单个样本的函数（用于并行）"""
    idx, con1_sample, con2_sample, current_label = args
    
    try:
        metrics1 = calculate_graph_metrics_fast(con1_sample)
    except:
        metrics1 = {'C': np.nan, 'Eloc': np.nan}
    
    try:
        metrics2 = calculate_graph_metrics_fast(con2_sample)
    except:
        metrics2 = {'C': np.nan, 'Eloc': np.nan}
    
    return idx, current_label, metrics1, metrics2

def main_analysis_fast(con1, con2, label, subject_id, group_mapping=None, n_jobs=-1):
    """
    优化后的主分析函数，支持并行计算
    """
    if group_mapping is None:
        group_mapping = {0: 'AD', 1: 'SCD', 2: 'MCI', 3: 'NC'}
    
    unique_labels = np.unique(label)
    print(f"Found groups: {[group_mapping[l] for l in unique_labels]}")
    print(f"Processing {len(con1)} samples...")
    
    # 准备并行处理的参数
    args_list = [(i, con1[i], con2[i], label[i]) for i in range(len(con1))]
    
    # 并行处理
    print("Using parallel processing...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_sample)(args) for args in args_list
    )
    
    # 收集结果
    con1_metrics = {l: {'C': [], 'Eloc': []} for l in unique_labels}
    con2_metrics = {l: {'C': [], 'Eloc': []} for l in unique_labels}
    
    for idx, current_label, metrics1, metrics2 in results:
        for key in metrics1:
            con1_metrics[current_label][key].append(metrics1[key])
            con2_metrics[current_label][key].append(metrics2[key])
    
    # 打印统计结果
    print_statistics(con1_metrics, con2_metrics, unique_labels, group_mapping)
    
    return con1_metrics, con2_metrics

def print_statistics(con1_metrics, con2_metrics, unique_labels, group_mapping):
    """打印统计结果"""
    print("\n" + "="*60)
    print("STATISTICAL RESULTS")
    print("="*60)
    
    for label in unique_labels:
        group_name = group_mapping[label]
        print(f"\nGroup: {group_name}")
        print("-"*40)
        
        for metric in ['C', 'Eloc']:
            data1 = [x for x in con1_metrics[label][metric] if not np.isnan(x)]
            data2 = [x for x in con2_metrics[label][metric] if not np.isnan(x)]
            
            if len(data1) > 0 and len(data2) > 0:
                print(f"\n{metric}:")
                print(f"  Original:   Mean={np.mean(data1):.4f}, SD={np.std(data1):.4f}, N={len(data1)}")
                print(f"  Denoised:   Mean={np.mean(data2):.4f}, SD={np.std(data2):.4f}, N={len(data2)}")
                
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                significace = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  Statistics: t={t_stat:.4f}, p={p_val:.4f} {significace}")


def plot_metrics_comparison(con1_metrics, con2_metrics, labels, group_names, save_path=None):
    """
    绘制原始数据和降噪后数据的图论指标对比 - 箱线图+趋势线版本
    保持原有的x轴分组结构：Original (con1) vs Denoised (con2)
    修改为两个独立的子图
    
    参数:
    con1_metrics: 原始数据的指标字典 {label: {'C': [], 'Eloc': []}}
    con2_metrics: 降噪后数据的指标字典
    labels: 标签列表
    group_names: 组名映射字典 {0: 'AD', 1: 'SCD', 2: 'MCI', 3: 'NC'}
    save_path: 保存路径
    """
    fs = 22
    
    metrics = ['C', 'Eloc']
    metric_names = {
        'C': 'Clustering Coefficient',
        'Eloc': 'Local Efficiency'
    }
    
    # 按照 AD, MCI, SCD, NC 的顺序排列（保持原有顺序）
    category_order = ['AD', 'MCI', 'SCD', 'NC']
    
    # 创建标签到名称的映射
    label_to_name = {label: group_names[label] for label in labels}
    
    # 按照指定顺序获取标签
    ordered_labels = []
    for cat in category_order:
        for label, name in label_to_name.items():
            if name == cat:
                ordered_labels.append(label)
                break
    
    # 颜色方案：为不同组使用不同颜色
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # AD, MCI, SCD, NC的颜色
    
    # 为每个指标创建独立的图形
    for idx, metric in enumerate(metrics):
        # 创建独立图形
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 准备数据
        con1_values = []
        con2_values = []
        con1_data_full = []
        con2_data_full = []
        
        for label in ordered_labels:
            con1_data = [x for x in con1_metrics[label][metric] if not np.isnan(x)]
            con2_data = [x for x in con2_metrics[label][metric] if not np.isnan(x)]
            
            con1_values.append(np.mean(con1_data))
            con2_values.append(np.mean(con2_data))
            con1_data_full.append(con1_data)
            con2_data_full.append(con2_data)
        
        # 设置柱状图位置 - 保持原有分组结构
        n_categories = len(ordered_labels)  # 4个类别
        bar_width = 0.8
        x1_positions = np.arange(n_categories)  # 原始数据组：4个位置
        x2_positions = np.arange(n_categories) + n_categories + 0.5  # 降噪数据组：4个位置，中间留间隙
        
        # 绘制箱线图
        bp1 = ax.boxplot(con1_data_full, positions=x1_positions, widths=bar_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.8),
                         medianprops=dict(color='darkblue', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='darkblue', markersize=4, alpha=0.5))
        
        bp2 = ax.boxplot(con2_data_full, positions=x2_positions, widths=bar_width,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.8),
                         medianprops=dict(color='darkred', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='darkred', markersize=4, alpha=0.5))
        
        # 计算中位数用于趋势线
        original_medians = [np.median(data) if len(data) > 0 else np.nan for data in con1_data_full]
        denoised_medians = [np.median(data) if len(data) > 0 else np.nan for data in con2_data_full]

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        
        # 绘制连接中位数的折线（在各自的分组内）
        # Original组内的趋势线（连接4个类别的中位数）
        ax.plot(x1_positions, original_medians, 'o-', color='darkblue', 
                linewidth=2.5, markersize=8, label='Original Median Trend', zorder=5)
        # 逐个修改Original组箱体颜色
        for i, box in enumerate(bp1['boxes']):
            box.set_facecolor(colors[i])
        
        # Denoised组内的趋势线（连接4个类别的中位数）
        ax.plot(x2_positions, denoised_medians, 's-', color='darkred', 
                linewidth=2.5, markersize=8, label='Denoised Median Trend', zorder=5)
        # 逐个修改Original组箱体颜色
        for i, box in enumerate(bp2['boxes']):
            box.set_facecolor(colors[i])
        
        # 添加线性回归趋势线
        if len(original_medians) >= 2:
            valid_indices = ~np.isnan(original_medians)
            if sum(valid_indices) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x1_positions[valid_indices], np.array(original_medians)[valid_indices]
                )
                x_line = np.linspace(min(x1_positions) - 0.5, max(x1_positions) + 0.5, 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, '--', color='darkblue', alpha=0.4, linewidth=1.5,
                       label=f'Original Trend (r={r_value:.3f}, p={p_value:.3f})')
        
        if len(denoised_medians) >= 2:
            valid_indices = ~np.isnan(denoised_medians)
            if sum(valid_indices) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x2_positions[valid_indices], np.array(denoised_medians)[valid_indices]
                )
                x_line = np.linspace(min(x2_positions) - 0.5, max(x2_positions) + 0.5, 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, '--', color='darkred', alpha=0.4, linewidth=1.5,
                       label=f'Denoised Trend (r={r_value:.3f}, p={p_value:.3f})')
        
        # 设置x轴刻度和标签
        all_positions = list(x1_positions) + list(x2_positions)
        group_centers = [(x1_positions[0] + x1_positions[-1]) / 2,
                        (x2_positions[0] + x2_positions[-1]) / 2]
        
        ax.set_xticks(group_centers)
        ax.set_xticklabels(['Original', 'Denoised'], fontsize=fs)
        
        # 为x轴添加类别标签
        # for i, (pos, cat) in enumerate(zip(x1_positions, category_order)):
        #     ax.text(pos, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
        #            cat, ha='center', va='top', fontsize=8, color=colors[i], fontweight='bold')
        
        # for i, (pos, cat) in enumerate(zip(x2_positions, category_order)):
        #     ax.text(pos, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
        #            cat, ha='center', va='top', fontsize=8, color=colors[i], fontweight='bold')
        
        # 添加类别图例
        # legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
        #                                 label=category_order[i]) for i in range(n_categories)]
        # legend_elements.extend([
        #     plt.Line2D([0], [0], color='darkblue', linewidth=2.5, marker='o', label='Original Trend'),
        #     plt.Line2D([0], [0], color='darkred', linewidth=2.5, marker='s', label='Denoised Trend')
        # ])
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=0.8, label='AD'),
            plt.Rectangle((0,0),1,1, facecolor=colors[1], alpha=0.8, label='MCI'),
            plt.Rectangle((0,0),1,1, facecolor=colors[2], alpha=0.8, label='SCD'),
            plt.Rectangle((0,0),1,1, facecolor=colors[3], alpha=0.8, label='NC'),
            # plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, label='Original (深色)'),
            # plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.4, label='Denoised (浅色)'),
            plt.Line2D([0], [0], color='darkblue', linewidth=2.5, marker='o', label='Original Trend'),
            plt.Line2D([0], [0], color='darkred', linewidth=2.5, marker='s', label='Denoised Trend')
        ]
        # ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        ax.legend(handles=legend_elements, loc='upper left', fontsize=fs)
        
        # 设置标签和标题
        ax.set_ylabel(metric_names[metric], fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        # ax.set_title(f'{metric_names[metric]} Comparison', fontsize=fs)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 调整y轴范围以留出一些空间
        all_data = [val for sublist in con1_data_full + con2_data_full for val in sublist]
        if all_data:
            y_min = min(all_data)
            y_max = max(all_data)
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        else:
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 为每个指标保存独立的图形
        plt.savefig(os.path.join(save_path, f"graph_metrics_comparison_boxplot_trend_{metric}.png"), dpi=300, bbox_inches='tight')
        # if save_path:
        #     # 在保存路径中添加指标名称
        #     path_parts = save_path.rsplit('.', 1)
        #     metric_save_path = f"{path_parts[0]}_{metric}.{path_parts[1]}"
        #     plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()  # 关闭当前图形，避免内存占用
    
# 使用优化版本
if __name__ == "__main__":
    os.makedirs('./output_graph', exist_ok=True)
    con1, con2, label, subject_id = load_and_preprocess_data()
    
    # 使用优化后的函数，n_jobs=-1使用所有CPU核心
    con1_metrics, con2_metrics = main_analysis_fast(
        con1, con2, label, subject_id, 
        n_jobs=-1  # 并行核心数，-1表示使用全部
    )
    
    # 绘图 - 使用箱线图+趋势线版本，保持原有分组结构
    plot_metrics_comparison(con1_metrics, con2_metrics, 
                          np.unique(label), 
                          {0: 'AD', 1: 'SCD', 2: 'MCI', 3: 'NC'},
                          save_path='./output_graph')