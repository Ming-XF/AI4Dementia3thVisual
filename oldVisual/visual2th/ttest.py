import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import pickle
import pandas as pd
import os

import pdb

def load_and_preprocess_data(cvib=0, positive_class=0, negative_class=3):
    """
    加载正负样本数据
    """
    with open('../../data.pkl', 'rb') as f:
        data = pickle.load(f)

    positive_data = data[cvib][positive_class]
    negative_data = data[cvib][negative_class]
    
    return positive_data, negative_data

def calculate_cohens_d(group1, group2):
    """
    计算Cohen's d效应量
    """
    n1, n2 = len(group1), len(group2)
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std > 0:
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        cohens_d = 0
    
    return cohens_d

def perform_group_comparison(positive_data, negative_data, alpha=0.05, correction_method='fdr_bh'):
    """
    对每个功能连接进行组间t检验，并进行多重比较校正
    """
    n_regions = positive_data.shape[1]
    
    t_stats = np.zeros((n_regions, n_regions))
    p_values = np.zeros((n_regions, n_regions))
    cohens_d = np.zeros((n_regions, n_regions))
    significant_connections = np.zeros((n_regions, n_regions), dtype=bool)
    
    lower_tri_indices = np.tril_indices(n_regions, k=-1)
    
    all_p_values = []
    connection_indices = []
    
    for i, j in zip(lower_tri_indices[0], lower_tri_indices[1]):
        positive_conn = positive_data[:, i, j]
        negative_conn = negative_data[:, i, j]
        
        t_stat, p_val = ttest_ind(positive_conn, negative_conn, equal_var=False)
        d_value = calculate_cohens_d(positive_conn, negative_conn)
        
        t_stats[i, j] = t_stat
        t_stats[j, i] = t_stat
        p_values[i, j] = p_val
        p_values[j, i] = p_val
        cohens_d[i, j] = d_value
        cohens_d[j, i] = d_value
        
        all_p_values.append(p_val)
        connection_indices.append((i, j))
    
    all_p_values = np.array(all_p_values)
    
    if correction_method == 'fdr_bh':
        rejected, corrected_pvals, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
    elif correction_method == 'bonferroni':
        rejected, corrected_pvals, _, _ = multipletests(all_p_values, alpha=alpha, method='bonferroni')
    else:
        rejected = all_p_values < alpha
        corrected_pvals = all_p_values
    
    for idx, ((i, j), is_sig) in enumerate(zip(connection_indices, rejected)):
        if is_sig:
            significant_connections[i, j] = True
            significant_connections[j, i] = True
    
    corrected_p_matrix = np.full((n_regions, n_regions), 1.0)
    for idx, (i, j) in enumerate(connection_indices):
        corrected_p_matrix[i, j] = corrected_pvals[idx]
        corrected_p_matrix[j, i] = corrected_pvals[idx]
    
    return t_stats, p_values, corrected_p_matrix, significant_connections, cohens_d

def filter_high_quality_connections(significant_connections, t_stats, corrected_p_matrix, 
                                   cohens_d, positive_means, negative_means,
                                   p_threshold=0.01, effect_threshold=0.8):
    """
    筛选高显著性和高效应量的连接
    """
    n_regions = t_stats.shape[0]
    high_quality_connections = []
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if significant_connections[i, j]:
                p_val = corrected_p_matrix[i, j]
                d_val = abs(cohens_d[i, j])
                
                if p_val < p_threshold and d_val > effect_threshold:
                    high_quality_connections.append({
                        'Region1': i,
                        'Region2': j,
                        't_statistic': t_stats[i, j],
                        'p_value': p_val,
                        'cohens_d': cohens_d[i, j],
                        'abs_cohens_d': d_val,
                        'positive_mean': positive_means[i, j],
                        'negative_mean': negative_means[i, j],
                        'mean_difference': positive_means[i, j] - negative_means[i, j]
                    })
    
    df = pd.DataFrame(high_quality_connections)
    if len(df) > 0:
        df = df.sort_values('abs_cohens_d', ascending=False)
    
    return df

def visualize_high_quality_heatmap(high_quality_df, n_regions=68, save_fig=False, fig_name='high_quality_heatmap.png'):
    """
    绘制高显著和高效应量连接的热图
    """
    if len(high_quality_df) == 0:
        return None
    
    plt.rcParams.update({'font.size': 14})
    
    # 创建矩阵，只标记高显著和高效应量的连接
    high_quality_matrix = np.zeros((n_regions, n_regions))
    for _, row in high_quality_df.iterrows():
        i, j = int(row['Region1']), int(row['Region2'])
        high_quality_matrix[i, j] = row['cohens_d']
        high_quality_matrix[j, i] = row['cohens_d']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    mask = np.triu(np.ones_like(high_quality_matrix, dtype=bool), k=1)
    masked_data = np.ma.masked_where(mask, high_quality_matrix)
    
    vmax = np.max(np.abs(high_quality_matrix[high_quality_matrix != 0])) if np.any(high_quality_matrix != 0) else 2.0
    vmax = min(vmax, 2.0)
    
    im = ax.imshow(masked_data, cmap='RdBu_r', aspect='auto', 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # 标记高显著和高效应量的连接
    for _, row in high_quality_df.iterrows():
        i, j = int(row['Region1']), int(row['Region2'])
        ax.plot(j, i, '*', color='yellow', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d", fontsize=14)
    
    ax.set_xlabel('Brain Region', fontsize=14)
    ax.set_ylabel('Brain Region', fontsize=14)
    ax.set_title(f'High Significance & High Effect Size Connections\n(p < 0.01, |d| > 0.8, n={len(high_quality_df)})', 
                 fontsize=14)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def save_high_quality_to_csv(high_quality_df, cvib, positive_class, negative_class):
    """
    保存高显著和高效应量连接信息到CSV文件
    """
    if len(high_quality_df) == 0:
        return
    
    filename = f'{cvib}_high_quality_connections_{positive_class}-{negative_class}.csv'
    high_quality_df.to_csv(filename, index=False)

def play_static(cvib=0, positive_class=2, negative_class=3, 
                p_threshold=0.01, effect_threshold=0.8):
    """
    主函数 - 筛选高显著和高效应量连接，保存CSV并绘制热图
    """
    # 加载数据
    positive_data, negative_data = load_and_preprocess_data(
        cvib=cvib, positive_class=positive_class, negative_class=negative_class
    )
    
    # 执行组间比较
    t_stats, p_values, corrected_p_matrix, significant_connections, cohens_d = perform_group_comparison(
        positive_data, negative_data, alpha=0.05, correction_method='fdr_bh'
    )
    
    # 计算均值
    positive_means = positive_data.mean(axis=0)
    negative_means = negative_data.mean(axis=0)
    
    # 筛选高显著和高效应量连接
    high_quality_df = filter_high_quality_connections(
        significant_connections, t_stats, corrected_p_matrix, 
        cohens_d, positive_means, negative_means,
        p_threshold=p_threshold, effect_threshold=effect_threshold
    )
    
    # 保存到CSV
    save_high_quality_to_csv(high_quality_df, cvib, positive_class, negative_class)
    
    # 绘制热图
    if len(high_quality_df) > 0:
        visualize_high_quality_heatmap(
            high_quality_df, 
            n_regions=positive_data.shape[1],
            save_fig=True,
            fig_name=f'{cvib}_high_quality_heatmap_{positive_class}-{negative_class}.png'
        )
    
    return high_quality_df

def analyze_common_high_quality(results_list, class_names=None, output_dir='./'):
    """
    分析多个类别对之间的共同高显著高效应量连接，并保存到CSV
    """
    if class_names is None:
        class_names = [f"Pair{i+1}" for i in range(len(results_list))]
    
    # 提取每个类别对的高质量连接集合
    connection_sets = []
    for result in results_list:
        if result is None or len(result) == 0:
            connection_sets.append(set())
        else:
            connections = set()
            for _, row in result.iterrows():
                i, j = int(row['Region1']), int(row['Region2'])
                connections.add((min(i, j), max(i, j)))
            connection_sets.append(connections)
    
    # 找出共同连接
    common_connections = set.intersection(*connection_sets) if connection_sets else set()
    
    if len(common_connections) == 0:
        return pd.DataFrame()
    
    # 收集共同连接的详细信息
    common_details = []
    for conn in sorted(common_connections):
        detail = {
            'Region1': conn[0],
            'Region2': conn[1],
            'Connection': f"{conn[0]}-{conn[1]}"
        }
        
        # 添加每个类别对的效应量和p值信息
        for idx, class_name in enumerate(class_names):
            if results_list[idx] is not None and len(results_list[idx]) > 0:
                df = results_list[idx]
                conn_data = df[(df['Region1'] == conn[0]) & (df['Region2'] == conn[1])]
                if len(conn_data) > 0:
                    detail[f'{class_name}_cohens_d'] = conn_data.iloc[0]['cohens_d']
                    detail[f'{class_name}_p_value'] = conn_data.iloc[0]['p_value']
        
        common_details.append(detail)
    
    df_common = pd.DataFrame(common_details)
    
    # 保存共同连接到CSV
    csv_filename = os.path.join(output_dir, 'common_high_quality_connections.csv')
    df_common.to_csv(csv_filename, index=False)
    
    return df_common

if __name__ == "__main__":
    # 创建输出目录
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # 设置阈值
    p_threshold = 0.01
    effect_threshold = 0.8
    
    # 运行三个类别对的比较
    hq1 = play_static(cvib=0, positive_class=0, negative_class=3, 
                      p_threshold=p_threshold, effect_threshold=effect_threshold)
    
    hq2 = play_static(cvib=0, positive_class=2, negative_class=3, 
                      p_threshold=p_threshold, effect_threshold=effect_threshold)
    
    hq3 = play_static(cvib=0, positive_class=1, negative_class=3, 
                      p_threshold=p_threshold, effect_threshold=effect_threshold)
    
    # 分析共同高显著高效应量连接
    results_list = [hq1, hq2, hq3]
    class_names = ['Nor_vs_AD', 'Nor_vs_MCI', 'Nor_vs_DSC']
    
    common_df = analyze_common_high_quality(results_list, class_names, output_dir='./')