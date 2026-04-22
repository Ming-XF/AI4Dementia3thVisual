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

    node_feature, adj, y, subject_id = data
    mask_pos = (y == positive_class)
    mask_nev = (y == negative_class)
    if cvib == 1:
        positive_data = node_feature[mask_pos]
        negative_data = node_feature[mask_nev]
    else:
        positive_data = adj[mask_pos]
        negative_data = adj[mask_nev]

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

def create_combined_figure(all_results, cvib_list, pair_names, fig_type='t_stats'):
    """
    创建包含6个子图的组合图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{fig_type} Heatmaps for All Comparisons', fontsize=16, fontweight='bold')
    
    for idx, (ax, result) in enumerate(zip(axes.flat, all_results)):
        cvib = cvib_list[idx]
        pair_name = pair_names[idx % 3]
        
        if fig_type == 't_stats':
            data_matrix = result['t_stats']
            cmap = 'RdBu_r'
            vmax = np.max(np.abs(data_matrix))
            title = f'cvib={cvib}, {pair_name}\nT-statistics'
            cbar_label = 'T-statistic'
        elif fig_type == 'significant':
            # 创建显著连接矩阵（只显示显著连接的t值或p值）
            data_matrix = result['t_stats'].copy()
            data_matrix[~result['significant_connections']] = 0
            cmap = 'RdBu_r'
            vmax = np.max(np.abs(data_matrix)) if np.any(data_matrix != 0) else 1
            title = f'cvib={cvib}, {pair_name}\nSignificant Connections (FDR<0.05)'
            cbar_label = 'T-statistic'
        elif fig_type == 'high_effect':
            # 创建高效应量矩阵（只显示|d|>0.8的连接）
            data_matrix = result['cohens_d'].copy()
            data_matrix[np.abs(data_matrix) <= 0.8] = 0
            cmap = 'RdBu_r'
            vmax = np.max(np.abs(data_matrix)) if np.any(data_matrix != 0) else 1
            title = f'cvib={cvib}, {pair_name}\nHigh Effect Size (|d|>0.8)'
            cbar_label = "Cohen's d"
        
        # 创建下三角掩码
        mask = np.triu(np.ones_like(data_matrix, dtype=bool), k=1)
        masked_data = np.ma.masked_where(mask, data_matrix)
        
        im = ax.imshow(masked_data, cmap=cmap, aspect='auto', 
                      vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1, 
                      interpolation='nearest')
        
        ax.set_xlabel('Brain Region')
        ax.set_ylabel('Brain Region')
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    
    plt.tight_layout()
    return fig

def save_connection_data_to_csv(result, cvib, pair_name, data_type='significant'):
    """
    保存连接数据到CSV文件
    """
    n_regions = result['t_stats'].shape[0]
    connections_data = []
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if data_type == 'significant':
                if result['significant_connections'][i, j]:
                    connections_data.append({
                        'Region1': i,
                        'Region2': j,
                        't_statistic': result['t_stats'][i, j],
                        'p_value': result['corrected_p_matrix'][i, j],
                        'cohens_d': result['cohens_d'][i, j]
                    })
            elif data_type == 'high_effect':
                if abs(result['cohens_d'][i, j]) > 0.8:
                    connections_data.append({
                        'Region1': i,
                        'Region2': j,
                        't_statistic': result['t_stats'][i, j],
                        'p_value': result['corrected_p_matrix'][i, j],
                        'cohens_d': result['cohens_d'][i, j],
                        'abs_cohens_d': abs(result['cohens_d'][i, j])
                    })
    
    if connections_data:
        df = pd.DataFrame(connections_data)
        if data_type == 'high_effect':
            df = df.sort_values('abs_cohens_d', ascending=False)
        filename = f'cvib{cvib}_{pair_name}_{data_type}_connections.csv'
        df.to_csv(filename, index=False)
        return df
    return pd.DataFrame()

def filter_high_quality_connections(result, p_threshold=0.01, effect_threshold=0.8):
    """
    筛选高显著性和高效应量的连接
    """
    n_regions = result['t_stats'].shape[0]
    high_quality_connections = []
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if result['significant_connections'][i, j]:
                p_val = result['corrected_p_matrix'][i, j]
                d_val = abs(result['cohens_d'][i, j])
                
                if p_val < p_threshold and d_val > effect_threshold:
                    high_quality_connections.append({
                        'Region1': i,
                        'Region2': j,
                        't_statistic': result['t_stats'][i, j],
                        'p_value': p_val,
                        'cohens_d': result['cohens_d'][i, j],
                        'abs_cohens_d': d_val
                    })
    
    df = pd.DataFrame(high_quality_connections)
    if len(df) > 0:
        df = df.sort_values('abs_cohens_d', ascending=False)
    
    return df

def find_common_significant_connections(results_cvib0, pair_names):
    """
    找出cvib=0的三个类别对共同存在的显著连接
    """
    if len(results_cvib0) != 3:
        return pd.DataFrame()
    
    # 提取每个类别对的显著连接集合
    sig_sets = []
    for result in results_cvib0:
        sig_set = set()
        n_regions = result['t_stats'].shape[0]
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                if result['significant_connections'][i, j]:
                    sig_set.add((i, j))
        sig_sets.append(sig_set)
    
    # 找出共同显著连接
    common_sig = set.intersection(*sig_sets)
    
    if not common_sig:
        return pd.DataFrame()
    
    # 收集共同连接的详细信息
    common_data = []
    for conn in sorted(common_sig):
        i, j = conn
        row_data = {
            'Region1': i,
            'Region2': j,
            'Connection': f"{i}-{j}"
        }
        
        # 添加每个类别对的统计信息
        for idx, pair_name in enumerate(pair_names):
            result = results_cvib0[idx]
            row_data[f'{pair_name}_t_statistic'] = result['t_stats'][i, j]
            row_data[f'{pair_name}_p_value'] = result['corrected_p_matrix'][i, j]
            row_data[f'{pair_name}_cohens_d'] = result['cohens_d'][i, j]
        
        common_data.append(row_data)
    
    df_common = pd.DataFrame(common_data)
    df_common.to_csv('cvib0_common_significant_connections.csv', index=False)
    return df_common

def main_analysis():
    """
    主分析函数
    """
    # 创建输出目录
    output_dir = './analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # 定义参数
    cvib_list = [0, 0, 0, 1, 1, 1]
    pos_classes = [0, 2, 1, 0, 2, 1]
    neg_class = 3
    pair_names = ['Nor_vs_AD', 'Nor_vs_MCI', 'Nor_vs_DSC']
    
    # 存储所有结果
    all_results = []
    results_cvib0 = []
    all_high_quality_dfs = []
    
    # 对每个比较进行分析
    for idx, (cvib, pos_class) in enumerate(zip(cvib_list, pos_classes)):
        pair_name = pair_names[idx % 3]
        print(f"Processing cvib={cvib}, {pair_name}...")
        
        # 加载数据
        positive_data, negative_data = load_and_preprocess_data(
            cvib=cvib, positive_class=pos_class, negative_class=neg_class
        )
        
        # 执行统计检验
        t_stats, p_values, corrected_p_matrix, significant_connections, cohens_d = perform_group_comparison(
            positive_data, negative_data, alpha=0.05, correction_method='fdr_bh'
        )
        
        # 保存结果
        result = {
            't_stats': t_stats,
            'p_values': p_values,
            'corrected_p_matrix': corrected_p_matrix,
            'significant_connections': significant_connections,
            'cohens_d': cohens_d
        }
        all_results.append(result)
        
        if cvib == 0:
            results_cvib0.append(result)
        
        # 保存显著连接数据
        sig_df = save_connection_data_to_csv(result, cvib, pair_name, 'significant')
        
        # 保存高效应量连接数据
        high_effect_df = save_connection_data_to_csv(result, cvib, pair_name, 'high_effect')
        
        # 筛选高显著和高效应量连接
        high_quality_df = filter_high_quality_connections(result, p_threshold=0.01, effect_threshold=0.8)
        if len(high_quality_df) > 0:
            high_quality_df.to_csv(f'cvib{cvib}_{pair_name}_high_quality_connections.csv', index=False)
            all_high_quality_dfs.append({
                'cvib': cvib,
                'pair': pair_name,
                'df': high_quality_df
            })
    
    # 绘制t值热图（6张子图）
    fig_t = create_combined_figure(all_results, cvib_list, pair_names, 't_stats')
    fig_t.savefig('combined_t_stats_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_t)
    
    # 绘制显著连接热图（6张子图）
    fig_sig = create_combined_figure(all_results, cvib_list, pair_names, 'significant')
    fig_sig.savefig('combined_significant_connections_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_sig)
    
    # 绘制高效应量热图（6张子图）
    fig_effect = create_combined_figure(all_results, cvib_list, pair_names, 'high_effect')
    fig_effect.savefig('combined_high_effect_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_effect)
    
    # 保存所有高显著高效应量连接到单个CSV
    all_hq_connections = []
    for item in all_high_quality_dfs:
        df_copy = item['df'].copy()
        df_copy['cvib'] = item['cvib']
        df_copy['comparison'] = item['pair']
        all_hq_connections.append(df_copy)
    
    if all_hq_connections:
        combined_hq_df = pd.concat(all_hq_connections, ignore_index=True)
        combined_hq_df.to_csv('all_high_quality_connections.csv', index=False)
        print(f"Total high-quality connections found: {len(combined_hq_df)}")
    
    # 找出cvib=0的三个类别对共同存在的显著连接
    common_sig_df = find_common_significant_connections(results_cvib0, pair_names)
    if len(common_sig_df) > 0:
        print(f"Found {len(common_sig_df)} common significant connections for cvib=0")
    else:
        print("No common significant connections found for cvib=0")
    
    print("\nAnalysis complete! All results saved in:", output_dir)
    
    return all_results, combined_hq_df if all_hq_connections else pd.DataFrame(), common_sig_df

if __name__ == "__main__":
    # 运行主分析
    all_results, high_quality_df, common_sig_df = main_analysis()
    
    print("\nAnalysis Summary:")
    print(f"- Total comparisons performed: {len(all_results)}")
    print(f"- High-quality connections identified: {len(high_quality_df)}")
    print(f"- Common significant connections (cvib=0): {len(common_sig_df)}")