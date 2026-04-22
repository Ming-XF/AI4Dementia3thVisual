import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import pickle
import pandas as pd

import pdb

def load_and_preprocess_data():
    """
    加载正负样本数据
    """
    # 加载数据
    # positive_data = np.load('./data/vae_positive_data.npy')  # 形状: (n_positive, 68, 68)
    # negative_data = np.load('./data/vae_negtive_data.npy')   # 形状: (n_negative, 68, 68)

    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    # pdb.set_trace()
    positive_data = data[0][0]
    negative_data = data[0][3]

    # positive_data = data[1][0]
    # negative_data = data[1][3]

    
    
    print(f"正样本数量: {positive_data.shape[0]}")
    print(f"负样本数量: {negative_data.shape[0]}")
    print(f"每个FC图的形状: {positive_data.shape[1:]}")
    
    return positive_data, negative_data

def perform_group_comparison(positive_data, negative_data, alpha=0.05, correction_method='fdr_bh'):
    """
    对每个功能连接进行组间t检验，并进行多重比较校正
    """
    n_regions = positive_data.shape[1]  # 脑区数量，应该是68
    
    # 初始化存储结果的数组
    t_stats = np.zeros((n_regions, n_regions))
    p_values = np.zeros((n_regions, n_regions))
    significant_connections = np.zeros((n_regions, n_regions), dtype=bool)
    
    # 新增：存储组均值
    positive_means = np.zeros((n_regions, n_regions))
    negative_means = np.zeros((n_regions, n_regions))
    mean_differences = np.zeros((n_regions, n_regions))
    
    # 提取所有连接的下三角索引（避免重复，因为FC矩阵是对称的）
    lower_tri_indices = np.tril_indices(n_regions, k=-1)
    
    # 存储所有连接的p值用于多重比较校正
    all_p_values = []
    connection_indices = []
    
    print("正在进行组间统计检验...")
    
    # 对每个连接进行t检验
    for i, j in zip(lower_tri_indices[0], lower_tri_indices[1]):
        # 提取两组在该连接上的值
        positive_conn = positive_data[:, i, j]
        negative_conn = negative_data[:, i, j]
        
        # 执行独立样本t检验
        t_stat, p_val = ttest_ind(positive_conn, negative_conn, equal_var=False)
        
        # 计算组均值
        pos_mean = np.mean(positive_conn)
        neg_mean = np.mean(negative_conn)
        mean_diff = pos_mean - neg_mean
        
        # 存储结果
        t_stats[i, j] = t_stat
        t_stats[j, i] = t_stat  # 对称矩阵
        p_values[i, j] = p_val
        p_values[j, i] = p_val
        
        positive_means[i, j] = pos_mean
        positive_means[j, i] = pos_mean
        negative_means[i, j] = neg_mean
        negative_means[j, i] = neg_mean
        mean_differences[i, j] = mean_diff
        mean_differences[j, i] = mean_diff
        
        all_p_values.append(p_val)
        connection_indices.append((i, j))
    
    # 多重比较校正
    print(f"正在进行多重比较校正 ({correction_method})...")
    all_p_values = np.array(all_p_values)
    
    if correction_method == 'fdr_bh':
        # FDR Benjamini-Hochberg校正
        rejected, corrected_pvals, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
    elif correction_method == 'bonferroni':
        # Bonferroni校正
        rejected, corrected_pvals, _, _ = multipletests(all_p_values, alpha=alpha, method='bonferroni')
    else:
        # 不进行校正
        rejected = all_p_values < alpha
        corrected_pvals = all_p_values
    
    # 标记显著连接
    for idx, ((i, j), is_sig) in enumerate(zip(connection_indices, rejected)):
        if is_sig:
            significant_connections[i, j] = True
            significant_connections[j, i] = True
    
    # 将校正后的p值填充到矩阵中
    corrected_p_matrix = np.full((n_regions, n_regions), 1.0)
    for idx, (i, j) in enumerate(connection_indices):
        corrected_p_matrix[i, j] = corrected_pvals[idx]
        corrected_p_matrix[j, i] = corrected_pvals[idx]
    
    return t_stats, p_values, corrected_p_matrix, significant_connections, positive_means, negative_means, mean_differences

def calculate_significant_connection_stats(significant_connections, positive_means, negative_means, mean_differences):
    """
    计算显著连接的平均功能连接强度统计信息
    """
    # 获取显著连接的索引
    sig_indices = np.where(significant_connections)
    
    # 提取显著连接的值
    sig_pos_means = positive_means[sig_indices]
    sig_neg_means = negative_means[sig_indices]
    sig_mean_diffs = mean_differences[sig_indices]
    
    # 由于矩阵对称，只取一半避免重复
    unique_sig_connections = []
    for i, j, pos_mean, neg_mean, mean_diff in zip(sig_indices[0], sig_indices[1], 
                                                   sig_pos_means, sig_neg_means, sig_mean_diffs):
        if i < j:  # 只取下三角
            unique_sig_connections.append({
                'region_i': i,
                'region_j': j,
                'positive_mean': pos_mean,
                'negative_mean': neg_mean,
                'mean_difference': mean_diff
            })
    
    # 转换为DataFrame便于分析
    stats_df = pd.DataFrame(unique_sig_connections)

    # 计算总体统计
    overall_stats = {
        'total_significant_connections': len(unique_sig_connections),
        'avg_positive_mean': np.mean(sig_pos_means),
        'avg_negative_mean': np.mean(sig_neg_means),
        'avg_mean_difference': np.mean(sig_mean_diffs),
        'std_positive_mean': np.std(sig_pos_means),
        'std_negative_mean': np.std(sig_neg_means),
        'std_mean_difference': np.std(sig_mean_diffs),
        'min_positive_mean': np.min(sig_pos_means),
        'max_positive_mean': np.max(sig_pos_means),
        'min_negative_mean': np.min(sig_neg_means),
        'max_negative_mean': np.max(sig_neg_means),
        'min_difference': np.min(sig_mean_diffs),
        'max_difference': np.max(sig_mean_diffs),
    }
    
    # 根据均值差异方向分类
    increased_in_positive = stats_df[stats_df['mean_difference'] > 0]
    decreased_in_positive = stats_df[stats_df['mean_difference'] < 0]
    
    overall_stats['connections_increased_in_positive'] = len(increased_in_positive)
    overall_stats['connections_decreased_in_positive'] = len(decreased_in_positive)
    
    if len(increased_in_positive) > 0:
        overall_stats['avg_increase_in_positive'] = np.mean(increased_in_positive['mean_difference'])
        overall_stats['avg_pos_mean_increased'] = np.mean(increased_in_positive['positive_mean'])
        overall_stats['avg_neg_mean_increased'] = np.mean(increased_in_positive['negative_mean'])
    
    if len(decreased_in_positive) > 0:
        overall_stats['avg_decrease_in_positive'] = np.mean(decreased_in_positive['mean_difference'])
        overall_stats['avg_pos_mean_decreased'] = np.mean(decreased_in_positive['positive_mean'])
        overall_stats['avg_neg_mean_decreased'] = np.mean(decreased_in_positive['negative_mean'])
    
    return stats_df, overall_stats

def visualize_results(t_stats, p_values, significant_connections, corrected_p_matrix, 
                     positive_means, negative_means, mean_differences):
    """
    可视化统计检验结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 绘制t统计量矩阵
    im1 = axes[0, 0].imshow(t_stats, cmap='RdBu_r', vmin=-np.max(np.abs(t_stats)), vmax=np.max(np.abs(t_stats)))
    axes[0, 0].set_title('T-statistics Matrix')
    axes[0, 0].set_xlabel('Brain Region')
    axes[0, 0].set_ylabel('Brain Region')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 绘制原始p值矩阵（-log10转换）
    p_log = -np.log10(p_values + 1e-10)  # 避免log(0)
    im2 = axes[0, 1].imshow(p_log, cmap='viridis')
    axes[0, 1].set_title('Original -log10(p-values)')
    axes[0, 1].set_xlabel('Brain Region')
    axes[0, 1].set_ylabel('Brain Region')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 绘制校正后p值矩阵（-log10转换）
    corrected_p_log = -np.log10(corrected_p_matrix + 1e-10)
    im3 = axes[0, 2].imshow(corrected_p_log, cmap='viridis')
    axes[0, 2].set_title('Corrected -log10(p-values)')
    axes[0, 2].set_xlabel('Brain Region')
    axes[0, 2].set_ylabel('Brain Region')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 绘制显著连接矩阵
    im4 = axes[1, 0].imshow(significant_connections, cmap='binary')
    axes[1, 0].set_title('Significant Connections')
    axes[1, 0].set_xlabel('Brain Region')
    axes[1, 0].set_ylabel('Brain Region')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 绘制组均值差异矩阵
    vmax = np.max(np.abs(mean_differences))
    im5 = axes[1, 1].imshow(mean_differences, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('Mean Difference (Positive - Negative)')
    axes[1, 1].set_xlabel('Brain Region')
    axes[1, 1].set_ylabel('Brain Region')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 绘制正组均值矩阵
    im6 = axes[1, 2].imshow(positive_means, cmap='viridis')
    axes[1, 2].set_title('Positive Group Means')
    axes[1, 2].set_xlabel('Brain Region')
    axes[1, 2].set_ylabel('Brain Region')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('./output/fc_vis_with_means.png', dpi=300, bbox_inches='tight')
    
    # 额外绘制显著连接的均值分布图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # 获取显著连接的值
    sig_indices = np.where(significant_connections)
    sig_pos_means = positive_means[sig_indices]
    sig_neg_means = negative_means[sig_indices]
    sig_mean_diffs = mean_differences[sig_indices]
    
    # 正组显著连接均值分布
    axes2[0].hist(sig_pos_means, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes2[0].set_xlabel('Functional Connectivity Strength')
    axes2[0].set_ylabel('Frequency')
    axes2[0].set_title('Distribution of Positive Group Means\nin Significant Connections')
    axes2[0].axvline(x=np.mean(sig_pos_means), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(sig_pos_means):.3f}')
    axes2[0].legend()
    
    # 负组显著连接均值分布
    axes2[1].hist(sig_neg_means, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes2[1].set_xlabel('Functional Connectivity Strength')
    axes2[1].set_ylabel('Frequency')
    axes2[1].set_title('Distribution of Negative Group Means\nin Significant Connections')
    axes2[1].axvline(x=np.mean(sig_neg_means), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(sig_neg_means):.3f}')
    axes2[1].legend()
    
    # 均值差异分布
    axes2[2].hist(sig_mean_diffs, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes2[2].set_xlabel('Mean Difference (Positive - Negative)')
    axes2[2].set_ylabel('Frequency')
    axes2[2].set_title('Distribution of Mean Differences\nin Significant Connections')
    axes2[2].axvline(x=np.mean(sig_mean_diffs), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(sig_mean_diffs):.3f}')
    axes2[2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes2[2].legend()
    
    plt.tight_layout()
    plt.savefig('./output/significant_means_distribution.png', dpi=300, bbox_inches='tight')

def save_significant_connections(significant_connections, t_stats, positive_means, 
                                negative_means, mean_differences, overall_stats, 
                                output_file='./output/significant_connections.npy'):
    """
    保存显著连接信息，包括均值信息
    """
    # 获取显著连接的索引
    sig_indices = np.where(significant_connections)
    
    # 创建详细的结果字典
    results = {
        'significant_mask': significant_connections,
        'significant_indices': list(zip(sig_indices[0], sig_indices[1])),
        't_statistics': t_stats,
        'positive_means': positive_means,
        'negative_means': negative_means,
        'mean_differences': mean_differences,
        'num_significant': len(sig_indices[0]) // 2,  # 除以2因为矩阵是对称的
        'overall_statistics': overall_stats
    }
    
    # 保存结果
    np.save(output_file, results, allow_pickle=True)
    
    # 同时保存为文本文件以便查看
    txt_file = output_file.replace('.npy', '_summary.txt')
    with open(txt_file, 'w') as f:
        f.write("显著连接统计摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总显著连接数: {overall_stats['total_significant_connections']}\n")
        f.write(f"正组中增强的连接数: {overall_stats['connections_increased_in_positive']}\n")
        f.write(f"正组中减弱的连接数: {overall_stats['connections_decreased_in_positive']}\n\n")
        
        f.write("均值统计:\n")
        f.write(f"  所有显著连接的正组平均功能连接强度: {overall_stats['avg_positive_mean']:.4f} ± {overall_stats['std_positive_mean']:.4f}\n")
        f.write(f"  所有显著连接的负组平均功能连接强度: {overall_stats['avg_negative_mean']:.4f} ± {overall_stats['std_negative_mean']:.4f}\n")
        f.write(f"  所有显著连接的平均差异: {overall_stats['avg_mean_difference']:.4f} ± {overall_stats['std_mean_difference']:.4f}\n\n")
        
        if 'avg_increase_in_positive' in overall_stats:
            f.write("在正组中增强的连接:\n")
            f.write(f"  平均增强幅度: {overall_stats['avg_increase_in_positive']:.4f}\n")
            f.write(f"  正组平均功能连接强度: {overall_stats['avg_pos_mean_increased']:.4f}\n")
            f.write(f"  负组平均功能连接强度: {overall_stats['avg_neg_mean_increased']:.4f}\n\n")
        
        if 'avg_decrease_in_positive' in overall_stats:
            f.write("在正组中减弱的连接:\n")
            f.write(f"  平均减弱幅度: {overall_stats['avg_decrease_in_positive']:.4f}\n")
            f.write(f"  正组平均功能连接强度: {overall_stats['avg_pos_mean_decreased']:.4f}\n")
            f.write(f"  负组平均功能连接强度: {overall_stats['avg_neg_mean_decreased']:.4f}\n\n")
        
        f.write("极值信息:\n")
        f.write(f"  正组功能连接强度范围: [{overall_stats['min_positive_mean']:.4f}, {overall_stats['max_positive_mean']:.4f}]\n")
        f.write(f"  负组功能连接强度范围: [{overall_stats['min_negative_mean']:.4f}, {overall_stats['max_negative_mean']:.4f}]\n")
        f.write(f"  差异范围: [{overall_stats['min_difference']:.4f}, {overall_stats['max_difference']:.4f}]\n")
    
    print(f"显著连接结果已保存到: {output_file}")
    print(f"统计摘要已保存到: {txt_file}")
    print(f"发现 {results['num_significant']} 个显著连接")
    
    return results

def main():
    """
    主函数
    """
    # 1. 加载数据
    print("正在加载数据...")
    positive_data, negative_data = load_and_preprocess_data()
    
    # 2. 执行组间比较（现在返回更多信息）
    t_stats, p_values, corrected_p_matrix, significant_connections, \
    positive_means, negative_means, mean_differences = perform_group_comparison(
        positive_data, negative_data, 
        alpha=0.05, 
        correction_method='fdr_bh'  # 可选: 'fdr_bh', 'bonferroni', None
    )
    
    # 3. 计算显著连接的均值统计
    print("正在计算显著连接的均值统计...")
    stats_df, overall_stats = calculate_significant_connection_stats(
        significant_connections, positive_means, negative_means, mean_differences
    )
    
    # 4. 可视化结果
    print("正在生成可视化...")
    visualize_results(t_stats, p_values, significant_connections, 
                     corrected_p_matrix, positive_means, negative_means, mean_differences)
    
    # 5. 保存显著连接（包含均值信息）
    results = save_significant_connections(
        significant_connections, t_stats, positive_means, 
        negative_means, mean_differences, overall_stats
    )
    
    # 6. 打印统计信息
    n_total_connections = (68 * 67) // 2  # 下三角连接总数
    n_sig_connections = results['num_significant']
    
    print(f"\n统计检验结果摘要:")
    print(f"总连接数: {n_total_connections}")
    print(f"显著连接数: {n_sig_connections}")
    print(f"显著连接比例: {n_sig_connections/n_total_connections*100:.2f}%")
    
    print(f"\n均值差异统计:")
    print(f"所有显著连接的正组平均功能连接强度: {overall_stats['avg_positive_mean']:.4f} ± {overall_stats['std_positive_mean']:.4f}")
    print(f"所有显著连接的负组平均功能连接强度: {overall_stats['avg_negative_mean']:.4f} ± {overall_stats['std_negative_mean']:.4f}")
    print(f"所有显著连接的平均差异: {overall_stats['avg_mean_difference']:.4f} ± {overall_stats['std_mean_difference']:.4f}")
    
    print(f"\n连接方向分类:")
    print(f"正组中增强的连接数: {overall_stats['connections_increased_in_positive']}")
    print(f"正组中减弱的连接数: {overall_stats['connections_decreased_in_positive']}")
    
    # 打印前10个最显著的连接（包含均值信息）
    print(f"\n前10个最显著的连接 (按t统计量绝对值排序):")
    sig_indices = np.where(significant_connections)
    sig_t_values = t_stats[sig_indices]
    sig_pos_means = positive_means[sig_indices]
    sig_neg_means = negative_means[sig_indices]
    
    # 创建连接列表并排序
    connections = list(zip(sig_indices[0], sig_indices[1], sig_t_values, 
                          sig_pos_means, sig_neg_means))
    connections_sorted = sorted(connections, key=lambda x: abs(x[2]), reverse=True)
    
    print(f"{'连接':<10} {'t统计量':<12} {'p值':<12} {'正组均值':<12} {'负组均值':<12} {'差异':<12}")
    print("-" * 70)
    
    count = 0
    printed = 0
    while printed < 10 and count < len(connections_sorted):
        region_i, region_j, t_val, pos_mean, neg_mean = connections_sorted[count]
        if region_i < region_j:  # 避免重复打印
            p_val = corrected_p_matrix[region_i, region_j]
            mean_diff = pos_mean - neg_mean
            print(f"{region_i}-{region_j:<8} {t_val:>10.4f} {p_val:>11.6f} "
                  f"{pos_mean:>11.4f} {neg_mean:>11.4f} {mean_diff:>11.4f}")
            printed += 1
        count += 1

if __name__ == "__main__":
    main()