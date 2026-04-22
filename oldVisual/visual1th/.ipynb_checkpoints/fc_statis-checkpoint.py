import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import pickle

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
    # positive_data = data[0][0]
    # negative_data = data[0][3]

    positive_data = data[1][0]
    negative_data = data[1][3]

    
    
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
        
        t_stats[i, j] = t_stat
        t_stats[j, i] = t_stat  # 对称矩阵
        p_values[i, j] = p_val
        p_values[j, i] = p_val
        
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
    
    return t_stats, p_values, corrected_p_matrix, significant_connections

def visualize_results(t_stats, p_values, significant_connections, corrected_p_matrix):
    """
    可视化统计检验结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
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
    im3 = axes[1, 0].imshow(corrected_p_log, cmap='viridis')
    axes[1, 0].set_title('Corrected -log10(p-values)')
    axes[1, 0].set_xlabel('Brain Region')
    axes[1, 0].set_ylabel('Brain Region')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 绘制显著连接矩阵
    im4 = axes[1, 1].imshow(significant_connections, cmap='binary')
    axes[1, 1].set_title('Significant Connections')
    axes[1, 1].set_xlabel('Brain Region')
    axes[1, 1].set_ylabel('Brain Region')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('./output/fc_vis.png')

def save_significant_connections(significant_connections, t_stats, output_file='./output/significant_connections.npy'):
    """
    保存显著连接信息
    """
    # 获取显著连接的索引
    sig_indices = np.where(significant_connections)
    
    # 创建结果字典
    results = {
        'significant_mask': significant_connections,
        'significant_indices': list(zip(sig_indices[0], sig_indices[1])),
        't_statistics': t_stats,
        'num_significant': len(sig_indices[0]) // 2  # 除以2因为矩阵是对称的
    }
    
    # 保存结果
    np.save(output_file, results)
    print(f"显著连接结果已保存到: {output_file}")
    print(f"发现 {results['num_significant']} 个显著连接")
    
    return results

def main():
    """
    主函数
    """
    # 1. 加载数据
    print("正在加载数据...")
    positive_data, negative_data = load_and_preprocess_data()
    
    # 2. 执行组间比较
    t_stats, p_values, corrected_p_matrix, significant_connections = perform_group_comparison(
        positive_data, negative_data, 
        alpha=0.05, 
        correction_method='fdr_bh'  # 可选: 'fdr_bh', 'bonferroni', None
    )
    
    # 3. 可视化结果
    print("正在生成可视化...")
    visualize_results(t_stats, p_values, significant_connections, corrected_p_matrix)
    
    # 4. 保存显著连接
    results = save_significant_connections(significant_connections, t_stats)
    
    # 5. 打印一些统计信息
    n_total_connections = (68 * 67) // 2  # 下三角连接总数
    n_sig_connections = results['num_significant']
    
    print(f"\n统计检验结果摘要:")
    print(f"总连接数: {n_total_connections}")
    print(f"显著连接数: {n_sig_connections}")
    print(f"显著连接比例: {n_sig_connections/n_total_connections*100:.2f}%")
    
    # 打印前10个最显著的连接
    print(f"\n前10个最显著的连接 (按t统计量绝对值排序):")
    sig_indices = np.where(significant_connections)
    sig_t_values = t_stats[sig_indices]
    
    # 创建连接列表并排序
    connections = list(zip(sig_indices[0], sig_indices[1], sig_t_values))
    connections_sorted = sorted(connections, key=lambda x: abs(x[2]), reverse=True)
    
    for i, (region_i, region_j, t_val) in enumerate(connections_sorted[:10]):
        if region_i < region_j:  # 避免重复打印
            p_val = corrected_p_matrix[region_i, region_j]
            print(f"连接 {region_i}-{region_j}: t = {t_val:.4f}, p = {p_val:.6f}")

if __name__ == "__main__":
    main()