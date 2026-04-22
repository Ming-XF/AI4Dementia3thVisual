import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import pickle
import pandas as pd

import pdb

def load_and_preprocess_data(cvib=0, positive_class=0, negative_class=3):
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

    # pdb.set_trace()
    positive_data = data[cvib][positive_class]
    negative_data = data[cvib][negative_class]

    
    
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
        
        # 计算组均值
        pos_mean = np.mean(positive_conn)
        neg_mean = np.mean(negative_conn)
        mean_diff = pos_mean - neg_mean
        
        # 存储结果
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

def visualize_pvalue_heatmap(corrected_p_matrix, significant_connections, 
                           save_fig=False, fig_name='pvalue_heatmap.png'):
    """
    绘制P值热图，并用鲜艳颜色标记显著连接
    
    参数:
    - corrected_p_matrix: 校正后的p值矩阵 (68x68)
    - significant_connections: 显著连接布尔矩阵 (68x68)
    - save_fig: 是否保存图像
    - fig_name: 保存的文件名
    """

    plt.rcParams.update({
        'font.size': 32        # 图形标题字体大小
    })
    
    n_regions = corrected_p_matrix.shape[0]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # 对P值进行-log10转换以便更好地可视化
    log_p = -np.log10(corrected_p_matrix + 1e-10)  # 避免log(0)
    log_p[log_p > 10] = 10  # 限制范围便于可视化
    
    # 创建掩码，只显示下三角
    mask = np.triu(np.ones_like(log_p, dtype=bool), k=1)
    
    # 创建带掩码的数据
    masked_data = np.ma.masked_where(mask, log_p)
    
    # 绘制热图
    im = ax.imshow(masked_data, cmap='hot', aspect='auto', vmin=0, vmax=5,
                   interpolation='nearest')
    
    # 在显著连接上标记鲜艳的标记（红色星号）
    for i in range(n_regions):
        for j in range(n_regions):
            if i > j and significant_connections[i, j]:  # 只在下三角标记
                ax.text(j, i, '★', ha='center', va='center', 
                       color='lime', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='red', 
                                alpha=0.7, pad=0.5))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('-log10(p)', fontsize=32)
    
    # 设置标题和标签
    # ax.set_title('功能连接组间差异显著性热图\n(下三角: -log10(P值), ★: 显著连接)', 
    #             fontsize=14, pad=20)
    ax.set_xlabel('Brain Region', fontsize=32)
    ax.set_ylabel('Brain Region', fontsize=32)
    
    # 计算并显示统计信息
    n_sig = np.sum(significant_connections) / 2  # 除以2得到实际连接数
    total_connections = n_regions * (n_regions - 1) / 2
    sig_percent = (n_sig / total_connections) * 100
    
    # 添加统计信息文本框
    # stats_text = f'显著连接数: {int(n_sig)} / {int(total_connections)}\n显著性比例: {sig_percent:.2f}%'
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图像已保存为: {fig_name}")
    
    return fig
def play_static(cvib=0, positive_class=2, negative_class=3):
    """
    主函数
    """
    # 1. 加载数据
    print("正在加载数据...")
    positive_data, negative_data = load_and_preprocess_data(cvib=cvib, positive_class=positive_class, negative_class=negative_class)
    
    # 2. 执行组间比较（现在返回更多信息）
    t_stats, p_values, corrected_p_matrix, significant_connections = perform_group_comparison(
        positive_data, negative_data, 
        alpha=0.05, 
        correction_method='fdr_bh'  # 可选: 'fdr_bh', 'bonferroni', None
    )

    positive_means = positive_data.mean(axis=0)
    negative_means = negative_data.mean(axis=0)

    sig_indices = np.where(significant_connections)
    if len(sig_indices[0]) == 0:
        print("无显著连接，程序结束")
        return None, None

    visualize_pvalue_heatmap(corrected_p_matrix, significant_connections, fig_name='pvalue_heatmap.png')
    
    
    # 打印前10个最显著的连接（包含均值信息）
    print(f"\n前100个最显著的连接 (按t统计量绝对值排序):")
    sig_indices = np.where(significant_connections)
    sig_t_values = t_stats[sig_indices]
    sig_pos_means = positive_means[sig_indices]
    sig_neg_means = negative_means[sig_indices]
    
    # 创建连接列表并排序
    connections = list(zip(sig_indices[0], sig_indices[1], sig_t_values, 
                          sig_pos_means, sig_neg_means))
    connections_sorted = sorted(connections, key=lambda x: abs(x[2]), reverse=True)
    
    print(f"{'连接':<20} {'t统计量':<12} {'p值':<12} {'正组均值':<12} {'负组均值':<12} {'差异':<12}")
    print("-" * 70)
    
    count = 0
    printed = 0
    while printed < 100 and count < len(connections_sorted):
        region_i, region_j, t_val, pos_mean, neg_mean = connections_sorted[count]
        if region_i < region_j:  # 避免重复打印
            p_val = corrected_p_matrix[region_i, region_j]
            mean_diff = pos_mean - neg_mean
            print(f"{region_i}-{region_j:<8} {t_val:>10.4f} {p_val:>11.6f} "
                  f"{pos_mean:>11.4f} {neg_mean:>11.4f} {mean_diff:>11.4f}")
            printed += 1
        count += 1

    return positive_means, negative_means

if __name__ == "__main__":
    pd1, nd1 = play_static(cvib=0, positive_class=0, negative_class=3)
    pd2, nd2 = play_static(cvib=0, positive_class=2, negative_class=3)
    pd3, nd3 = play_static(cvib=0, positive_class=1, negative_class=3)

    # pdb.set_trace()

    # pdb.set_trace()






    print("Finish")
    