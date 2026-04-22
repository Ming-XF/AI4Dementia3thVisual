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
                       color='lime', fontsize=4, fontweight='bold',
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


def analyze_common_connections(results_list, class_names=None):
    """
    分析多个类别对之间的共同显著连接
    
    参数:
    - results_list: 列表，包含多个play_static返回的结果元组
    - class_names: 类别对名称列表，用于打印标识
    
    返回:
    - common_connections: 包含共同连接信息的DataFrame
    """
    if class_names is None:
        class_names = [f"Pair{i+1}" for i in range(len(results_list))]
    
    # 提取每个类别对的显著连接集合
    sig_connection_sets = []
    for sig_indices, _, _, _ in results_list:
        if sig_indices is None:
            sig_connection_sets.append(set())
        else:
            # 创建连接对集合（确保(i,j)格式统一，i<j）
            connections = set()
            for i, j in zip(sig_indices[0], sig_indices[1]):
                if i < j:
                    connections.add((i, j))
                else:
                    connections.add((j, i))
            sig_connection_sets.append(connections)
    
    # 找出所有类别对中的共同连接
    common_connections = set.intersection(*sig_connection_sets) if sig_connection_sets else set()
    
    print("\n" + "="*100)
    print(f"共同显著连接分析")
    print(f"类别对: {', '.join(class_names)}")
    print(f"共同显著连接数量: {len(common_connections)}")
    print("="*100)
    
    if len(common_connections) == 0:
        print("未找到共同显著连接")
        return pd.DataFrame()
    
    # 为每个共同连接收集详细信息
    connection_details = []
    
    for conn in sorted(common_connections):
        i, j = conn
        detail = {'connection': f"{i}-{j}"}
        
        # 为每个类别对收集统计信息
        for pair_idx, (sig_indices, sig_t_values, sig_pos_means, sig_neg_means) in enumerate(results_list):
            if sig_indices is not None:
                # 找到该连接在结果中的索引
                for idx, (si, sj) in enumerate(zip(sig_indices[0], sig_indices[1])):
                    if (si == i and sj == j) or (si == j and sj == i):
                        detail[f'{class_names[pair_idx]}_t'] = sig_t_values[idx]
                        detail[f'{class_names[pair_idx]}_pos_mean'] = sig_pos_means[idx]
                        detail[f'{class_names[pair_idx]}_neg_mean'] = sig_neg_means[idx]
                        detail[f'{class_names[pair_idx]}_mean_diff'] = sig_pos_means[idx] - sig_neg_means[idx]
                        break
        
        connection_details.append(detail)
    
    # 转换为DataFrame便于查看
    df_details = pd.DataFrame(connection_details)
    
    # 打印详细信息
    print(f"\n共同显著连接详细信息:")
    print("-" * 100)
    
    # 构建打印格式
    header = f"{'连接':<10}"
    for name in class_names:
        header += f" {name}_t值:<12 {name}_正均值:<10 {name}_负均值:<10 {name}_差异:<10"
    print(header)
    print("-" * 150)
    
    for detail in connection_details:
        line = f"{detail['connection']:<10}"
        for name in class_names:
            line += (f" {detail.get(f'{name}_t', 'N/A'):<12.4f} "
                    f"{detail.get(f'{name}_pos_mean', 'N/A'):<10.4f} "
                    f"{detail.get(f'{name}_neg_mean', 'N/A'):<10.4f} "
                    f"{detail.get(f'{name}_mean_diff', 'N/A'):<10.4f}")
        print(line)
    
    # 创建汇总统计
    print("\n" + "="*100)
    print("汇总统计:")
    print("-"*100)
    
    for detail in connection_details:
        conn = detail['connection']
        print(f"\n连接 {conn}:")
        
        for name in class_names:
            if f'{name}_t' in detail:
                print(f"  {name}: t={detail[f'{name}_t']:.4f}, "
                      f"正均值={detail[f'{name}_pos_mean']:.4f}, "
                      f"负均值={detail[f'{name}_neg_mean']:.4f}, "
                      f"差异={detail[f'{name}_mean_diff']:.4f}")
    
    return df_details

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
        return None, None, None, None

    visualize_pvalue_heatmap(corrected_p_matrix, significant_connections, fig_name='{}_pvalue_heatmap_{}-{}.png'.format(cvib, positive_class, negative_class))
    
    
    # 打印前10个最显著的连接（包含均值信息）
    print(f"\n前100个最显著的连接 (按t统计量绝对值排序):")
    sig_indices = np.where(significant_connections)
    sig_t_values = t_stats[sig_indices]
    sig_pos_means = positive_means[sig_indices]
    sig_neg_means = negative_means[sig_indices]


    return sig_indices, sig_t_values, sig_pos_means, sig_neg_means
    
    # # 创建连接列表并排序
    # connections = list(zip(sig_indices[0], sig_indices[1], sig_t_values, 
    #                       sig_pos_means, sig_neg_means))
    # connections_sorted = sorted(connections, key=lambda x: abs(x[2]), reverse=True)
    
    # print(f"{'连接':<20} {'t统计量':<12} {'p值':<12} {'正组均值':<12} {'负组均值':<12} {'差异':<12}")
    # print("-" * 70)
    
    # count = 0
    # printed = 0
    # while printed < 100 and count < len(connections_sorted):
    #     region_i, region_j, t_val, pos_mean, neg_mean = connections_sorted[count]
    #     if region_i < region_j:  # 避免重复打印
    #         p_val = corrected_p_matrix[region_i, region_j]
    #         mean_diff = pos_mean - neg_mean
    #         print(f"{region_i}-{region_j:<8} {t_val:>10.4f} {p_val:>11.6f} "
    #               f"{pos_mean:>11.4f} {neg_mean:>11.4f} {mean_diff:>11.4f}")
    #         printed += 1
    #     count += 1

    # return positive_means, negative_means




if __name__ == "__main__":
    sig_indices1, sig_t_values1, sig_pos_means1, sig_neg_means1 = play_static(cvib=0, positive_class=0, negative_class=3)
    sig_indices2, sig_t_values2, sig_pos_means2, sig_neg_means2 = play_static(cvib=0, positive_class=2, negative_class=3)
    sig_indices3, sig_t_values3, sig_pos_means3, sig_neg_means3 = play_static(cvib=0, positive_class=1, negative_class=3)
    results_list = [
        (sig_indices1, sig_t_values1, sig_pos_means1, sig_neg_means1),
        (sig_indices2, sig_t_values2, sig_pos_means2, sig_neg_means2),
        (sig_indices3, sig_t_values3, sig_pos_means3, sig_neg_means3)
    ]
    
    class_names = ['Class-Nor_vs_AD', 'Class-Nor_vs_MCI', 'Class-Nor_vs_DSC']

    pdb.set_trace()
    # 分析共同显著连接
    common_df = analyze_common_connections(results_list, class_names)

    # pdb.set_trace()

    # pdb.set_trace()






    print("Finish")
    