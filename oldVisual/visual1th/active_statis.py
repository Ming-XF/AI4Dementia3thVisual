import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

import pdb

def load_and_preprocess_data():
    """
    加载正负样本数据并计算脑区激活
    """
    # 加载数据
    positive_data = np.load('./data/vae_positive_data.npy')  # 形状: (n_positive, 68, 68)
    negative_data = np.load('./data/vae_negtive_data.npy')   # 形状: (n_negative, 68, 68)
    
    print(f"正样本数量: {positive_data.shape[0]}")
    print(f"负样本数量: {negative_data.shape[0]}")
    print(f"每个FC图的形状: {positive_data.shape[1:]}")
    
    # 计算脑区激活：对每个样本的第二个维度取绝对值后取均值
    positive_activation = np.mean(np.abs(positive_data), axis=2)  # 形状: (n_positive, 68)
    negative_activation = np.mean(np.abs(negative_data), axis=2)  # 形状: (n_negative, 68)
    
    print(f"正样本脑区激活形状: {positive_activation.shape}")
    print(f"负样本脑区激活形状: {negative_activation.shape}")
    
    return positive_activation, negative_activation

def perform_group_comparison(positive_activation, negative_activation, alpha=0.05, correction_method='fdr_bh'):
    """
    对每个脑区进行组间t检验，并进行多重比较校正
    """
    n_regions = positive_activation.shape[1]  # 脑区数量
    
    # 初始化存储结果的数组
    t_stats = np.zeros(n_regions)
    p_values = np.zeros(n_regions)
    significant_regions = np.zeros(n_regions, dtype=bool)
    
    print("正在进行组间统计检验...")
    
    # 对每个脑区进行t检验
    for region in range(n_regions):
        # 提取两组在该脑区上的激活值
        positive_region = positive_activation[:, region]
        negative_region = negative_activation[:, region]
        
        # 执行独立样本t检验
        t_stat, p_val = ttest_ind(positive_region, negative_region, equal_var=False)
        
        t_stats[region] = t_stat
        p_values[region] = p_val
    
    # 多重比较校正
    print(f"正在进行多重比较校正 ({correction_method})...")
    
    if correction_method == 'fdr_bh':
        # FDR Benjamini-Hochberg校正
        rejected, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    elif correction_method == 'bonferroni':
        # Bonferroni校正
        rejected, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    else:
        # 不进行校正
        rejected = p_values < alpha
        corrected_pvals = p_values
    
    # 标记显著脑区
    significant_regions = rejected
    
    return t_stats, p_values, corrected_pvals, significant_regions

def visualize_results(t_stats, p_values, significant_regions, corrected_pvals):
    """
    可视化统计检验结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 脑区索引
    regions = np.arange(len(t_stats))
    
    # 1. 绘制t统计量条形图
    colors = ['red' if sig else 'blue' for sig in significant_regions]
    axes[0, 0].bar(regions, t_stats, color=colors, alpha=0.7)
    axes[0, 0].set_title('T-statistics for Each Brain Region')
    axes[0, 0].set_xlabel('Brain Region Index')
    axes[0, 0].set_ylabel('T-statistic')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. 绘制原始p值（-log10转换）
    p_log = -np.log10(p_values + 1e-10)  # 避免log(0)
    axes[0, 1].bar(regions, p_log, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[0, 1].set_title('Original -log10(p-values)')
    axes[0, 1].set_xlabel('Brain Region Index')
    axes[0, 1].set_ylabel('-log10(p-value)')
    axes[0, 1].legend()
    
    # 3. 绘制校正后p值（-log10转换）
    corrected_p_log = -np.log10(corrected_pvals + 1e-10)
    axes[1, 0].bar(regions, corrected_p_log, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[1, 0].set_title('Corrected -log10(p-values)')
    axes[1, 0].set_xlabel('Brain Region Index')
    axes[1, 0].set_ylabel('-log10(corrected p-value)')
    axes[1, 0].legend()
    
    # 4. 绘制显著脑区
    axes[1, 1].bar(regions, significant_regions.astype(int), color=colors, alpha=0.7)
    axes[1, 1].set_title('Significant Brain Regions')
    axes[1, 1].set_xlabel('Brain Region Index')
    axes[1, 1].set_ylabel('Significant (1) / Not Significant (0)')
    axes[1, 1].set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('./output/active_vis.png')

def plot_activation_comparison(positive_activation, negative_activation, significant_regions):
    """
    绘制两组脑区激活的对比图
    """
    n_regions = positive_activation.shape[1]
    
    # 计算均值和中位数
    positive_mean = np.mean(positive_activation, axis=0)
    negative_mean = np.mean(negative_activation, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 均值对比条形图
    x = np.arange(n_regions)
    width = 0.35
    
    axes[0].bar(x - width/2, positive_mean, width, label='Positive Group', alpha=0.7)
    axes[0].bar(x + width/2, negative_mean, width, label='Negative Group', alpha=0.7)
    
    # 标记显著脑区
    for i, sig in enumerate(significant_regions):
        if sig:
            axes[0].text(i, max(positive_mean[i], negative_mean[i]) + 0.01, '*', 
                        ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
    
    axes[0].set_xlabel('Brain Region Index')
    axes[0].set_ylabel('Mean Region Activation')
    axes[0].set_title('Mean Activation Comparison by Group')
    axes[0].legend()
    axes[0].set_xticks(x)
    
    # 2. 箱线图展示分布
    region_data = []
    labels = []
    for region in range(min(10, n_regions)):  # 只显示前10个脑区避免过于拥挤
        region_data.extend([positive_activation[:, region], negative_activation[:, region]])
        labels.extend([f'Region {region}\nPositive', f'Region {region}\nNegative'])
    
    axes[1].boxplot(region_data, labels=labels)
    axes[1].set_ylabel('Region Activation')
    axes[1].set_title('Activation Distribution (First 10 Regions)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('./output/active_contrast.png')

def save_significant_regions(t_stats, p_values, corrected_pvals, significant_regions, output_file='./output/significant_regions.npy'):
    """
    保存显著脑区信息
    """
    # 获取显著脑区的索引
    sig_indices = np.where(significant_regions)[0]
    
    # 创建结果字典
    results = {
        't_statistics': t_stats,
        'p_values': p_values,
        'corrected_p_values': corrected_pvals,
        'significant_regions': significant_regions,
        'significant_indices': sig_indices,
        'num_significant': len(sig_indices)
    }
    
    # 保存结果
    np.save(output_file, results)
    print(f"显著脑区结果已保存到: {output_file}")
    print(f"发现 {results['num_significant']} 个显著脑区")
    
    return results

def main():
    """
    主函数
    """
    # 1. 加载数据并计算脑区激活
    print("正在加载数据并计算脑区激活...")
    positive_activation, negative_activation = load_and_preprocess_data()
    
    # 2. 执行组间比较
    t_stats, p_values, corrected_pvals, significant_regions = perform_group_comparison(
        positive_activation, negative_activation, 
        alpha=0.05, 
        correction_method='fdr_bh'  # 可选: 'fdr_bh', 'bonferroni', None
    )
    
    # 3. 可视化结果
    print("正在生成可视化...")
    visualize_results(t_stats, p_values, significant_regions, corrected_pvals)
    plot_activation_comparison(positive_activation, negative_activation, significant_regions)
    
    # 4. 保存显著脑区
    results = save_significant_regions(t_stats, p_values, corrected_pvals, significant_regions)
    
    # 5. 打印统计信息
    n_total_regions = len(t_stats)
    n_sig_regions = results['num_significant']
    
    print(f"\n统计检验结果摘要:")
    print(f"总脑区数: {n_total_regions}")
    print(f"显著脑区数: {n_sig_regions}")
    print(f"显著脑区比例: {n_sig_regions/n_total_regions*100:.2f}%")
    
    # 打印所有显著脑区的详细信息
    if n_sig_regions > 0:
        print(f"\n显著脑区详细信息:")
        print("Region\tT-statistic\tP-value\tCorrected P")
        print("-" * 50)
        
        sig_indices = np.where(significant_regions)[0]
        for region in sig_indices:
            print(f"{region}\t{t_stats[region]:.4f}\t{p_values[region]:.6f}\t{corrected_pvals[region]:.6f}")
    else:
        print("未发现显著差异的脑区")
    
#     # 打印效应量（Cohen's d）
#     print(f"\n效应量分析 (Cohen's d):")
#     for region in range(min(5, n_total_regions)):  # 只显示前5个脑区
#         positive_data = positive_activation[:, region]
#         negative_data = negative_activation[:, region]
        
#         # 计算Cohen's d
#         mean_diff = np.mean(positive_data) - np.mean(negative_data)
#         pooled_std = np.sqrt((np.std(positive_data, ddof=1)**2 + np.std(negative_data, ddof=1)**2) / 2)
#         cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
#         sig_marker = "*" if significant_regions[region] else ""
#         print(f"Region {region}{sig_marker}: Cohen's d = {cohens_d:.3f}")

if __name__ == "__main__":
    main()