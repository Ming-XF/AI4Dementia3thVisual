# 给定原始样本数据，存储在名为“./data/ori_all_data.npy”的文件中，数量为n；给定处理好后的样本数据，存储在名为“./data/vae_all_data.npy”的文件中，数量也为n。每个样本数据为形状（68，68）的数组，该数组为每个样本的脑功能连接图。给定每个样本按照顺序对应的subject_id数据，存储在名为“./data/subject_ids.npy”的文件中，形状为（n）。然后，然后筛选出那些在组间存在显著差异的连接。请给出python代码
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def load_data():
    """加载数据"""
    print("正在加载数据...")
    
    # 加载原始数据
    ori_data = np.load('./data/ori_all_data.npy')
    print(f"原始数据形状: {ori_data.shape}")
    
    # 加载处理后的数据
    vae_data = np.load('./data/vae_all_data.npy')
    print(f"处理数据形状: {vae_data.shape}")
    
    # 加载subject_id
    subject_ids = np.load('./data/subject_ids.npy')
    print(f"subject_id形状: {subject_ids.shape}")
    
    return ori_data, vae_data, subject_ids

def calculate_within_subject_correlation(data, subject_ids):
    """
    计算每个subject内3个fc图之间的相关性
    
    参数:
    data: 形状为(n, 68, 68)的脑功能连接图数据
    subject_ids: 形状为(n)的subject_id数组
    
    返回:
    correlations: 每个subject内部的相关性矩阵列表
    mean_correlations: 每个subject内部相关性的平均值
    """
    unique_subjects = np.unique(subject_ids)
    correlations = []
    mean_correlations = []
    
    for subject in unique_subjects:
        # 获取该subject的所有样本索引
        subject_indices = np.where(subject_ids == subject)[0]
        
        if len(subject_indices) != 3:
            print(f"警告: subject {subject} 有 {len(subject_indices)} 个样本，期望3个")
            continue
        
        # 提取该subject的3个fc图
        subject_fc = data[subject_indices]  # 形状: (3, 68, 68)
        
        # 将每个fc图展平为向量
        fc_vectors = subject_fc.reshape(3, -1)  # 形状: (3, 68*68)
        
        # 计算3个fc图之间的相关性矩阵
        corr_matrix = np.corrcoef(fc_vectors)  # 形状: (3, 3)
        
        # 提取上三角部分（不包括对角线）
        upper_tri_indices = np.triu_indices(3, k=1)
        within_correlations = corr_matrix[upper_tri_indices]
        
        correlations.append(within_correlations)
        mean_correlations.append(np.mean(within_correlations))
    
    return correlations, mean_correlations

def analyze_correlation_comparison(ori_correlations, vae_correlations, 
                                 ori_mean_corrs, vae_mean_corrs):
    """分析处理前后相关性的变化"""
    
    # 将列表转换为数组以便统计分析
    ori_corr_array = np.concatenate(ori_correlations)
    vae_corr_array = np.concatenate(vae_correlations)
    
    ori_mean_array = np.array(ori_mean_corrs)
    vae_mean_array = np.array(vae_mean_corrs)
    
    print("\n" + "="*50)
    print("处理前后样本内相关性统计分析")
    print("="*50)
    
    print(f"\n原始数据:")
    print(f"  平均相关性: {np.mean(ori_mean_array):.4f} ± {np.std(ori_mean_array):.4f}")
    print(f"  中位数相关性: {np.median(ori_mean_array):.4f}")
    print(f"  最小值: {np.min(ori_mean_array):.4f}, 最大值: {np.max(ori_mean_array):.4f}")
    
    print(f"\n处理后数据:")
    print(f"  平均相关性: {np.mean(vae_mean_array):.4f} ± {np.std(vae_mean_array):.4f}")
    print(f"  中位数相关性: {np.median(vae_mean_array):.4f}")
    print(f"  最小值: {np.min(vae_mean_array):.4f}, 最大值: {np.max(vae_mean_array):.4f}")
    
    # 配对t检验
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(ori_mean_array, vae_mean_array)
    print(f"\n配对t检验:")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    
    # 计算改善的subject比例
    improvement = vae_mean_array - ori_mean_array
    improved_count = np.sum(improvement > 0)
    total_count = len(improvement)
    
    print(f"\n改善情况:")
    print(f"  改善的subject数量: {improved_count}/{total_count} ({improved_count/total_count*100:.1f}%)")
    print(f"  平均改善幅度: {np.mean(improvement):.4f}")
    
    return improvement

def plot_results(ori_mean_corrs, vae_mean_corrs, improvement):
    """绘制结果图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 处理前后相关性分布对比
    axes[0, 0].boxplot([ori_mean_corrs, vae_mean_corrs], 
                      labels=['origin data', 'processed data'])
    axes[0, 0].set_title('Comparison of intra-sample correlations before and after processing')
    axes[0, 0].set_ylabel('Average correlation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 相关性变化散点图
    axes[0, 1].scatter(ori_mean_corrs, vae_mean_corrs, alpha=0.6)
    min_val = min(np.min(ori_mean_corrs), np.min(vae_mean_corrs))
    max_val = max(np.max(ori_mean_corrs), np.max(vae_mean_corrs))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Correlation of original data')
    axes[0, 1].set_ylabel('Correlation of processed data')
    axes[0, 1].set_title('Scatter plot of correlation changes before and after processing')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 改善幅度分布
    axes[1, 0].hist(improvement, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', label='No change')
    axes[1, 0].axvline(np.mean(improvement), color='green', linestyle='--', label='Average improvement')
    axes[1, 0].set_xlabel('The extent of improvement in correlation')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of the extent of improvement in correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累积分布函数
    from scipy.stats import gaussian_kde
    x = np.linspace(min(ori_mean_corrs + vae_mean_corrs), 
                   max(ori_mean_corrs + vae_mean_corrs), 100)
    kde_ori = gaussian_kde(ori_mean_corrs)
    kde_vae = gaussian_kde(vae_mean_corrs)
    axes[1, 1].plot(x, kde_ori(x), label='origin data', linewidth=2)
    axes[1, 1].plot(x, kde_vae(x), label='processed data', linewidth=2)
    axes[1, 1].set_xlabel('Average correlation')
    axes[1, 1].set_ylabel('Probability density')
    axes[1, 1].set_title('Correlation distribution density estimation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/correlation_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def save_detailed_results(ori_mean_corrs, vae_mean_corrs, subject_ids, improvement):
    """保存详细结果到CSV文件"""
    unique_subjects = np.unique(subject_ids)
    
    results_df = pd.DataFrame({
        'subject_id': unique_subjects[:len(ori_mean_corrs)],
        'original_mean_correlation': ori_mean_corrs,
        'processed_mean_correlation': vae_mean_corrs,
        'improvement': improvement
    })
    
    results_df.to_csv('./output/correlation_analysis_results.csv', index=False)
    print(f"\n详细结果已保存到: ./correlation_analysis_results.csv")

def main():
    """主函数"""
    # 加载数据
    ori_data, vae_data, subject_ids = load_data()
    
    # 计算原始数据的样本内相关性
    print("\n计算原始数据的样本内相关性...")
    ori_correlations, ori_mean_corrs = calculate_within_subject_correlation(ori_data, subject_ids)
    
    # 计算处理后数据的样本内相关性
    print("计算处理后数据的样本内相关性...")
    vae_correlations, vae_mean_corrs = calculate_within_subject_correlation(vae_data, subject_ids)
    
    print(f"\n分析完成!")
    print(f"分析的subject数量: {len(ori_mean_corrs)}")
    
    # 统计分析
    improvement = analyze_correlation_comparison(ori_correlations, vae_correlations,
                                               ori_mean_corrs, vae_mean_corrs)
    
    # 绘制图表
    print("\n生成图表...")
    plot_results(ori_mean_corrs, vae_mean_corrs, improvement)
    
    # 保存详细结果
    save_detailed_results(ori_mean_corrs, vae_mean_corrs, subject_ids, improvement)
    
    # 结论
    print("\n" + "="*50)
    print("结论")
    print("="*50)
    mean_improvement = np.mean(vae_mean_corrs) - np.mean(ori_mean_corrs)
    if mean_improvement > 0:
        print(f"✅ 处理方法有效！平均相关性提高了 {mean_improvement:.4f}")
        print(f"   {np.sum(improvement > 0)/len(improvement)*100:.1f}% 的subject显示出改善")
    else:
        print(f"❌ 处理方法未显示出明显改善")
        print(f"   平均相关性变化: {mean_improvement:.4f}")

if __name__ == "__main__":
    main()