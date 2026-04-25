import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import pickle


def load_data():
    """加载数据"""
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, y, subject_id = data

    return node_feature, adj, subject_id

def calculate_within_subject_correlation(data, subject_ids):
    """
    计算每个subject内m个fc图之间的相关性
    
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

        print(f"subject {subject} 有 {len(subject_indices)} 个样本")
        if len(subject_indices) < 2:
            print(f"subject {subject} 跳过")
            continue
        m = len(subject_indices)
        
        # 提取该subject的m个fc图
        subject_fc = data[subject_indices]  # 形状: (m, 68, 68)
        
        # 将每个fc图展平为向量
        fc_vectors = subject_fc.reshape(m, -1)  # 形状: (m, 68*68)
        
        # 计算m个fc图之间的相关性矩阵
        corr_matrix = np.corrcoef(fc_vectors)  # 形状: (m, m)
        
        # 提取上三角部分（不包括对角线）
        upper_tri_indices = np.triu_indices(m, k=1)
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

def plot_results(ori_mean_corrs, vae_mean_corrs):
    """绘制小提琴图"""

    fs = 22
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': fs,
        'axes.titlesize': fs,
        'axes.labelsize': fs,
        'legend.fontsize': fs,
        'figure.titlesize': fs,
    })
    
    # 创建图形，只包含一个小提琴图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 准备数据格式用于小提琴图
    plot_data = []
    plot_labels = []
    
    for val in ori_mean_corrs:
        plot_data.append(val)
        plot_labels.append('Raw Data')
    
    for val in vae_mean_corrs:
        plot_data.append(val)
        plot_labels.append('Denoised Data')
    
    # 创建DataFrame用于seaborn
    df_violin = pd.DataFrame({
        'Correlation': plot_data,
        'Data Type': plot_labels
    })
    
    # 绘制小提琴图
    sns.violinplot(x='Data Type', y='Correlation', data=df_violin, 
                  ax=ax, palette=['#3498db', '#2ecc71'],  # 蓝色和绿色
                  inner='box',  # 内部显示箱线图
                  linewidth=2,
                  saturation=0.8)
    
    # 添加散点显示个体数据点
    sns.stripplot(x='Data Type', y='Correlation', data=df_violin,
                 ax=ax, color='black', size=6, alpha=0.6,
                 jitter=0.2)
    
    # 设置标题和标签
    # ax.set_title('Comparison of Intra-Subject Correlations\nBefore and After Processing', 
    #             fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=fs)
    ax.set_ylabel('Average Correlation', fontsize=fs)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('./output/correlation_violin_plot.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    # plt.show()
    
    print(f"\n小提琴图已保存到: ./output/correlation_violin_plot.png")

def save_detailed_results(ori_mean_corrs, vae_mean_corrs, subject_ids, improvement):
    """保存详细结果到CSV文件"""
    unique_subjects = np.unique(subject_ids)
    
    results_df = pd.DataFrame({
        'subject_id': unique_subjects[:len(ori_mean_corrs)],
        'original_mean_correlation': ori_mean_corrs,
        'processed_mean_correlation': vae_mean_corrs,
        'improvement': improvement
    })
    
    # 创建输出目录
    os.makedirs('./output', exist_ok=True)
    
    results_df.to_csv('./output/correlation_analysis_results.csv', index=False)
    print(f"\n详细结果已保存到: ./output/correlation_analysis_results.csv")

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
    
    # 绘制小提琴图（只绘制第一张图）
    print("\n生成小提琴图...")
    plot_results(ori_mean_corrs, vae_mean_corrs)
    
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
    # 创建输出目录
    os.makedirs('./output_cfcws', exist_ok=True)
    main()