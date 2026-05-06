import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, Tuple, List, Optional

import pickle
import os
from scipy import signal
from scipy.integrate import simpson
from sklearn.decomposition import PCA

import pdb

import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图风格
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")

def calculate_band_power_efficient(time_series, hz, freq_bands):
    """
    最高效版本：完全向量化，避免Python循环
    适合大规模数据处理
    
    Parameters:
    -----------
    time_series : ndarray
        形状为 (N, C, L)
    hz : float
        采样频率
    freq_bands : dict
        频段字典
    
    Returns:
    --------
    band_powers : dict
        每个频段的功率，形状为 (N, C)
    """
    N, C, L = time_series.shape
    band_powers = {}
    
    # 将数据重塑为 (N*C, L) 以便一次性处理所有通道
    data_reshaped = time_series.reshape(-1, L)
    
    # 一次性对所有(样本, 通道)组合计算功率谱
    freqs, psd_matrix = signal.welch(
        data_reshaped,
        fs=hz,
        nperseg=min(L, hz * 2),
        noverlap=min(L, hz * 2) // 2,
        axis=-1
    )
    
    # 为每个频段计算功率并重塑回 (N, C)
    for band_name, (f_low, f_high) in freq_bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        freq_range = freqs[freq_mask]
        psd_band = psd_matrix[:, freq_mask]
        
        # 向量化积分
        band_power = simpson(psd_band, freq_range, axis=-1)
        
        # 重塑回 (N, C)
        band_powers[band_name] = band_power.reshape(N, C)
    
    return band_powers

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """
    计算Cohen's d效应量
    
    Parameters:
    -----------
    group1, group2 : ndarray
        两组数据的数组
    
    Returns:
    --------
    d : ndarray
        Cohen's d效应量
    """
    # 计算均值差
    mean_diff = np.mean(group1, axis=0) - np.mean(group2, axis=0)
    
    # 计算合并标准差
    n1, n2 = group1.shape[0], group2.shape[0]
    var1 = np.var(group1, axis=0, ddof=1)
    var2 = np.var(group2, axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # 避免除零
    pooled_std[pooled_std == 0] = 1e-10
    
    d = mean_diff / pooled_std
    return d


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行FDR校正（Benjamini-Hochberg方法）
    
    Parameters:
    -----------
    p_values : ndarray
        原始p值数组
    alpha : float
        显著性水平
    
    Returns:
    --------
    rejected : ndarray
        是否拒绝原假设的布尔数组
    p_corrected : ndarray
        FDR校正后的p值
    """
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return rejected, p_corrected


def analyze_viewpoint(data: np.ndarray, labels: np.ndarray, viewpoint_name: str,
                     comparisons: List[Tuple[str, str]], label_map: Dict,
                     alpha: float = 0.05) -> Dict:
    """
    分析单个视角的所有组间比较
    
    Parameters:
    -----------
    data : ndarray
        重建误差数据，形状为(N, C)
    labels : ndarray
        样本标签数组
    viewpoint_name : str
        视角名称
    comparisons : list
        组间比较列表，如[('NC', 'AD'), ('NC', 'MCI'), ('NC', 'DSC')]
    label_map : dict
        标签映射字典
    alpha : float
        显著性水平
    
    Returns:
    --------
    results : dict
        包含所有分析结果的字典
    """
    results = {}
    
    for group1, group2 in comparisons:
        # 获取组索引
        idx1 = np.where(labels == label_map[group1])[0]
        idx2 = np.where(labels == label_map[group2])[0]
        
        # 提取数据
        data1 = data[idx1]
        data2 = data[idx2]
        
        # 计算统计量
        t_stats, p_values, effect_sizes = channel_wise_ttest_direct(data1, data2)
        
        # FDR校正
        rejected, p_corrected = fdr_correction(p_values, alpha)
        
        # 存储结果
        comparison_name = f"{group1}_vs_{group2}"
        results[comparison_name] = {
            't_stats': t_stats,
            'p_values': p_values,
            'p_corrected': p_corrected,
            'effect_sizes': effect_sizes,
            'significant': rejected,
            'n_significant': np.sum(rejected)
        }
        
    return results


def channel_wise_ttest_direct(data1: np.ndarray, data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    直接对两组数据进行通道级别的t检验
    
    Parameters:
    -----------
    data1, data2 : ndarray
        两组数据，形状分别为(N1, C)和(N2, C)
    
    Returns:
    --------
    t_stats, p_values, effect_sizes : ndarray
    """
    n_channels = data1.shape[1]
    
    t_stats = np.zeros(n_channels)
    p_values = np.zeros(n_channels)
    effect_sizes = np.zeros(n_channels)
    
    for ch in range(n_channels):
        t_stat, p_val = stats.ttest_ind(data1[:, ch], data2[:, ch])
        t_stats[ch] = t_stat
        p_values[ch] = p_val
        effect_sizes[ch] = cohens_d(
            data1[:, ch].reshape(-1, 1),
            data2[:, ch].reshape(-1, 1)
        )[0]
    
    return t_stats, p_values, effect_sizes


def plot_viewpoint_results(results: Dict, viewpoint_name: str, 
                          channel_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (18, 12)):
    """
    为单个视角绘制三张统计图（T值、P值、效应量），每张包含三个子图
    
    Parameters:
    -----------
    results : dict
        analyze_viewpoint的输出结果
    viewpoint_name : str
        视角名称
    channel_names : list, optional
        通道名称列表
    figsize : tuple
        图形大小
    """
    comparisons = list(results.keys())
    n_comparisons = len(comparisons)
    n_channels = len(results[comparisons[0]]['t_stats'])
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # 创建颜色映射
    colors = plt.cm.Set2(np.linspace(0, 1, n_channels))
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(3, n_comparisons, figsize=figsize)
    fig.suptitle(f'{viewpoint_name} - 组间差异分析', fontsize=16, fontweight='bold')
    
    metrics = ['t_stats', 'p_values', 'effect_sizes']
    metric_labels = ['T统计量', 'P值 (FDR校正)', "Cohen's d 效应量"]
    color_maps = ['RdBu_r', 'YlOrRd', 'RdBu_r']
    
    for row_idx, (metric, metric_label, cmap) in enumerate(zip(metrics, metric_labels, color_maps)):
        # 获取全局范围用于统一colorbar
        all_values = []
        for comp in comparisons:
            if metric == 'p_values':
                # 对于p值，使用校正后的p值并取-log10
                values = -np.log10(results[comp]['p_corrected'] + 1e-10)
            else:
                values = results[comp][metric]
            all_values.extend(values)
        
        vmin, vmax = min(all_values), max(all_values)
        
        for col_idx, comp in enumerate(comparisons):
            ax = axes[row_idx, col_idx] if n_comparisons > 1 else axes[row_idx]
            
            if metric == 'p_values':
                # 使用校正后的p值，转换为-log10尺度
                values = -np.log10(results[comp]['p_corrected'] + 1e-10)
                # 标记显著性
                significant = results[comp]['significant']
                colors_bar = ['red' if sig else 'steelblue' for sig in significant]
            else:
                values = results[comp][metric]
                colors_bar = colors[:len(values)]
            
            # 创建柱状图
            x_pos = np.arange(len(channel_names))
            bars = ax.bar(x_pos, values, color=colors_bar, edgecolor='black', linewidth=0.5)
            
            # 标注显著通道
            if metric == 'p_values':
                sig_idx = np.where(results[comp]['significant'])[0]
                for idx in sig_idx:
                    ax.text(idx, values[idx] + 0.1, '*', ha='center', fontsize=12, fontweight='bold')
            
            # 设置标题和标签
            comp_name = comp.replace('_', ' ')
            ax.set_title(f'{comp_name}\n(显著通道: {results[comp]["n_significant"]})', fontsize=11)
            ax.set_xlabel('通道', fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xticks(x_pos[::5])  # 每隔5个显示一个标签
            ax.set_xticklabels(channel_names[::5], rotation=45, ha='right', fontsize=8)
            
            # 添加水平参考线
            if metric == 'p_values':
                ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
                ax.legend(fontsize=8)
            elif metric == 'effect_sizes':
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                # 添加效应量解释线
                ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='小效应')
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='中效应')
                ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='大效应')
                ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
                ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.7)
                ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.9)
                ax.legend(fontsize=7, loc='upper right')
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def comprehensive_analysis(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray,
                          labels: np.ndarray, label_map: Dict = None,
                          channel_names: List[str] = None, output_dir: str = "./") -> Dict:
    """
    综合分析三个视角的组间差异
    
    Parameters:
    -----------
    r1, r2, r3 : ndarray
        时域、频域、相位域的重建误差，形状为(N, C)
    labels : ndarray
        样本标签数组
    label_map : dict
        标签映射，默认{0: 'AD', 1: 'DSC', 2: 'MCI', 3: 'NC'}
    channel_names : list
        通道名称列表
    
    Returns:
    --------
    all_results : dict
        包含所有视角分析结果的字典
    """
    if label_map is None:
        label_map = {'AD': 0, 'DSC': 1, 'MCI': 2, 'NC': 3}
    
    comparisons = [('NC', 'AD'), ('NC', 'MCI'), ('NC', 'DSC')]
    
    viewpoints = {
        'time': r1,
        'frequency': r2,
        'phase': r3
    }
    
    all_results = {}
    
    for vp_name, vp_data in viewpoints.items():
        print(f"\n分析视角: {vp_name}")
        print("=" * 50)
        
        results = analyze_viewpoint(vp_data, labels, vp_name, comparisons, label_map)
        all_results[vp_name] = results
        
        # 打印统计摘要
        for comp_name, comp_results in results.items():
            print(f"\n{comp_name}:")
            print(f"  显著通道数 (FDR校正): {comp_results['n_significant']}")
            print(f"  最大效应量: {np.max(np.abs(comp_results['effect_sizes'])):.3f}")
            print(f"  平均效应量: {np.mean(np.abs(comp_results['effect_sizes'])):.3f}")
        
        # 绘制该视角的结果
        fig = plot_viewpoint_results(results, vp_name, channel_names, figsize=(16, 10))
        plt.savefig(f"{output_dir}/{vp_name}.png", dpi=300, bbox_inches='tight')
        # plt.show()
    
    return all_results


def export_results_to_csv(all_results: Dict, channel_names: List[str], 
                         output_dir: str = './'):
    """
    将分析结果导出为CSV文件
    
    Parameters:
    -----------
    all_results : dict
        综合分析的结果字典
    channel_names : list
        通道名称列表
    output_dir : str
        输出目录
    """
    for vp_name, vp_results in all_results.items():
        for comp_name, comp_results in vp_results.items():
            df = pd.DataFrame({
                'Channel': channel_names,
                'T_statistic': comp_results['t_stats'],
                'P_value': comp_results['p_values'],
                'P_corrected_FDR': comp_results['p_corrected'],
                'Effect_size_Cohens_d': comp_results['effect_sizes'],
                'Significant_FDR': comp_results['significant']
            })
            
            filename = f"{output_dir}/{vp_name}_{comp_name}_results.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"结果已保存至: {filename}")


# ==================== 使用示例 ====================

def load_and_preprocess_data():
    """
    加载正负样本数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    _, _, labels, _, _, r1, r2, r3 = data
    return labels, r1, r2, r3

if __name__ == "__main__":

    channel_names = []
    labels, r1, r2, r3 = load_and_preprocess_data()
    N, C, L, D = r1.shape
    for i in range(r1.shape[1]):
        channel_names.append('region_'+str(i))
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    # pdb.set_trace()
    # r1 = calculate_band_power_efficient(r1, 250, freq_bands)['gamma']
    # r2 = calculate_band_power_efficient(r2, 250, freq_bands)['gamma']
    # r3 = calculate_band_power_efficient(r3, 250, freq_bands)['gamma']

    pca = PCA(n_components=1)
    r1 = pca.fit_transform(r1.reshape(-1, D)).reshape(N, C, L)
    r2 = pca.fit_transform(r2.reshape(-1, D)).reshape(N, C, L)
    r3 = pca.fit_transform(r3.reshape(-1, D)).reshape(N, C, L)

    # pdb.set_trace()
    
    
    print("数据形状:")
    print(f"labels: {labels.shape}")
    print(f"r1 (时域): {r1.shape}")
    print(f"r2 (频域): {r2.shape}")
    print(f"r3 (相位域): {r3.shape}")
        
    for key in freq_bands.keys():
        k1 = calculate_band_power_efficient(r1, 250, freq_bands)[key]
        k2 = calculate_band_power_efficient(r2, 250, freq_bands)[key]
        k3 = calculate_band_power_efficient(r3, 250, freq_bands)[key]

        output_dir = f'./output_msettest/{key}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行综合分析
        all_results = comprehensive_analysis(
            k1, k2, k3, 
            labels, 
            channel_names=channel_names,
            output_dir=output_dir
        )
        
        # 导出结果
        export_results_to_csv(all_results, channel_names, output_dir=output_dir)