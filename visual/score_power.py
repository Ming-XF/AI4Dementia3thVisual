import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.integrate import simpson
from sklearn.decomposition import PCA
import pickle
import os
import warnings

import pdb

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    加载数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, label, subject_id, cnns, r1, r2, r3 = data
    return subject_id, r1, r2, r3

def load_clinical_scores(filepath: str) -> pd.DataFrame:
    """
    加载临床量表评分数据
    
    Parameters:
    -----------
    filepath : str
        量表评分txt文件路径
    
    Returns:
    --------
    pd.DataFrame
        包含受试者量表评分的DataFrame，索引为subject_id
    """
    # 读取数据，使用制表符分隔
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    
    # 重命名第一列为subject_id，并创建标准化的subject_id
    df = df.rename(columns={df.columns[0]: 'subject_id'})
    
    # 提取subject_id中的数字部分，转换为标准格式
    df['subject_id'] = df['subject_id'].apply(
        lambda x: int(''.join(filter(str.isdigit, str(x))))
    )
    
    # 设置subject_id为索引
    df.set_index('subject_id', inplace=True)
    
    return df

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

def extract_power_feature(
    power,
    subject_ids: np.ndarray,
    channel,
) -> pd.DataFrame:
    """
    提取time_series的指定通道，指定频段的功率
    """

    feature_values = power[:, channel]
    feature_name = f'power_channel_{channel}'
    
    # 创建DataFrame
    df_feature = pd.DataFrame({
        'subject_id': subject_ids,
        feature_name: feature_values
    })

    # 每个受试者只保留第一个数据（如果有重复）
    df_feature = df_feature.groupby('subject_id').first().reset_index()
    
    return df_feature


def merge_clinical_and_cnn_data(
    df_clinical: pd.DataFrame,
    df_cnn: pd.DataFrame,
    clinical_score: str
) -> pd.DataFrame:
    """
    合并临床量表评分和CNN特征数据
    
    Parameters:
    -----------
    df_clinical : pd.DataFrame
        临床量表评分数据
    df_cnn : pd.DataFrame
        CNN特征数据
    clinical_score : str
        要分析的临床量表评分名称
    
    Returns:
    --------
    pd.DataFrame
        合并后的数据
    """
    # 确保clinical_score存在于df_clinical中
    if clinical_score not in df_clinical.columns:
        raise ValueError(f"Clinical score '{clinical_score}' not found in data. "
                         f"Available scores: {list(df_clinical.columns)}")
    
    # 提取需要的临床评分
    df_scores = df_clinical[[clinical_score]].copy()
    
    # 合并数据
    df_merged = df_cnn.merge(
        df_scores,
        left_on='subject_id',
        right_index=True,
        how='left'
    )
    
    # 移除缺失值
    df_merged = df_merged.dropna(subset=[clinical_score])
    
    return df_merged


def plot_scatter_with_fit(
    df_merged: pd.DataFrame,
    channel,
    cnn_feature_col: str,
    clinical_score: str,
    save_path,
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True,
    color_by_subject: bool = True,
    thred: float = 0.3
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """
    绘制散点图和拟合线
    
    Parameters:
    -----------
    df_merged : pd.DataFrame
        合并后的数据
    cnn_feature_col : str
        CNN特征列名
    clinical_score : str
        临床量表评分列名
    feature_dim : int
        CNN特征维度索引
    figsize : Tuple[int, int]
        图形大小
    save_path : Optional[str]
        保存路径
    show_stats : bool
        是否显示统计信息
    color_by_subject : bool
        是否按受试者着色
    thred : float
        相关系数阈值，低于此值不绘图
    
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes, dict]
        图形对象、轴对象和统计信息
    """
    
    # 数据
    x = df_merged[cnn_feature_col].values
    y = df_merged[clinical_score].values

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 如果相关系数低于阈值，不绘图
    if abs(r_value) < thred:
        return None, None, None

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 按受试者着色
    if color_by_subject:
        unique_subjects = df_merged['subject_id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        for idx, subject in enumerate(unique_subjects):
            mask = df_merged['subject_id'] == subject
            ax.scatter(x[mask], y[mask], 
                      color=colors[idx], 
                      label=f'Sub-{subject}',
                      alpha=0.7, 
                      s=80,
                      edgecolors='black',
                      linewidth=0.5)
        
        # 如果受试者太多，不显示图例
        if len(unique_subjects) > 10:
            ax.legend().set_visible(False)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(x, y, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # 绘制拟合线
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
            label=f'Linear fit (R²={r_value**2:.3f}, p={p_value:.4f})')
    
    # 添加置信区间
    n = len(x)
    x_mean = np.mean(x)
    confidence_interval = 1.96 * np.sqrt(
        np.sum((y - (slope * x + intercept))**2) / (n - 2)
    ) * np.sqrt(1/n + (x_fit - x_mean)**2 / np.sum((x - x_mean)**2))
    
    ax.fill_between(x_fit, 
                    y_fit - confidence_interval, 
                    y_fit + confidence_interval, 
                    alpha=0.2, color='red',
                    label='95% CI')
    
    # 设置标签和标题
    ax.set_xlabel(cnn_feature_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{clinical_score} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{clinical_score} vs Power Feature {cnn_feature_col}', 
                 fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息文本框
    if show_stats:
        stats_text = f'N = {n}\n'
        stats_text += f'R = {r_value:.3f}\n'
        stats_text += f'R² = {r_value**2:.3f}\n'
        stats_text += f'p = {p_value:.4f}\n'
        stats_text += f'Slope = {slope:.4f}\n'
        stats_text += f'Intercept = {intercept:.4f}'
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 美化图形
    sns.despine()
    plt.tight_layout()

    # 统计信息
    stats_dict = {
        'clinical_score': clinical_score,
        'n_samples': n,
        'pearson_r': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }
    
    # 保存图形
    filename = f'power_channel_{channel}_{clinical_score}_r{stats_dict["pearson_r"]:.2f}_p{stats_dict["p_value"]:.3f}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(save_path, filename)}")
    
    return fig, ax, stats_dict


def analyze_cnn_clinical_correlation(
    power,
    channel: int,
    subject_ids: np.ndarray,
    clinical_scores_path: str,
    clinical_score: str,
    save_path,
    verbose: bool = True,
    thred: float = 0.3
) -> dict:
    """
    综合分析CNN特征维度与临床量表评分的相关性
    
    Parameters:
    -----------
    cnn_features : np.ndarray
        CNN特征数组，形状为(N, 32)
    subject_ids : np.ndarray
        受试者ID数组
    clinical_scores_path : str
        临床量表评分文件路径
    feature_dim : int
        CNN特征维度索引（0-31）
    clinical_score : str
        要分析的临床量表评分
    save_path : Optional[str]
        图形保存路径
    verbose : bool
        是否打印详细信息
    thred : float
        相关系数阈值（绝对值），低于此值不保存图片
    
    Returns:
    --------
    dict
        分析结果统计信息
    """
    if verbose:
        print("=" * 60)
        print("Power-Clinical Correlation Analysis")
        print("=" * 60)
        print(f"Clinical score: {clinical_score}")
        print("-" * 60)
    
    # 1. 加载临床数据
    df_clinical = load_clinical_scores(clinical_scores_path)
    if verbose:
        print(f"\nLoaded clinical data for {len(df_clinical)} subjects")
        print(f"Available clinical scores: {list(df_clinical.columns)}")
    
    # 2. 提取Power特征维度
    df_cnn = extract_power_feature(power, subject_ids, channel)
    if verbose:
        print(f"Unique subjects in CNN data: {df_cnn['subject_id'].nunique()}")
    
    # 3. 合并数据
    cnn_feature_col = df_cnn.columns[1]  # 获取CNN特征列名
    df_merged = merge_clinical_and_cnn_data(
        df_clinical, df_cnn, clinical_score
    )
    if verbose:
        print(f"\nMerged data: {len(df_merged)} samples")
        print(f"Subjects with both Power and clinical data: {df_merged['subject_id'].nunique()}")
        if len(df_merged) > 0:
            print(f"\nData summary:")
            print(df_merged[[cnn_feature_col, clinical_score]].describe())
    
    if len(df_merged) == 0:
        print("Warning: No overlapping subjects between CNN features and clinical scores!")
        return None
    
    # 4. 绘制散点图和拟合线
    fig, ax, stats_dict = plot_scatter_with_fit(
        df_merged, channel, cnn_feature_col, clinical_score, save_path, thred=thred
    )
    
    # 5. 打印统计结果
    if verbose and stats_dict is not None:
        print("\n" + "=" * 60)
        print("Statistical Results")
        print("=" * 60)
        print(f"Clinical Score: {stats_dict['clinical_score']}")
        print(f"Pearson r: {stats_dict['pearson_r']:.4f}")
        print(f"R-squared: {stats_dict['r_squared']:.4f}")
        print(f"P-value: {stats_dict['p_value']:.4f}")
        print(f"Slope: {stats_dict['slope']:.4f}")
        print(f"Intercept: {stats_dict['intercept']:.4f}")
        print(f"Sample size: {stats_dict['n_samples']}")
        
        # 显著性判断
        if stats_dict['p_value'] < 0.001:
            print("\n*** Correlation is statistically significant (p < 0.001)")
        elif stats_dict['p_value'] < 0.01:
            print("\n** Correlation is statistically significant (p < 0.01)")
        elif stats_dict['p_value'] < 0.05:
            print("\n* Correlation is statistically significant (p < 0.05)")
        else:
            print("\nCorrelation is not statistically significant (p >= 0.05)")
    
    return stats_dict


if __name__ == "__main__":
    # 临床评分文件路径
    clinical_scores_path = "./MMS.txt"  # 请替换为您的实际文件路径
    
    # 加载数据
    subject_ids, r1, r2, r3 = load_and_preprocess_data()
    N, C, L, D = r1.shape

    pca = PCA(n_components=1)
    r1 = pca.fit_transform(r1.reshape(-1, D)).reshape(N, C, L)
    r2 = pca.fit_transform(r2.reshape(-1, D)).reshape(N, C, L)
    r3 = pca.fit_transform(r3.reshape(-1, D)).reshape(N, C, L)

    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    r1Power = calculate_band_power_efficient(r1, 250, freq_bands)
    r2Power = calculate_band_power_efficient(r2, 250, freq_bands)
    r3Power = calculate_band_power_efficient(r3, 250, freq_bands)

    powers = [r1Power, r2Power, r3Power]
    
    # 临床量表评分列表
    items = [
        'MMSE', 'MoCA总分', '即刻记忆', '延迟回忆', '线索回忆', 
        '长时延迟再认', '数字广度顺向', '数字广度逆向', 
        '连线测验A', '连线测验B', 'Boston-初始命名', 
        'CDR_SOB', 'CDR', 'TMT B-A', 'CDT'
    ]
    
    # 创建输出目录
    output_dir = './output_score_power'
    os.makedirs(output_dir, exist_ok=True)

    names = ['time', 'frequency', 'phase']
    for name, power in zip(names, powers):
        output_dir1 = output_dir+"/"+name
        os.makedirs(output_dir1, exist_ok=True)
        
        for band_name, _ in freq_bands.items():
            path = "./"+"output_msettest/"+band_name+f"/{name}_NC_vs_AD_results.csv"
    
            output_dir2 = output_dir1+"/"+band_name
            os.makedirs(output_dir2, exist_ok=True)
    
            df = pd.read_csv(path)
            channels = df[df['Significant_FDR'] == True].index
    
            for channel in channels:
                for item in items:
                    result = analyze_cnn_clinical_correlation(
                        power=power[band_name],
                        channel=channel,
                        subject_ids=subject_ids,
                        clinical_scores_path=clinical_scores_path,
                        clinical_score=item,
                        save_path=output_dir2,
                        verbose=True,
                        thred=0.3  # 强制保存
                    )
        
    