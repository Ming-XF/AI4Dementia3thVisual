import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path
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

    node_feature, adj, label, subject_id = data
    return node_feature, adj, label, subject_id

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


def extract_connection_strength(
    node_feature: np.ndarray,
    adj: np.ndarray,
    subject_ids: np.ndarray,
    i: int,
    j: int,
    use_original: bool = True
) -> pd.DataFrame:
    """
    提取指定连接的强度值及对应的受试者ID
    
    Parameters:
    -----------
    node_feature : np.ndarray
        原始EEG构建的FC图，形状为(n_samples, n_nodes, n_nodes)或(n_samples, n_features)
    adj : np.ndarray
        VAE降噪后的FC图，形状为(n_samples, n_nodes, n_nodes)
    subject_ids : np.ndarray
        每个FC图对应的受试者ID，形状为(n_samples,)
    i, j : int
        要提取的连接索引
    use_original : bool
        是否使用原始FC图(node_feature)，False则使用VAE降噪后的图(adj)
    
    Returns:
    --------
    pd.DataFrame
        包含subject_id和连接强度的DataFrame
    """
    n_samples = len(subject_ids)
    
    # 选择使用的FC图数据
    if use_original:
        fc_data = node_feature
        data_type = 'original'
    else:
        fc_data = adj
        data_type = 'vae_denoised'
    
    # 提取指定连接的强度
    connection_strengths = []
    for idx in range(n_samples):
        # 根据数据维度提取连接强度
        if len(fc_data.shape) == 3:
            # 如果是3D数组 (n_samples, n_nodes, n_nodes)
            strength = fc_data[idx, i, j]
        elif len(fc_data.shape) == 2:
            # 如果是2D数组 (n_samples, n_features)
            # 假设是展平的FC矩阵，需要计算索引
            n_nodes = int(np.sqrt(fc_data.shape[1]))
            flat_idx = i * n_nodes + j
            strength = fc_data[idx, flat_idx]
        else:
            raise ValueError(f"Unexpected fc_data shape: {fc_data.shape}")
        
        connection_strengths.append(strength)
    
    # 创建DataFrame
    df_connection = pd.DataFrame({
        'subject_id': subject_ids,
        f'connection_strength_{data_type}': connection_strengths
    })

    # 新增：每个受试者只保留第一个数据
    df_connection = df_connection.groupby('subject_id').first().reset_index()
    
    return df_connection


def merge_clinical_and_fc_data(
    df_clinical: pd.DataFrame,
    df_connection: pd.DataFrame,
    clinical_score: str
) -> pd.DataFrame:
    """
    合并临床量表评分和FC连接强度数据
    
    Parameters:
    -----------
    df_clinical : pd.DataFrame
        临床量表评分数据
    df_connection : pd.DataFrame
        FC连接强度数据
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
    df_merged = df_connection.merge(
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
    connection_col: str,
    clinical_score: str,
    i: int,
    j: int,
    use_original: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_stats: bool = True,
    color_by_subject: bool = True
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """
    绘制散点图和拟合线
    
    Parameters:
    -----------
    df_merged : pd.DataFrame
        合并后的数据
    connection_col : str
        连接强度列名
    clinical_score : str
        临床量表评分列名
    i, j : int
        连接索引
    use_original : bool
        是否使用原始FC图
    figsize : Tuple[int, int]
        图形大小
    save_path : Optional[str]
        保存路径
    show_stats : bool
        是否显示统计信息
    color_by_subject : bool
        是否按受试者着色
    
    Returns:
    --------
    Tuple[plt.Figure, plt.Axes, dict]
        图形对象、轴对象和统计信息
    """
    
    # 数据
    x = df_merged[connection_col].values
    y = df_merged[clinical_score].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    if abs(r_value) < 0.3:
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
    
    # 线性拟合
    
    
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
    data_type = "Original FC" if use_original else "VAE Denoised FC"
    ax.set_xlabel(f'Connection Strength ({i}, {j})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{clinical_score} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{clinical_score} vs FC Connection ({i},{j})\n{data_type}', 
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
        'n_samples': n,
        'pearson_r': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }
    
    # 保存图形
    if save_path:
        plt.savefig(os.path.join(save_path, f'{i}_{j}_{clinical_score}_{stats_dict['pearson_r']:.2f}_{stats_dict['p_value']:.3f}.png'), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    
    return fig, ax, stats_dict


def analyze_fc_clinical_correlation(
    node_feature: np.ndarray,
    adj: np.ndarray,
    subject_ids: np.ndarray,
    clinical_scores_path: str,
    i: int,
    j: int,
    clinical_score: str,
    use_original: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    综合分析FC连接强度与临床量表评分的相关性
    
    Parameters:
    -----------
    node_feature : np.ndarray
        原始FC图数据
    adj : np.ndarray
        VAE降噪后的FC图数据
    subject_ids : np.ndarray
        受试者ID数组
    clinical_scores_path : str
        临床量表评分文件路径
    i, j : int
        连接索引
    clinical_score : str
        要分析的临床量表评分
    use_original : bool
        是否使用原始FC图
    save_path : Optional[str]
        图形保存路径
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict
        分析结果统计信息
    """
    if verbose:
        print("=" * 60)
        print("FC-Clinical Correlation Analysis")
        print("=" * 60)
        print(f"Analyzing connection: ({i}, {j})")
        print(f"Clinical score: {clinical_score}")
        print(f"Using {'Original' if use_original else 'VAE Denoised'} FC data")
        print("-" * 60)
    
    # 1. 加载临床数据
    df_clinical = load_clinical_scores(clinical_scores_path)
    if verbose:
        print(f"\nLoaded clinical data for {len(df_clinical)} subjects")
        print(f"Available clinical scores: {list(df_clinical.columns)}")
    
    # 2. 提取FC连接强度
    df_connection = extract_connection_strength(
        node_feature, adj, subject_ids, i, j, use_original
    )
    if verbose:
        print(f"\nExtracted connection strengths for {len(df_connection)} FC graphs")
        print(f"Unique subjects in FC data: {df_connection['subject_id'].nunique()}")
    
    # 3. 合并数据
    connection_col = df_connection.columns[1]  # 获取连接强度列名
    df_merged = merge_clinical_and_fc_data(
        df_clinical, df_connection, clinical_score
    )
    if verbose:
        print(f"\nMerged data: {len(df_merged)} samples")
        print(f"Subjects with both FC and clinical data: {df_merged['subject_id'].nunique()}")
        print(f"\nData summary:")
        print(df_merged.describe())
    
    # 4. 绘制散点图和拟合线
    fig, ax, stats_dict = plot_scatter_with_fit(
        df_merged, connection_col, clinical_score, i, j, 
        use_original, save_path=save_path
    )
    
    # 5. 打印统计结果
    if verbose and fig is not None:
        print("\n" + "=" * 60)
        print("Statistical Results")
        print("=" * 60)
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
    
    # plt.show()
    
    return stats_dict

if __name__ == "__main__":
    # 临床评分文件路径
    os.makedirs('./output_score', exist_ok=True)
    clinical_scores_path = "./MMS.txt"  # 请替换为您的实际文件路径
    node_feature, adj, _, subject_ids = load_and_preprocess_data()
    
    # 示例1：分析单个连接
    print("Example 1: Single connection analysis")
    print("-" * 40)

    items = ['MMSE','MoCA总分','即刻记忆','延迟回忆','线索回忆','长时延迟再认','数字广度顺向','数字广度逆向','连线测验A','连线测验B','Boston-初始命名','CDR_SOB','CDR','TMT B-A','CDT']

    for i in range(adj.shape[1]):
        for j in range(i-1):
            for clinical_score in items:
                result = analyze_fc_clinical_correlation(
                    node_feature=node_feature,
                    adj=adj,
                    subject_ids=subject_ids,
                    clinical_scores_path=clinical_scores_path,
                    i=i,
                    j=j,
                    clinical_score=clinical_score,
                    use_original=False,
                    save_path=f'./output_score'
                )


