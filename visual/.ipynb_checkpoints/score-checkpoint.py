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

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    加载数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, label, subject_id, cnns = data
    return node_feature, adj, label, subject_id, cnns

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

def extract_cnn_feature(
    cnn_features: np.ndarray,
    subject_ids: np.ndarray,
    feature_dim: int
) -> pd.DataFrame:
    """
    提取CNN指定维度的特征
    
    Parameters:
    -----------
    cnn_features : np.ndarray
        CNN特征数组，形状为(N, 32)，N为样本个数，32为特征维度
    subject_ids : np.ndarray
        每个CNN特征对应的受试者ID，形状为(N,)
    feature_dim : int
        要提取的特征维度索引（0-31）
    
    Returns:
    --------
    pd.DataFrame
        包含subject_id和CNN特征的DataFrame
    """
    n_samples = cnn_features.shape[0]
    
    if feature_dim >= cnn_features.shape[1]:
        raise ValueError(f"feature_dim {feature_dim} exceeds CNN feature dimensions {cnn_features.shape[1]}")
    
    # 提取指定维度的特征值
    feature_values = cnn_features[:, feature_dim]
    feature_name = f'cnn_dim_{feature_dim}'
    
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
    cnn_feature_col: str,
    clinical_score: str,
    feature_dim: int,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
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
    ax.set_title(f'{clinical_score} vs CNN Feature {cnn_feature_col}', 
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
        'cnn_dim': feature_dim,
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
    if save_path:
        filename = f'cnn_dim{feature_dim}_{clinical_score}_r{stats_dict["pearson_r"]:.2f}_p{stats_dict["p_value"]:.3f}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {os.path.join(save_path, filename)}")
    
    return fig, ax, stats_dict


def analyze_cnn_clinical_correlation(
    cnn_features: np.ndarray,
    subject_ids: np.ndarray,
    clinical_scores_path: str,
    feature_dim: int,
    clinical_score: str,
    save_path: Optional[str] = None,
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
        print("CNN-Clinical Correlation Analysis")
        print("=" * 60)
        print(f"Analyzing CNN dimension: {feature_dim}")
        print(f"Clinical score: {clinical_score}")
        print("-" * 60)
    
    # 1. 加载临床数据
    df_clinical = load_clinical_scores(clinical_scores_path)
    if verbose:
        print(f"\nLoaded clinical data for {len(df_clinical)} subjects")
        print(f"Available clinical scores: {list(df_clinical.columns)}")
    
    # 2. 提取CNN特征维度
    df_cnn = extract_cnn_feature(cnn_features, subject_ids, feature_dim)
    if verbose:
        print(f"\nExtracted CNN dimension {feature_dim} for {len(df_cnn)} samples")
        print(f"Unique subjects in CNN data: {df_cnn['subject_id'].nunique()}")
    
    # 3. 合并数据
    cnn_feature_col = df_cnn.columns[1]  # 获取CNN特征列名
    df_merged = merge_clinical_and_cnn_data(
        df_clinical, df_cnn, clinical_score
    )
    if verbose:
        print(f"\nMerged data: {len(df_merged)} samples")
        print(f"Subjects with both CNN and clinical data: {df_merged['subject_id'].nunique()}")
        if len(df_merged) > 0:
            print(f"\nData summary:")
            print(df_merged[[cnn_feature_col, clinical_score]].describe())
    
    if len(df_merged) == 0:
        print("Warning: No overlapping subjects between CNN features and clinical scores!")
        return None
    
    # 4. 绘制散点图和拟合线
    fig, ax, stats_dict = plot_scatter_with_fit(
        df_merged, cnn_feature_col, clinical_score, feature_dim,
        save_path=save_path, thred=thred
    )
    
    # 5. 打印统计结果
    if verbose and stats_dict is not None:
        print("\n" + "=" * 60)
        print("Statistical Results")
        print("=" * 60)
        print(f"CNN Dimension: {stats_dict['cnn_dim']}")
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


def batch_analysis_cnn_clinical(
    cnn_features: np.ndarray,
    subject_ids: np.ndarray,
    clinical_scores_path: str,
    clinical_scores_list: List[str],
    save_path: str,
    thred: float = 0.3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    批量分析所有CNN特征维度与所有临床量表评分的相关性
    
    Parameters:
    -----------
    cnn_features : np.ndarray
        CNN特征数组，形状为(N, 32)
    subject_ids : np.ndarray
        受试者ID数组
    clinical_scores_path : str
        临床量表评分文件路径
    clinical_scores_list : List[str]
        要分析的临床量表评分列表
    save_path : str
        结果保存路径
    thred : float
        相关系数阈值
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    pd.DataFrame
        包含所有分析结果的DataFrame
    """
    os.makedirs(save_path, exist_ok=True)
    
    n_dims = cnn_features.shape[1]
    all_results = []
    
    print("=" * 80)
    print("BATCH ANALYSIS: CNN Features vs Clinical Scores")
    print("=" * 80)
    print(f"CNN feature dimensions: {n_dims}")
    print(f"Clinical scores to analyze: {len(clinical_scores_list)}")
    print(f"Total combinations: {n_dims * len(clinical_scores_list)}")
    print("-" * 80)
    
    for dim in range(n_dims):
        for clinical_score in clinical_scores_list:
            result = analyze_cnn_clinical_correlation(
                cnn_features=cnn_features,
                subject_ids=subject_ids,
                clinical_scores_path=clinical_scores_path,
                feature_dim=dim,
                clinical_score=clinical_score,
                save_path=save_path,
                verbose=False,  # 批量分析时不打印详细信息
                thred=thred
            )
            
            if result is not None:
                all_results.append(result)
    
    # 创建结果DataFrame
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # 按相关系数绝对值排序
        df_results['abs_r'] = np.abs(df_results['pearson_r'])
        df_results = df_results.sort_values('abs_r', ascending=False)
        
        # 保存结果到CSV
        results_file = os.path.join(save_path, 'cnn_clinical_correlation_results.csv')
        df_results.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # 打印重要发现
        print("\n" + "=" * 80)
        print("TOP SIGNIFICANT CORRELATIONS (p < 0.05)")
        print("=" * 80)
        sig_results = df_results[df_results['p_value'] < 0.05]
        if len(sig_results) > 0:
            for idx, row in sig_results.iterrows():
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
                print(f"CNN Dim {int(row['cnn_dim'])} vs {row['clinical_score']}: "
                      f"r={row['pearson_r']:.4f}, p={row['p_value']:.4f} {significance}")
        else:
            print("No significant correlations found at p < 0.05")
        
        return df_results
    else:
        print("\nNo results found that meet the threshold criteria.")
        return pd.DataFrame()


if __name__ == "__main__":
    # 临床评分文件路径
    clinical_scores_path = "./MMS.txt"  # 请替换为您的实际文件路径
    
    # 加载数据
    node_feature, adj, label, subject_ids, cnns = load_and_preprocess_data()
    
    # CNN特征数据 - cnns 的形状应该是 (N, 32)
    print(f"CNN features shape: {cnns.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    
    # 临床量表评分列表
    items = [
        'MMSE', 'MoCA总分', '即刻记忆', '延迟回忆', '线索回忆', 
        '长时延迟再认', '数字广度顺向', '数字广度逆向', 
        '连线测验A', '连线测验B', 'Boston-初始命名', 
        'CDR_SOB', 'CDR', 'TMT B-A', 'CDT'
    ]
    
    # 创建输出目录
    output_dir = './output_score_cnn'
    
    # 批量分析所有CNN维度与所有量表评分的相关性
    df_results = batch_analysis_cnn_clinical(
        cnn_features=cnns,
        subject_ids=subject_ids,
        clinical_scores_path=clinical_scores_path,
        clinical_scores_list=items,
        save_path=output_dir,
        thred=0.3,  # 只保存|r| > 0.3的结果图
        verbose=True
    )
    
    # 可选：分析单个CNN维度与特定量表评分
    # if len(df_results) > 0:
    #     print("\n" + "=" * 80)
    #     print("SINGLE ANALYSIS EXAMPLE")
    #     print("=" * 80)
        
    #     # 选择相关性最高的一对进行分析
    #     top_result = df_results.iloc[0]
    #     best_dim = int(top_result['cnn_dim'])
    #     best_score = top_result['clinical_score']
        
    #     print(f"Analyzing best correlation: CNN Dim {best_dim} vs {best_score}")
        
    #     result = analyze_cnn_clinical_correlation(
    #         cnn_features=cnns,
    #         subject_ids=subject_ids,
    #         clinical_scores_path=clinical_scores_path,
    #         feature_dim=best_dim,
    #         clinical_score=best_score,
    #         save_path=output_dir,
    #         verbose=True,
    #         thred=0  # 强制保存
    #     )