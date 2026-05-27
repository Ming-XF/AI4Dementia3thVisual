import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path
import pickle
import os
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    加载数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, label, subject_id, cnns, r1, r2, r3, _ = data
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
    thred: float = 0.3,
    p_value_fdr: Optional[float] = None
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
    p_value_fdr : Optional[float]
        FDR校正后的p值
    
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
        
    else:
        ax.scatter(x, y, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # 绘制拟合线
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # 使用FDR校正后的p值显示
    p_display = p_value_fdr if p_value_fdr is not None else p_value
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
            label=f'Linear fit (R²={r_value**2:.3f}, p_fdr={p_display:.4f})')
    
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
    
    # 量表名称翻译
    SCALE_NAME_TRANSLATION = {
        'MoCA总分': 'MoCA Total',
        '即刻记忆': 'Immediate Memory',
        '延迟回忆': 'Delayed Recall',
        '线索回忆': 'Cued Recall',
        '长时延迟再认': 'Long-delayed Recognition',
        '数字广度顺向': 'Digit Span Forward',
        '数字广度逆向': 'Digit Span Backward',
        '连线测验A': 'TMT-A',
        '连线测验B': 'TMT-B',
        'Boston-初始命名': 'Boston Naming',
    }
    clinical_display = SCALE_NAME_TRANSLATION.get(clinical_score, clinical_score)

    # 设置标签
    ax.set_xlabel(cnn_feature_col.replace("_", " "), fontsize=28)
    ax.set_ylabel(f'{clinical_display} Score', fontsize=28)
    ax.tick_params(axis='both', labelsize=28)
    
    # 添加统计信息文本框（显示校正后的p值）
    if show_stats:
        stats_text = f'N = {n}\n'
        stats_text += f'R = {r_value:.3f}\n'
        # stats_text += f'R² = {r_value**2:.3f}\n'
        # stats_text += f'p_uncorrected = {p_value:.4f}\n'
        if p_value_fdr is not None:
            stats_text += f'p_fdr = {p_value_fdr:.4f}'
        # stats_text += f'Slope = {slope:.4f}\n'
        # stats_text += f'Intercept = {intercept:.4f}'

        # 根据相关性正负设置文本框位置
        if r_value > 0:
            text_x, text_ha = 0.05, 'left'
        else:
            text_x, text_ha = 0.95, 'right'

        ax.text(text_x, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=28,
                verticalalignment='top',
                horizontalalignment=text_ha,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 根据相关性正负设置图例位置
    if r_value > 0:
        legend_loc = 'upper left'
    else:
        legend_loc = 'upper right'

    if color_by_subject and len(unique_subjects) <= 10:
        ax.legend(loc=legend_loc, fontsize=28)

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
        'p_value_fdr': p_value_fdr,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }
    
    # 保存图形
    if save_path:
        p_str = f"p{p_value:.3f}"
        if p_value_fdr is not None:
            p_str += f"_pfdr{p_value_fdr:.3f}"
        filename = f'cnn_dim{feature_dim}_{clinical_display}_r{r_value:.2f}_{p_str}.png'
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
    thred: float = 0.3,
    p_value_fdr: Optional[float] = None
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
    p_value_fdr : Optional[float]
        FDR校正后的p值
    
    Returns:
    --------
    dict
        分析结果统计信息，始终返回（除非没有重叠样本）
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
    cnn_feature_col = df_cnn.columns[1]
    df_merged = merge_clinical_and_cnn_data(
        df_clinical, df_cnn, clinical_score
    )
    if verbose:
        print(f"\nMerged data: {len(df_merged)} samples")
        print(f"Subjects with both CNN and clinical data: {df_merged['subject_id'].nunique()}")
    
    if len(df_merged) == 0:
        print("Warning: No overlapping subjects between CNN features and clinical scores!")
        return None
    
    # 4. 计算统计量（始终计算）
    x = df_merged[cnn_feature_col].values
    y = df_merged[clinical_score].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 构建统计结果字典（始终返回）
    stats_dict = {
        'cnn_dim': feature_dim,
        'clinical_score': clinical_score,
        'n_samples': len(df_merged),
        'pearson_r': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'p_value_fdr': p_value_fdr,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }
    
    # 5. 绘制散点图和拟合线（仅在超过阈值时绘图和保存）
    if abs(r_value) >= thred:
        fig, ax, _ = plot_scatter_with_fit(
            df_merged, cnn_feature_col, clinical_score, feature_dim,
            save_path=save_path, thred=thred, p_value_fdr=p_value_fdr
        )
    
    # 6. 打印统计结果
    if verbose:
        print("\n" + "=" * 60)
        print("Statistical Results")
        print("=" * 60)
        print(f"CNN Dimension: {stats_dict['cnn_dim']}")
        print(f"Clinical Score: {stats_dict['clinical_score']}")
        print(f"Pearson r: {stats_dict['pearson_r']:.4f}")
        print(f"R-squared: {stats_dict['r_squared']:.4f}")
        print(f"P-value (uncorrected): {stats_dict['p_value']:.4f}")
        if p_value_fdr is not None:
            print(f"P-value (FDR corrected): {p_value_fdr:.4f}")
        print(f"Slope: {stats_dict['slope']:.4f}")
        print(f"Intercept: {stats_dict['intercept']:.4f}")
        print(f"Sample size: {stats_dict['n_samples']}")
        
        # 显著性判断（使用FDR校正后的p值）
        p_for_significance = p_value_fdr if p_value_fdr is not None else p_value
        if p_for_significance < 0.001:
            print("\n*** Correlation is statistically significant (p_fdr < 0.001)")
        elif p_for_significance < 0.01:
            print("\n** Correlation is statistically significant (p_fdr < 0.01)")
        elif p_for_significance < 0.05:
            print("\n* Correlation is statistically significant (p_fdr < 0.05)")
        else:
            print("\nCorrelation is not statistically significant (p_fdr >= 0.05)")
    
    return stats_dict


def plot_correlation_heatmap(
    df_results: pd.DataFrame,
    save_path: str,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = 'RdBu_r',
    annotate: bool = True,
    thred: float = 0.3
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot correlation heatmap of CNN feature dimensions vs clinical scores.

    Parameters:
    -----------
    df_results : pd.DataFrame
        Must contain columns: cnn_dim, clinical_score, pearson_r, significant_fdr.
    save_path : str
        Output directory for the figure.
    figsize : Tuple[int, int]
        Figure size.
    cmap : str
        Colormap.
    annotate : bool
        Whether to show numeric values in cells.
    thred : float
        Correlation threshold for black border marking.

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
    """
    pivot_corr = df_results.pivot_table(
        values='pearson_r',
        index='clinical_score',
        columns='cnn_dim',
        aggfunc='first'
    )

    pivot_sig = df_results.pivot_table(
        values='significant_fdr',
        index='clinical_score',
        columns='cnn_dim',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=figsize)

    if annotate:
        annot_matrix = pivot_corr.round(2).values
    else:
        annot_matrix = None

    sns.heatmap(
        pivot_corr,
        annot=annot_matrix if annotate else False,
        fmt='.2f' if annotate else '',
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.5},
        ax=ax,
        annot_kws={'fontsize': 18, 'fontweight': 'bold'} if annotate else None
    )

    count = 0
    # Black border for cells with FDR < 0.05 AND |r| > thred
    for i in range(pivot_corr.shape[0]):
        for j in range(pivot_corr.shape[1]):
            is_sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else False
            corr_val = pivot_corr.iloc[i, j]
            if is_sig and not pd.isna(corr_val) and abs(corr_val) > thred:
                count += 1
                rect = plt.Rectangle((j, i), 1, 1,
                                     fill=False,
                                     edgecolor='black',
                                     linewidth=2.5)
                ax.add_patch(rect)

    print(f"total {count} / {pivot_corr.shape[0] * pivot_corr.shape[1]}\n")
    ax.set_xlabel('CNN Feature Dimension', fontsize=16)
    ax.set_ylabel('Clinical Assessment Score', fontsize=16)
    # ax.set_title('Correlation Heatmap: CNN Features vs Clinical Scores',
    #              fontsize=18, fontweight='bold', pad=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=16)

    plt.tight_layout()

    heatmap_path = os.path.join(save_path, 'cnn_clinical_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")

    return fig, ax


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
        相关系数阈值，仅用于控制是否保存散点图
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    pd.DataFrame
        包含所有分析结果的DataFrame（包括低于阈值的结果）
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
    
    # 第一阶段：计算所有相关系数和原始p值
    print("\nPhase 1: Computing all correlations...")
    for dim in range(n_dims):
        for clinical_score in clinical_scores_list:
            result = analyze_cnn_clinical_correlation(
                cnn_features=cnn_features,
                subject_ids=subject_ids,
                clinical_scores_path=clinical_scores_path,
                feature_dim=dim,
                clinical_score=clinical_score,
                save_path=None,  # 先不保存散点图
                verbose=False,
                thred=0,  # 强制返回所有结果
                p_value_fdr=None  # 先不传入FDR值
            )
            
            if result is not None:
                all_results.append(result)
    
    # 创建初步DataFrame用于FDR校正
    if all_results:
        df_temp = pd.DataFrame(all_results)
        
        # FDR校正（基于所有p值）
        p_values = df_temp['p_value'].values
        reject_fdr, p_fdr, _, _ = multipletests(
            p_values, 
            alpha=0.05, 
            method='fdr_bh'  # Benjamini-Hochberg FDR校正
        )
        
        # 更新结果中的FDR值
        for idx, result in enumerate(all_results):
            result['p_value_fdr'] = p_fdr[idx]
            result['significant_fdr'] = reject_fdr[idx]
        
        # 创建最终DataFrame
        df_results = pd.DataFrame(all_results)
        
        # 按相关系数绝对值排序
        df_results['abs_r'] = np.abs(df_results['pearson_r'])
        df_results = df_results.sort_values('abs_r', ascending=False)
        
        print(f"\nFDR correction completed for {len(df_results)} tests")
        print(f"Significant after FDR correction: {df_results['significant_fdr'].sum()}")
        
        # 第二阶段：为超过阈值的相关绘制散点图（使用FDR校正后的p值）
        print("\nPhase 2: Generating scatter plots for correlations above threshold...")
        for idx, row in df_results.iterrows():
            if row['abs_r'] >= thred:
                analyze_cnn_clinical_correlation(
                    cnn_features=cnn_features,
                    subject_ids=subject_ids,
                    clinical_scores_path=clinical_scores_path,
                    feature_dim=int(row['cnn_dim']),
                    clinical_score=row['clinical_score'],
                    save_path=save_path,
                    verbose=False,
                    thred=thred,
                    p_value_fdr=row['p_value_fdr']
                )
        
        # 保存完整结果到CSV
        results_file = os.path.join(save_path, 'cnn_clinical_correlation_results.csv')
        df_results.to_csv(results_file, index=False)
        print(f"\nComplete results saved to: {results_file}")
        
        # 保存超过阈值的结果到单独文件
        df_results_above_threshold = df_results[df_results['abs_r'] >= thred]
        if len(df_results_above_threshold) > 0:
            threshold_file = os.path.join(save_path, f'cnn_clinical_correlation_results_r>{thred}.csv')
            df_results_above_threshold.to_csv(threshold_file, index=False)
            print(f"Results above threshold (|r| >= {thred}) saved to: {threshold_file}")
        
        # 打印FDR显著的发现
        print("\n" + "=" * 80)
        print("TOP SIGNIFICANT CORRELATIONS (FDR corrected p < 0.05)")
        print("=" * 80)
        sig_results = df_results[df_results['significant_fdr']]
        if len(sig_results) > 0:
            for idx, row in sig_results.iterrows():
                significance = "***" if row['p_value_fdr'] < 0.001 else "**" if row['p_value_fdr'] < 0.01 else "*"
                print(f"CNN Dim {int(row['cnn_dim'])} vs {row['clinical_score']}: "
                      f"r={row['pearson_r']:.4f}, p_uncorrected={row['p_value']:.4f}, "
                      f"p_fdr={row['p_value_fdr']:.4f} {significance}")
        else:
            print("No significant correlations found after FDR correction")
        
        # 打印未校正的显著结果（前20个）
        print("\n" + "=" * 80)
        print("TOP SIGNIFICANT CORRELATIONS (uncorrected p < 0.05, top 20)")
        print("=" * 80)
        sig_uncorrected = df_results[df_results['p_value'] < 0.05]
        if len(sig_uncorrected) > 0:
            for idx, row in sig_uncorrected.head(20).iterrows():
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
                print(f"CNN Dim {int(row['cnn_dim'])} vs {row['clinical_score']}: "
                      f"r={row['pearson_r']:.4f}, p_uncorrected={row['p_value']:.4f} {significance}")
        else:
            print("No significant correlations found at p < 0.05")
        
        # Translate Chinese scale names to English for heatmap display
        SCALE_NAME_TRANSLATION = {
            'MoCA总分': 'MoCA Total',
            '即刻记忆': 'Immediate Memory',
            '延迟回忆': 'Delayed Recall',
            '线索回忆': 'Cued Recall',
            '长时延迟再认': 'Long-delayed Recognition',
            '数字广度顺向': 'Digit Span Forward',
            '数字广度逆向': 'Digit Span Backward',
            '连线测验A': 'TMT-A',
            '连线测验B': 'TMT-B',
            'Boston-初始命名': 'Boston Naming',
        }
        df_heatmap = df_results.copy()
        df_heatmap['clinical_score'] = df_heatmap['clinical_score'].replace(SCALE_NAME_TRANSLATION)

        # Generate correlation heatmap
        print("\n" + "=" * 80)
        print("GENERATING CORRELATION HEATMAP")
        print("=" * 80)

        try:
            fig, ax = plot_correlation_heatmap(
                df_results=df_heatmap,
                save_path=save_path,
                figsize=(16, 12),
                annotate=False,
                thred=thred
            )
            plt.close()

            print("\nHeatmap generated successfully!")

        except Exception as e:
            print(f"Error generating heatmap: {e}")
        
        return df_results
    else:
        print("\nNo results found.")
        return pd.DataFrame()


if __name__ == "__main__":
    # 临床评分文件路径
    clinical_scores_path = "./MMS.txt"  # 请替换为您的实际文件路径
    
    # 加载数据
    node_feature, adj, label, subject_ids, cnns = load_and_preprocess_data()
    
    # CNN特征数据 - cnns 的形状应该是 (N, 32)
    print(f"CNN features shape: {cnns.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    print(f"Unique subjects: {np.unique(subject_ids).shape[0]}")
    
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
        thred=0.3,  # 只保存|r| > 0.3的
    )