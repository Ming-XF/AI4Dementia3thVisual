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

from graph import calculate_graph_metrics_fast

import pdb
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    加载数据
    """
    with open('../data.pkl', 'rb') as f:
        data = pickle.load(f)

    node_feature, adj, label, subject_id, cnn, r1, r2, r3, ts = data
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

def extract_fc_feature(
    node_feature: np.ndarray,
    adj: np.ndarray,
    subject_ids: np.ndarray,
    feature_type: str = 'connection',
    connections_group: str = 'positive',  # 'positive' 或 'negative'
    significant_connections_file: str = None,  # 显著连接文件路径
    use_original: bool = True
) -> pd.DataFrame:
    """
    提取FC特征（连接强度或图论指标）
    
    Parameters:
    -----------
    node_feature : np.ndarray
        原始EEG构建的FC图，形状为(n_samples, n_nodes, n_nodes)或(n_samples, n_features)
    adj : np.ndarray
        VAE降噪后的FC图，形状为(n_samples, n_nodes, n_nodes)
    subject_ids : np.ndarray
        每个FC图对应的受试者ID，形状为(n_samples,)
    feature_type : str
        特征类型：'connection'表示连接强度，'clustering'表示平均聚类系数
    connections_group : str
        连接分组：'positive'为正T值组，'negative'为负T值组
    significant_connections_file : str
        显著连接文件的路径
    use_original : bool
        是否使用原始FC图(node_feature)，False则使用VAE降噪后的图(adj)
    
    Returns:
    --------
    pd.DataFrame
        包含subject_id和FC特征的DataFrame
    """
    n_samples = len(subject_ids)
    
    # 选择使用的FC图数据
    if use_original:
        fc_data = node_feature
        data_type = 'original'
    else:
        fc_data = adj
        data_type = 'vae_denoised'
    
    # 根据feature_type提取不同特征
    if feature_type == 'connection':
        # 读取显著连接并分组
        positive_conns, negative_conns = load_significant_connections(significant_connections_file)
        
        # 根据connections_group选择对应组
        if connections_group == 'positive':
            target_connections = positive_conns
            group_name = 'positive'
        elif connections_group == 'negative':
            target_connections = negative_conns
            group_name = 'negative'
        else:
            raise ValueError("connections_group must be 'positive' or 'negative'")
        
        if not target_connections:
            raise ValueError(f"No {group_name} connections found in the file")
        
        # 计算平均连接强度
        avg_connection_strengths = []
        for idx in range(n_samples):
            connection_values = []
            for i, j in target_connections:
                strength = fc_data[idx, i, j]
                connection_values.append(strength)
            
            # 计算该样本所有显著连接的平均强度
            avg_strength = np.mean(connection_values)
            avg_connection_strengths.append(avg_strength)
        
        feature_values = avg_connection_strengths
        feature_name = f'avg_connection_strength_{group_name}_{data_type}'
        
    elif feature_type == 'clustering':
        # 计算平均聚类系数
        clustering_coeffs = []
        for idx in range(n_samples):
            fc_matrix = fc_data[idx, :, :]
            
            metrics = calculate_graph_metrics_fast(fc_matrix, threshold=0.25, sub_graph=True)
            clustering_coeffs.append(metrics['C'])
        
        feature_values = clustering_coeffs
        feature_name = f'avg_clustering_{data_type}'
        
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Use 'connection' or 'clustering'")
    
    # 创建DataFrame
    df_feature = pd.DataFrame({
        'subject_id': subject_ids,
        feature_name: feature_values
    })

    # 每个受试者只保留第一个数据
    df_feature = df_feature.groupby('subject_id').first().reset_index()
    
    return df_feature


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
    use_original: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_stats: bool = True,
    color_by_subject: bool = True,
    thred = 0.3
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

    n = len(x)

    # 统计信息（无论 r 是否低于阈值都计算）
    stats_dict = {
        'n_samples': n,
        'pearson_r': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'connection': connection_col,
        'score': clinical_score,
    }

    if abs(r_value) < thred:
        return None, None, stats_dict

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
    ax.set_xlabel(connection_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{clinical_score} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{clinical_score} vs FC {connection_col}\n{data_type}', 
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

    # 保存图形
    if save_path:
        plt.savefig(os.path.join(save_path, f'{connection_col}_{clinical_score}_{stats_dict['pearson_r']:.2f}_{stats_dict['p_value']:.3f}.png'), dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    
    return fig, ax, stats_dict

def load_significant_connections(filepath: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    从文件中读取显著连接，按T值正负分组
    
    Parameters:
    -----------
    filepath : str
        显著连接CSV文件路径
    
    Returns:
    --------
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
        (正T值连接列表, 负T值连接列表)
    """
    df = pd.read_csv(filepath)
    
    # 按T值正负分组
    positive_connections = df[df['t_statistic'] > 0][['Region1', 'Region2']].apply(
        lambda row: (int(row['Region1']), int(row['Region2'])), axis=1
    ).tolist()
    
    negative_connections = df[df['t_statistic'] < 0][['Region1', 'Region2']].apply(
        lambda row: (int(row['Region1']), int(row['Region2'])), axis=1
    ).tolist()
    
    return positive_connections, negative_connections

def analyze_fc_clinical_correlation(

    node_feature: np.ndarray,
    adj: np.ndarray,
    subject_ids: np.ndarray,
    clinical_scores_path: str,
    clinical_score: str,
    feature_type: str = 'connection',
    connections_group: str = 'positive',  # 新增参数
    significant_connections_file: str = None,  # 新增参数
    use_original: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True,
    thred = 0.3
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
        print(f"Analyzing {feature_type}")
        print(f"Clinical score: {clinical_score}")
        print(f"Using {'Original' if use_original else 'VAE Denoised'} FC data")
        print("-" * 60)
    
    # 1. 加载临床数据
    df_clinical = load_clinical_scores(clinical_scores_path)
    if verbose:
        print(f"\nLoaded clinical data for {len(df_clinical)} subjects")
        print(f"Available clinical scores: {list(df_clinical.columns)}")
    
    # 2. 提取FC特征
    df_feature = extract_fc_feature(
        node_feature, adj, subject_ids, feature_type, connections_group, significant_connections_file, use_original
    )
    if verbose:
        print(f"\nExtracted {feature_type} for {len(df_feature)} FC graphs")
        print(f"Unique subjects in FC data: {df_feature['subject_id'].nunique()}")
    
    # 3. 合并数据
    feature_col = df_feature.columns[1]  # 获取指标列名
    df_merged = merge_clinical_and_fc_data(
        df_clinical, df_feature, clinical_score
    )
    if verbose:
        print(f"\nMerged data: {len(df_merged)} samples")
        print(f"Subjects with both FC and clinical data: {df_merged['subject_id'].nunique()}")
        print(f"\nData summary:")
        print(df_merged.describe())
    
    # 4. 绘制散点图和拟合线
    fig, ax, stats_dict = plot_scatter_with_fit(
        df_merged, feature_col, clinical_score, 
        use_original, save_path=save_path, thred=thred
    )
    
    # 5. 打印统计结果
    if verbose:
        print("\n" + "=" * 60)
        print("Statistical Results")
        print("=" * 60)
        if fig is None:
            print("(Figure skipped: |r| below threshold)")
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

# 认知领域分组定义
COGNITIVE_DOMAINS = {
    'CDR_SOB':       ('Global Dementia Severity', '#8B0000'),
    'CDR':           ('Global Dementia Severity', '#8B0000'),
    '即刻记忆':          ('Episodic Memory', '#2166AC'),
    '线索回忆':          ('Episodic Memory', '#2166AC'),
    '延迟回忆':          ('Episodic Memory', '#2166AC'),
    '长时延迟再认':        ('Episodic Memory', '#2166AC'),
    'MMSE':           ('Global Cognition', '#D73027'),
    'MoCA总分':        ('Global Cognition', '#D73027'),
    '连线测验A':         ('Executive Function', '#E08214'),
    '连线测验B':         ('Executive Function', '#E08214'),
    'TMT B-A':        ('Executive Function', '#E08214'),
    'Boston-初始命名':    ('Language', '#4DAF4A'),
    'CDT':            ('Visuospatial', '#7B3294'),
    '数字广度逆向':         ('Attention / WM', '#999999'),
    '数字广度顺向':         ('Attention / WM', '#999999'),
}

# 用于在柱状图中替代过长的中文标签的英文短标签
SCORE_SHORT_LABELS = {
    'MMSE': 'MMSE', 'MoCA总分': 'MoCA', '即刻记忆': 'Immediate\nRecall',
    '延迟回忆': 'Delayed\nRecall', '线索回忆': 'Cued\nRecall',
    '长时延迟再认': 'Long-delayed\nRecognition', '数字广度顺向': 'Digit Span\nForward',
    '数字广度逆向': 'Digit Span\nBackward', '连线测验A': 'TMT-A',
    '连线测验B': 'TMT-B', 'Boston-初始命名': 'Boston\nNaming',
    'CDR_SOB': 'CDR-SOB', 'CDR': 'CDR', 'TMT B-A': 'TMT B-A', 'CDT': 'CDT',
}


def plot_grouped_bar_chart(df_pos, df_neg, save_path):
    """分组柱状图：x轴为临床量表，按量表类型分左右两组，竖线分隔，展示镜像反转及FDR显著性"""
    plt.rcParams['font.family'] = 'DejaVu Sans'

    IMPAIRMENT_SCALES = {'CDR_SOB', 'CDR', '连线测验A', '连线测验B', 'TMT B-A'}

    merged = df_pos[['score', 'pearson_r', 'significant_fdr']].merge(
        df_neg[['score', 'pearson_r', 'significant_fdr']],
        on='score', suffixes=('_pos', '_neg')
    )
    merged['is_impairment'] = merged['score'].isin(IMPAIRMENT_SCALES)
    # 损伤量表组按 |r| 降序，能力量表组也按 |r| 降序
    merged = pd.concat([
        merged[merged['is_impairment']].sort_values('pearson_r_pos', ascending=True),
        merged[~merged['is_impairment']].sort_values('pearson_r_pos', ascending=False),
    ])

    labels = [SCORE_SHORT_LABELS.get(s, s) for s in merged['score']]
    r_pos = merged['pearson_r_pos'].values
    r_neg = merged['pearson_r_neg'].values
    sig_pos = merged['significant_fdr_pos'].values
    sig_neg = merged['significant_fdr_neg'].values
    is_impairment = merged['is_impairment'].values

    x = np.arange(len(labels))
    width = 0.35

    # ====== 可调参数：控制x轴延伸和图例偏移距离 ======
    X_AXIS_RIGHT_EXTEND = 1.5   # x轴向右延伸的额外单位（增大则柱子区域更宽，图例更靠右）
    LEGEND_X_OFFSET = 0.7       # 图例在axes中的x位置（0~1，越大越靠右）
    FIG_WIDTH = 20               # 画布宽度
    # ====================================================

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 9))

    ax.bar(x - width/2, r_pos, width,
           color='#2166AC', edgecolor='white', linewidth=0.5,
           label='Degenerative Disconnection (NC > AD)')
    ax.bar(x + width/2, r_neg, width,
           color='#E08214', edgecolor='white', linewidth=0.5,
           label='Pathological Hyper-connectivity (AD > NC)')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

    # 两类量表之间的分隔竖线
    n_impairment = is_impairment.sum()
    if 0 < n_impairment < len(labels):
        ax.axvline(x=n_impairment - 0.5, color='#555555', linewidth=1.2, linestyle='-', alpha=0.6)

    for i in range(len(x)):
        if sig_pos[i]:
            ax.text(x[i] - width/2, r_pos[i] + (0.04 if r_pos[i] >= 0 else -0.06),
                    '*', ha='center', va='center', fontsize=20, color='#2166AC', fontweight='bold')
        else:
            ax.text(x[i] - width/2, r_pos[i] + (0.04 if r_pos[i] >= 0 else -0.06),
                    '×', ha='center', va='center', fontsize=20, color='#888888', fontweight='bold')
        if sig_neg[i]:
            ax.text(x[i] + width/2, r_neg[i] + (0.04 if r_neg[i] >= 0 else -0.06),
                    '*', ha='center', va='center', fontsize=20, color='#E08214', fontweight='bold')
        else:
            ax.text(x[i] + width/2, r_neg[i] + (0.04 if r_neg[i] >= 0 else -0.06),
                    '×', ha='center', va='center', fontsize=20, color='#888888', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22, rotation=45, ha='right')
    ax.set_ylabel("Pearson's r", fontsize=22)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=22)

    # 图例：两组连接 + FDR显著性标记
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#2166AC', edgecolor='white', label='Degenerative Disconnection'),
        Patch(facecolor='#E08214', edgecolor='white', label='Pathological Hyper-connectivity'),
        Line2D([0], [0], marker='*', linestyle='None', markerfacecolor='black', markersize=12,
               label='FDR significant (p < 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20, framealpha=0.9,
              handlelength=1.5, handleheight=1.0)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlim(-0.6, len(labels) - 0.4 + X_AXIS_RIGHT_EXTEND)

    # 分组标注：放在x轴上方、柱子下方
    if n_impairment > 0:
        x_impair_axes = ((n_impairment - 1) / 2) / (len(labels) - 1) if len(labels) > 1 else 0.5
        ax.text(x_impair_axes, 0.02, 'Impairment scales (higher = worse)',
                ha='center', va='bottom', fontsize=18, color='#555555',
                transform=ax.transAxes)
    if n_impairment < len(labels):
        x_ability_axes = (n_impairment + (len(labels) - n_impairment - 1) / 2) / (len(labels) - 1) if len(labels) > 1 else 0.5
        ax.text(x_ability_axes, 0.02, 'Ability scales (higher = better)',
                ha='center', va='bottom', fontsize=18, color='#555555',
                transform=ax.transAxes)

    sns.despine()
    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.95)
    fig.savefig(os.path.join(save_path, 'fig_grouped_bar_mirror_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Grouped bar chart saved to: {save_path}/fig_grouped_bar_mirror_correlation.png")


def plot_horizontal_bar_chart(df_pos, save_path, legend_y=0.92):
    """水平条形图：按认知领域颜色编码展示|r|梯度，映射AD病理层级

    Parameters:
    -----------
    legend_y : float
        图例顶部在axes中的y位置（0~1），默认0.98（最顶部）
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'

    df = df_pos.copy()
    df['domain'], df['color'] = zip(*df['score'].map(
        lambda s: COGNITIVE_DOMAINS.get(s, ('Other', '#AAAAAA'))))
    df = df.sort_values('abs_r', ascending=True)

    domain_order = [
        'Global Dementia Severity', 'Episodic Memory', 'Global Cognition',
        'Executive Function', 'Language', 'Visuospatial', 'Attention / WM'
    ]
    domain_colors = {
        'Global Dementia Severity': '#8B0000',
        'Episodic Memory': '#2166AC',
        'Global Cognition': '#D73027',
        'Executive Function': '#E08214',
        'Language': '#4DAF4A',
        'Visuospatial': '#7B3294',
        'Attention / WM': '#999999',
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = 0
    y_ticks, y_labels, y_colors = [], [], []
    for domain in domain_order:
        sub = df[df['domain'] == domain]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            ax.barh(y_pos, row['abs_r'], height=0.55,
                    color=domain_colors[domain], edgecolor='white', linewidth=0.5)
            ax.text(row['abs_r'] + 0.012, y_pos, f"{row['abs_r']:.2f}", va='center', fontsize=9)
            y_ticks.append(y_pos)
            label_text = SCORE_SHORT_LABELS.get(row['score'], row['score']).replace('\n', ' ')
            y_labels.append(label_text)
            y_colors.append(domain_colors[domain])
            y_pos += 1
        y_pos += 0.45

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    for i, (tick, color) in enumerate(zip(ax.get_yticklabels(), y_colors)):
        tick.set_color(color)
    ax.set_xlabel("|Pearson's r| (Degenerative Disconnection Group)", fontsize=11)
    ax.tick_params(axis='x', labelsize=11)

    ax.set_xlim(0, 0.85)
    ax.invert_yaxis()

    y_bottom = y_pos + 2
    ax.set_ylim(-0.8, y_bottom)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=domain_colors[d], ec='white') for d in domain_order]
    ax.legend(legend_handles, domain_order, loc='upper right', fontsize=11, framealpha=0.9, ncol=1,
              title='Cognitive Domain', title_fontsize=12,
              bbox_to_anchor=(0.98, legend_y))

    sns.despine()
    plt.subplots_adjust(left=0.28, right=0.95, top=0.92, bottom=0.15)

    pathology_labels = ['MTL (Early)', 'Neocortex (Mid)', 'Primary (Late)']
    pathology_x_data = [0.68, 0.53, 0.32]
    for px, pl in zip(pathology_x_data, pathology_labels):
        ax.text(px, y_bottom - 1.4, pl, fontsize=10, ha='center', va='center', color='#555555')
    ax.text(0.49, y_bottom - 0.6, '<-- AD Pathology Progression -->', fontsize=11,
            ha='center', va='center', color='#555555', fontstyle='italic')

    fig.savefig(os.path.join(save_path, 'fig_horizontal_bar_domain_gradient.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Horizontal bar chart saved to: {save_path}/fig_horizontal_bar_domain_gradient.png")


if __name__ == "__main__":
    os.makedirs('./output_score_fc', exist_ok=True)
    clinical_scores_path = "./MMS.txt"
    node_feature, adj, _, subject_ids = load_and_preprocess_data()
    significant_connections_file = "./model_2_testset_result/cvib0_NC vs AD_high_quality_connections.csv"

    items = ['MMSE','MoCA总分','即刻记忆','延迟回忆','线索回忆','长时延迟再认','数字广度顺向','数字广度逆向','连线测验A','连线测验B','Boston-初始命名','CDR_SOB','CDR','TMT B-A','CDT']

    saved_results = {}

    for connections_group in ['positive', 'negative']:
        all_results = []
        for clinical_score in items:
            result = analyze_fc_clinical_correlation(
                node_feature=node_feature,
                adj=adj,
                subject_ids=subject_ids,
                clinical_scores_path=clinical_scores_path,
                feature_type='connection',
                connections_group=connections_group,
                significant_connections_file=significant_connections_file,
                clinical_score=clinical_score,
                use_original=False,
                save_path='./output_score_fc',
                thred=0.3,
            )
            if result is not None:
                all_results.append(result)
        if all_results:
            df_results = pd.DataFrame(all_results)

            p_values = df_results['p_value'].values
            reject_fdr, p_fdr, _, _ = multipletests(
                p_values,
                alpha=0.05,
                method='fdr_bh'
            )

            df_results['p_value_fdr'] = p_fdr
            df_results['significant_fdr'] = reject_fdr
            df_results['abs_r'] = np.abs(df_results['pearson_r'])
            df_results = df_results.sort_values('abs_r', ascending=False)

            results_file = os.path.join("./output_score_fc", f'fc_clinical_correlation_{connections_group}_results.csv')
            df_results.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")

            print("\n" + "=" * 80)
            print(f"FDR-Corrected Results Summary [{connections_group} connections]")
            print("=" * 80)
            print(f"{'Clinical Score':<20s} {'r':>8s} {'p_raw':>10s} {'p_fdr':>10s} {'FDR sig':>8s}")
            print("-" * 60)
            for _, row in df_results.iterrows():
                print(f"{row['score']:<20s} {row['pearson_r']:>8.3f} {row['p_value']:>10.4f} {row['p_value_fdr']:>10.4f} {'*' if row['significant_fdr'] else '':>8s}")
            print("-" * 60)
            n_sig = df_results['significant_fdr'].sum()
            print(f"Significant after FDR correction: {n_sig}/{len(df_results)}")

            saved_results[connections_group] = df_results

    if 'positive' in saved_results and 'negative' in saved_results:
        print("\n" + "=" * 80)
        print("Generating summary figures...")
        print("=" * 80)
        plot_grouped_bar_chart(saved_results['positive'], saved_results['negative'], './output_score_fc')
        plot_horizontal_bar_chart(saved_results['positive'], './output_score_fc')
        print("\nAll done.")


