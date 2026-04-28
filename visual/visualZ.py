import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import Tuple, Optional, List, Union
import seaborn as sns
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler

import pickle
import os

def procrustes_align(source: np.ndarray, target: np.ndarray, 
                     use_scaling: bool = False) -> np.ndarray:
    """
    将 source 通过 Procrustes 分析对齐到 target 空间
    
    Parameters:
    -----------
    source : ndarray, shape (N, 2)
        待对齐的嵌入
    target : ndarray, shape (N, 2)
        目标嵌入（作为参考）
    
    Returns:
    --------
    aligned : ndarray, shape (N, 2)
        对齐后的嵌入
    """
    # 去均值
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # 计算缩放因子和正交旋转矩阵
    # orthogonal_procrustes 返回旋转矩阵 R 和缩放因子 scale
    R, scale = orthogonal_procrustes(source_centered, target_centered)
    if not use_scaling:
        scale = 1.0  # 强制不缩放
    
    # 对齐：缩放 + 旋转 + 平移
    aligned = scale * source_centered @ R + target_mean
    
    return aligned


def align_all_to_reference(embeddings: List[np.ndarray], 
                           reference_idx: int = 0) -> List[np.ndarray]:
    """
    将所有嵌入对齐到指定的参考嵌入
    
    Parameters:
    -----------
    embeddings : list of ndarray
        三个VAE的嵌入列表
    reference_idx : int
        参考嵌入的索引（默认0，即对齐到VAE1）
    
    Returns:
    --------
    aligned_embeddings : list of ndarray
        对齐后的嵌入列表
    """
    reference = embeddings[reference_idx]
    aligned = [reference]  # 参考嵌入不变
    
    for i, emb in enumerate(embeddings):
        if i == reference_idx:
            continue
        aligned.append(procrustes_align(emb, reference))
    
    # 按原顺序返回
    result = []
    aligned_idx = 0
    for i in range(len(embeddings)):
        if i == reference_idx:
            result.append(embeddings[i])
        else:
            aligned_idx += 1
            result.append(aligned[aligned_idx])
    
    return result

def visualize_procrustes_with_connections(
    embeddings: List[np.ndarray],
    vae_names: List[str],
    labels: np.ndarray,
    reference_idx: int = 0,
    class_names: Optional[List[str]] = None,
    sample_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    绘制 Procrustes 对齐后的单张图，用不同形状区分VAE，用线连接同一个体
    
    Parameters:
    -----------
    embeddings : list of ndarray
        三个VAE的原始嵌入列表
    vae_names : list of str
        VAE名称列表
    labels : ndarray
        类别标签
    reference_idx : int
        对齐参考（默认对齐到VAE1）
    sample_indices : list or None
        要展示连线的样本索引（None则随机选取20个，设为'none'则不画连线）
    class_names : list of str, optional
        类别名称
    figsize : tuple
        图像大小
    save_path : str, optional
        保存路径
    """
    # 对齐
    aligned_embeddings = align_all_to_reference(embeddings, reference_idx)

    # 调试：检查 VAE1 的坐标范围
    # print("VAE1 坐标范围:")
    # print(f"  X: [{aligned_embeddings[0][:, 0].min():.4f}, {aligned_embeddings[0][:, 0].max():.4f}]")
    # print(f"  Y: [{aligned_embeddings[0][:, 1].min():.4f}, {aligned_embeddings[0][:, 1].max():.4f}]")
    # print(f"  均值: ({aligned_embeddings[0][:, 0].mean():.4f}, {aligned_embeddings[0][:, 1].mean():.4f})")
    # print(f"  前5个点:\n{aligned_embeddings[0][:5]}")
    
    n_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # 配色与形状
    class_colors = sns.color_palette('husl', n_classes)
    vae_markers = ['o', 's', '^']  # ○ □ △
    marker_size = 40
    vae_alphas = [1.0, 0.5, 0.5]
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for vae_idx in [2, 1, 0]:  # 改为倒序：先VAE3, 再VAE2, 最后VAE1
        marker = vae_markers[vae_idx]
        aligned_emb = aligned_embeddings[vae_idx]
        vae_name = vae_names[vae_idx]
        for class_idx in range(n_classes):
            mask = labels == class_idx
            ax.scatter(
                aligned_emb[mask, 0],
                aligned_emb[mask, 1],
                c=[class_colors[class_idx]],
                marker=marker,
                label=f'{vae_name} - {class_names[class_idx]}',
                alpha=vae_alphas[vae_idx],
                s=marker_size,
                edgecolors='white' if vae_idx == 0 else 'black',
                linewidth=0.5,
                zorder=2 if vae_idx == 0 else 2  # VAE1最高
            )
    
    ax.set_title(
        f'Procrustes Alignment (Reference: {vae_names[reference_idx]})\n'
        f'○={vae_names[0]}  □={vae_names[1]}  △={vae_names[2]}  ×=Center',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel(f'{vae_names[reference_idx]} Component 1 (aligned)', fontsize=11)
    ax.set_ylabel(f'{vae_names[reference_idx]} Component 2 (aligned)', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    
    # 图例优化：只显示类别和形状含义
    handles, labels_ax = ax.get_legend_handles_labels()
    # 简化图例：只保留类别相关的
    unique_class_handles = []
    unique_class_labels = []
    for class_idx in range(n_classes):
        mask = labels == class_idx
        if np.any(mask):
            # 取第一个VAE的该类别的handle
            for h, l in zip(handles, labels_ax):
                if l.startswith(vae_names[0]) and class_names[class_idx] in l:
                    unique_class_handles.append(h)
                    unique_class_labels.append(class_names[class_idx])
                    break
    
    legend1 = ax.legend(unique_class_handles, unique_class_labels, 
                        title='Classes', loc='upper left', fontsize=9)
    ax.add_artist(legend1)
    
    # 添加形状图例
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker='o', color='gray', label=vae_names[0], 
               markersize=8, linestyle='None'),
        Line2D([0], [0], marker='s', color='gray', label=vae_names[1], 
               markersize=8, linestyle='None'),
        Line2D([0], [0], marker='^', color='gray', label=vae_names[2], 
               markersize=8, linestyle='None'),
    ]
    ax.legend(handles=shape_handles, title='VAE Domain', loc='upper right', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Procrustes 对齐图已保存: {save_path}")
    
    # plt.show()


def compute_procrustes_disparity(
    embeddings: List[np.ndarray],
    labels: np.ndarray,
    reference_idx: int = 0
) -> np.ndarray:
    """
    计算每个样本在三个VAE对齐后的位移量（互补性度量）
    
    Returns:
    --------
    disparities : ndarray, shape (N, 3)
        每列是该样本在某个VAE与参考VAE的欧氏距离
        第一列为0（参考VAE与自身距离为0）
    """
    aligned = align_all_to_reference(embeddings, reference_idx)
    N = len(labels)
    disparities = np.zeros((N, 3))
    
    for i in range(3):
        disparities[:, i] = np.sqrt(np.sum((aligned[i] - aligned[reference_idx])**2, axis=1))
    
    return disparities


def reparameterize(mu: np.ndarray, logvar: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    重参数化采样：从 mu 和 logvar 中采样得到隐变量 z
    
    Parameters:
    -----------
    mu : ndarray, shape (N, L, C, D)
        均值
    logvar : ndarray, shape (N, L, C, D)
        对数方差
    random_state : int
        随机种子
    
    Returns:
    --------
    z : ndarray, shape (N, L, C, D)
        采样后的隐变量
    """
    rng = np.random.RandomState(random_state)
    std = np.exp(0.5 * logvar)
    eps = rng.randn(*mu.shape)
    z = mu + eps * std
    return z


def get_latent_variable(mu: np.ndarray, 
                        logvar: Optional[np.ndarray] = None, 
                        use_sampling: bool = True,
                        random_state: int = 42) -> np.ndarray:
    """
    获取隐变量：可选择采样或直接使用均值
    
    Parameters:
    -----------
    mu : ndarray, shape (N, L, C, D)
        编码器输出的均值
    logvar : ndarray, shape (N, L, C, D) or None
        编码器输出的对数方差（use_sampling=True 时必需）
    use_sampling : bool
        True: 使用重参数化采样得到 z
        False: 直接使用 mu 作为确定性表示
    random_state : int
        随机种子
    
    Returns:
    --------
    z : ndarray, shape (N, L, C, D)
        隐变量
    """
    if use_sampling:
        if logvar is None:
            raise ValueError("当 use_sampling=True 时，必须提供 logvar")
        return reparameterize(mu, logvar, random_state=random_state)
    else:
        return mu


def temporal_pooling(z: np.ndarray, pooling_type: str = 'mean_std') -> np.ndarray:
    """
    对隐变量的时间维度进行池化
    
    Parameters:
    -----------
    z : ndarray, shape (N, L, C, D)
        隐变量
    pooling_type : str
        池化类型: 'mean_std', 'mean_only', 'all_stats'
    
    Returns:
    --------
    features : ndarray
        池化后的特征
    """
    if pooling_type == 'mean_std':
        z_mean = np.mean(z, axis=1)  # (N, C, D)
        z_std = np.std(z, axis=1)    # (N, C, D)
        features = np.concatenate([z_mean, z_std], axis=-1)  # (N, C, 2*D)
    
    elif pooling_type == 'mean_only':
        features = np.mean(z, axis=1)  # (N, C, D)
    
    elif pooling_type == 'all_stats':
        z_mean = np.mean(z, axis=1)
        z_std = np.std(z, axis=1)
        z_max = np.max(z, axis=1)
        z_min = np.min(z, axis=1)
        features = np.concatenate([z_mean, z_std, z_max, z_min], axis=-1)
    
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")
    
    return features


def flatten_channels(features: np.ndarray) -> np.ndarray:
    """将通道维度展平"""
    N = features.shape[0]
    return features.reshape(N, -1)


def standardize_features(features: np.ndarray) -> np.ndarray:
    """组级别标准化"""
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def reduce_with_supervised_umap(
    features: np.ndarray, 
    labels: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    **umap_kwargs
) -> np.ndarray:
    """监督式 UMAP 降维"""
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        **umap_kwargs
    )
    embedding = reducer.fit_transform(features, y=labels)
    return embedding


def reduce_with_lda(
    features: np.ndarray, 
    labels: np.ndarray,
    n_components: int = 2
) -> np.ndarray:
    """LDA 降维"""
    lda = LDA(n_components=n_components)
    embedding = lda.fit_transform(features, labels)
    return embedding


def reduce_with_tsne(
    features: np.ndarray,
    labels: np.ndarray = None,
    n_components: int = 2,
    random_state: int = 42,
    **tsne_kwargs
) -> np.ndarray:
    """t-SNE 降维"""
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        **tsne_kwargs
    )
    embedding = tsne.fit_transform(features)
    return embedding


# =================== 主处理流程（修正版） ===================

def process_single_vae(
    mu: np.ndarray,
    logvar: Optional[np.ndarray],
    labels: np.ndarray,
    use_sampling: bool = True,
    pooling_type: str = 'mean_std',
    reduction_method: str = 'umap',
    random_state: int = 42
) -> np.ndarray:
    """
    处理单个 VAE 的完整流程
    """
    # 1. 获取隐变量
    z = get_latent_variable(mu, logvar, use_sampling, random_state)
    
    # 2. 时间池化
    features = temporal_pooling(z, pooling_type)
    
    # 3. 展平
    features = flatten_channels(features)
    
    # 4. 标准化
    features = standardize_features(features)
    
    # 5. 降维
    if reduction_method == 'umap':
        embedding = reduce_with_supervised_umap(features, labels, random_state=random_state)
    elif reduction_method == 'lda':
        n_classes = len(np.unique(labels))
        max_components = min(n_classes - 1, features.shape[1])
        if max_components < 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10, random_state=random_state)
            features = pca.fit_transform(features)
            max_components = min(n_classes - 1, 10)
        embedding = reduce_with_lda(features, labels, n_components=min(2, max_components))
    elif reduction_method == 'tsne':
        embedding = reduce_with_tsne(features, labels, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {reduction_method}")
    
    return embedding


def process_all_vaes(
    mu1: np.ndarray, logvar1: np.ndarray,
    mu2: np.ndarray, logvar2: Optional[np.ndarray],
    mu3: np.ndarray, logvar3: Optional[np.ndarray],
    labels: np.ndarray,
    use_sampling: Union[bool, List[bool]] = True,
    method: str = 'umap',
    pooling_type: str = 'mean_std',
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    完整处理三个 VAE
    
    Parameters:
    -----------
    mu1, logvar1 : VAE1 的均值和logvar
    mu2, logvar2 : VAE2 的均值和logvar（logvar可以为None如果不采样）
    mu3, logvar3 : VAE3 的均值和logvar（logvar可以为None如果不采样）
    use_sampling : bool or list of bool
        True/False: 统一设置
        [True, False, True]: 分别对应三个VAE
    """
    # 统一 use_sampling 格式
    if isinstance(use_sampling, bool):
        use_sampling_list = [use_sampling] * 3
    else:
        use_sampling_list = use_sampling
    
    # 准备数据
    vaes_data = [
        {'mu': mu1, 'logvar': logvar1, 'name': 'VAE1 (Time)'},
        {'mu': mu2, 'logvar': logvar2, 'name': 'VAE2 (Frequency)'},
        {'mu': mu3, 'logvar': logvar3, 'name': 'VAE3 (Phase)'}
    ]
    
    embeddings = []
    vae_names = []
    
    for idx, vae in enumerate(vaes_data):
        sampling_flag = use_sampling_list[idx]
        sampling_str = "采样" if sampling_flag else "均值"
        
        embedding = process_single_vae(
            mu=vae['mu'],
            logvar=vae['logvar'],
            labels=labels,
            use_sampling=sampling_flag,
            pooling_type=pooling_type,
            reduction_method=method,
            random_state=random_state
        )
        
        embeddings.append(embedding)
        vae_names.append(f"{vae['name']} ({sampling_str})")
        print(f"{vae['name']}: 使用{sampling_str}, 嵌入形状 {embedding.shape}")
    
    return embeddings, vae_names, labels


# =================== 可视化 ===================

def visualize_embeddings(
    embeddings: List[np.ndarray],
    vae_names: List[str],
    labels: np.ndarray,
    method: str = 'UMAP',
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None
):
    """可视化三个VAE的降维结果（并排对比）"""
    n_classes = len(np.unique(labels))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    colors = sns.color_palette('husl', n_classes)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
    
    for idx, (embedding, name, ax) in enumerate(zip(embeddings, vae_names, axes)):
        for class_idx in range(n_classes):
            mask = labels == class_idx
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[class_idx]],
                label=class_names[class_idx],
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )
        
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.legend(loc='best', fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    fig.suptitle(f'Latent Space Visualization ({method})', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图像已保存: {save_path}")
    
    # plt.show()


# =================== 使用示例 ===================
def load_data():
    """加载数据"""
    with open('../data2.pkl', 'rb') as f:
        data = pickle.load(f)

    mu1, mu2, mu3, logvar1, logvar2, logvar3, labels, weight, bias = data

    return mu1, mu2, mu3, logvar1, logvar2, logvar3, labels

if __name__ == '__main__':

    class_names = ['AD', 'SCD', 'MCI', 'NC']
    mu1, mu2, mu3, logvar1, logvar2, logvar3, labels = load_data()

    path = './output_visualZ'
    os.makedirs(path, exist_ok=True)

    # ===== Procrustes 对齐可视化 =====
    print("\n" + "=" * 60)
    print("Procrustes 对齐 — 多视角互补性可视化")
    print("=" * 60)
    
    # 使用UMAP降维结果做对齐（类别界限最分明）
    embeddings_for_align, names_for_align, labs_for_align = process_all_vaes(
        mu1, logvar1, mu2, logvar2, mu3, logvar3,
        labels,
        use_sampling=True,
        method='umap',
        random_state=42
    )
    
    # 绘制对齐图（对齐到VAE1，随机选30个样本画连线）
    visualize_procrustes_with_connections(
        embeddings=embeddings_for_align,
        vae_names=['VAE1 (Time)', 'VAE2 (Freq)', 'VAE3 (Phase)'],
        labels=labs_for_align,
        reference_idx=0,           # 对齐到时间域VAE
        class_names=class_names,
        sample_indices=None,       # None=随机选20个；设为'none'=不画连线
        save_path=os.path.join(path, 'procrustes_alignment.png')
    )
    
    # 计算并输出互补性统计
    disparities = compute_procrustes_disparity(
        embeddings_for_align, labs_for_align, reference_idx=0
    )
    
    print("\n样本跨视角平均位移（互补性指标）：")
    for i, name in enumerate(['VAE1 (ref)', 'VAE2', 'VAE3']):
        print(f"  {name}: {disparities[:, i].mean():.4f} ± {disparities[:, i].std():.4f}")
    
    print(f"\n平均跨视角位移: {disparities[:, 1:].mean():.4f}")
    print("位移越大 → 三视角互补性越强")
    
    
    # ===== 正确做法：三个VAE都有mu和logvar，都进行采样 =====
    print("=" * 60)
    print("【正确做法】三个VAE都进行重参数化采样")
    print("=" * 60)
    
    # 方法1：全部采样
    embeddings_sampled, names_sampled, labs = process_all_vaes(
        mu1, logvar1, mu2, logvar2, mu3, logvar3,
        labels,
        use_sampling=True,  # 全部采样
        method='umap'
    )
    visualize_embeddings(
        embeddings_sampled, names_sampled, labs,
        method='UMAP (全部采样)', class_names=class_names,
        save_path=os.path.join(path, 'vae_all_sampled.png')
    )
    
    # # 方法2：全部使用均值（确定性编码）
    # print("\n" + "=" * 60)
    # print("【对比】三个VAE都使用均值（不采样）")
    # print("=" * 60)
    # embeddings_det, names_det, labs = process_all_vaes(
    #     mu1, logvar1, mu2, logvar2, mu3, logvar3,
    #     labels,
    #     use_sampling=False,  # 全部用均值
    #     method='umap'
    # )
    # visualize_embeddings(
    #     embeddings_det, names_det, labs,
    #     method='UMAP (均值)', class_names=class_names,
    #     save_path='vae_all_deterministic.png'
    # )
    
    # # 方法3：混合使用（展示灵活性）
    # print("\n" + "=" * 60)
    # print("【混合】VAE1采样, VAE2和VAE3用均值")
    # print("=" * 60)
    # embeddings_mix, names_mix, labs = process_all_vaes(
    #     mu1, logvar1, mu2, logvar2, mu3, logvar3,
    #     labels,
    #     use_sampling=[True, False, False],  # 逐个指定
    #     method='umap'
    # )
    # visualize_embeddings(
    #     embeddings_mix, names_mix, labs,
    #     method='UMAP (混合)', class_names=class_names,
    #     save_path='vae_mixed.png'
    # )