import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import entropy
import os
import pickle

import pdb


def softmax(x):
    """计算softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def compute_entropy(logits):
    """计算分类器的熵"""
    probs = softmax(logits)
    return entropy(probs, axis=-1)


def compute_classifier_logits(z_2d, weight_2d, bias):
    """
    在2D网格上计算分类器的logits
    z_2d: (H, W, 2) 网格点
    weight_2d: (num_classes, 2) 降维后的权重
    bias: (num_classes,) 原始偏置
    """
    H, W, _ = z_2d.shape
    num_classes = weight_2d.shape[0]
    z_flat = z_2d.reshape(-1, 2)  # (H*W, 2)
    logits = z_flat @ weight_2d.T + bias  # (H*W, num_classes)
    return logits


def draw_confidence_ellipse(ax, mu, cov, color, alpha=0.3, n_std=2.0):
    """
    绘制95%置信椭圆（n_std=2.0近似于95%）
    mu: (2,) 均值
    cov: (2, 2) 协方差矩阵（对角矩阵，因为论文使用的是对角高斯）
    """
    from matplotlib.patches import Ellipse
    
    # 对于对角协方差矩阵，椭圆轴对齐于坐标轴
    # 如果cov是对角的，直接用sqrt(diag)作为宽度和高度
    var = np.diag(cov)  # 方差
    std = np.sqrt(var)  # 标准差
    
    # 椭圆宽度和高度（95%置信区间：n_std=2.0近似）
    width = 2 * n_std * std[0]
    height = 2 * n_std * std[1]
    
    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=0,  # 对角协方差矩阵，无旋转
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.5
    )
    ax.add_patch(ellipse)


def reduce_dimension(data, method='pca', n_components=2, random_state=42):
    """
    降维函数
    data: (N, D) 待降维数据
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(data), reducer

def load_data():
    """加载数据"""
    with open('../data2.pkl', 'rb') as f:
        data = pickle.load(f)

    mu1, mu2, mu3, logvar1, logvar2, logvar3, labels, weight, bias = data

    result = {}
    result['vae1_mu'] = mu1 #shape (N, L, C, D) N样本个数，L时间长度，C通道个数，D特征维度
    result['vae2_mu'] = mu2 #shape (N, L, C, D)
    result['vae3_mu'] = mu3 #shape (N, L, C, D)
    result['vae1_logvar'] = logvar1 #shape (N, L, C, D)
    result['vae2_logvar'] = logvar2 #shape (N, L, C, D)
    result['vae3_logvar'] = logvar3 #shape (N, L, C, D)
    result['labels'] = labels
    result['dense3_weight'] = weight
    result['dense3_bias'] = bias

    return result

def visualize_vib(data, save_dir='./vis_results', method='pca'):
    """
    主可视化函数
    分别对三个VAE生成可视化图
    
    参数:
        data_path: npz文件路径
        save_dir: 保存图片的目录
        method: 降维方法 'pca' 或 'tsne'
    """
    
    vae_keys = ['vae1', 'vae2', 'vae3']
    labels = data['labels']  # (N,)
    dense3_weight = data['dense3_weight']  # (num_classes, 32)
    dense3_bias = data['dense3_bias']  # (num_classes,)
    num_classes = dense3_weight.shape[0]
    
    # 定义颜色映射（从论文中看，使用10种颜色对应类别）
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(num_classes)]
    
    for vae_key in vae_keys:
        mu_key = f'{vae_key}_mu'
        logvar_key = f'{vae_key}_logvar'
        
        if mu_key not in data or data[mu_key] is None:
            print(f"Skip {vae_key}: no data")
            continue
        
        mu = data[mu_key]      # (N, L, C, D) 或 (N, C, D)
        logvar = data[logvar_key]  # (N, L, C, D) 或 (N, C, D)
        
        # ============================================================
        # 步骤1：展平并降维 mu 和 logvar
        # ============================================================
        N = mu.shape[0]
        
        mu_flat = mu.reshape(N, -1)
        logvar_flat = logvar.reshape(N, -1)
        
        # weight展平或pad到与mu相同维度
        mu_dim = mu_flat.shape[1]
        weight_flat = dense3_weight  # (num_classes, 32)
        if weight_flat.shape[1] < mu_dim:
            weight_padded = np.pad(weight_flat, ((0, 0), (0, mu_dim - weight_flat.shape[1])), mode='constant')
        else:
            weight_padded = weight_flat[:, :mu_dim]
        
        # 用mu和weight统一拟合PCA
        all_for_reduce = np.concatenate([mu_flat, weight_padded], axis=0)
        all_2d, reducer = reduce_dimension(all_for_reduce, method=method, n_components=2)
        
        mu_2d = all_2d[:N]
        weight_2d = all_2d[N:]
        
        # logvar用同一个PCA变换（不重新拟合）
        var_flat = np.exp(np.clip(logvar_flat, -20, 20))
        all_var = np.concatenate([var_flat, weight_padded], axis=0)
        all_var_2d = reducer.transform(all_var)
        var_2d = np.abs(all_var_2d[:N]) + 1e-6
        
        # ============================================================
        # 步骤2：计算背景熵热力图
        # ============================================================
        # 创建2D网格
        x_min, x_max = mu_2d[:, 0].min() - 1, mu_2d[:, 0].max() + 1
        y_min, y_max = mu_2d[:, 1].min() - 1, mu_2d[:, 1].max() + 1
        
        grid_size = 100
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                              np.linspace(y_min, y_max, grid_size))
        grid_points = np.stack([xx, yy], axis=-1)  # (grid_size, grid_size, 2)
        
        # 在网格上计算分类器logits和熵
        grid_logits = compute_classifier_logits(grid_points, weight_2d, dense3_bias)
        grid_entropy = compute_entropy(grid_logits).reshape(grid_size, grid_size)
        
        # ============================================================
        # 步骤3：绘制主图
        # ============================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制背景：熵热力图（灰度，低熵=暗，高熵=亮）
        # 论文中使用灰度图
        im = ax.imshow(grid_entropy, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='gray', aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Entropy H(q(y|z))', shrink=0.8)


        # 在 imshow 和椭圆绘制之后，设置坐标轴范围并自动缩放
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.autoscale_view()

        global_std_x = np.median(np.sqrt(var_2d[:, 0]))
        global_std_y = np.median(np.sqrt(var_2d[:, 1]))
        

        # 调整这个值来控制椭圆大小
        if vae_key == "vae1":
            scale_factor = 1
        elif vae_key == "vae2":
            scale_factor = 0.03
        else:
            scale_factor = 1
        
        # 绘制每个样本的置信椭圆
        for i in range(N):
            label_i = int(labels[i])
            color = colors[label_i % num_classes]
            
            local_std_x = np.sqrt(var_2d[i, 0]) * scale_factor
            local_std_y = np.sqrt(var_2d[i, 1]) * scale_factor
            
            # 混合局部和全局标准差
            std_x = np.exp(0.3 * np.log(local_std_x + 1e-6) + 0.7 * np.log(global_std_x + 1e-6))
            std_y = np.exp(0.3 * np.log(local_std_y + 1e-6) + 0.7 * np.log(global_std_y + 1e-6))

            cov_i = np.diag([std_x**2, std_y**2])
            
            # 降低alpha值，增加透明度
            draw_confidence_ellipse(ax, mu_2d[i], cov_i, color, alpha=0.05, n_std=1.0)  # 降低alpha和n_std
        
        
        # 保存
        fig.savefig(os.path.join(save_dir, f'{vae_key}_vib_visualization.png'), dpi=150, bbox_inches='tight')
        # fig2.savefig(os.path.join(save_dir, f'{vae_key}_vib_class_centers.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        # plt.close(fig2)
        
        print(f"Saved {vae_key} visualizations to {save_dir}")
    
    print("All visualizations complete.")


if __name__ == '__main__':
    # 调用方式
    save_dir = './output_visualZ'                # 输出目录
    os.makedirs(save_dir, exist_ok=True)

    data = load_data()
    
    # 使用PCA降维
    visualize_vib(data, save_dir, method='pca')
    
    # 或使用t-SNE降维
    # visualize_vib(data_path, save_dir, method='tsne')