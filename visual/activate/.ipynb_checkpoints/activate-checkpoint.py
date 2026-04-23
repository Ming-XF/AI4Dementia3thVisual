import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import templateflow.api as tflow
from nilearn.surface import SurfaceImage, load_surf_data
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_roi, show

from brain import coordinates_data


import pdb

def get_active_from_csv(path):
    # 读取 CSV 文件
    df = pd.read_csv(path)  # 请替换为实际文件路径
    
    # 合并 Region1 和 Region2 两列，统计每个区域出现的次数
    region_counts = pd.concat([df['Region1'], df['Region2']]).value_counts().sort_index()

    return region_counts.to_dict()

# fs_name格式：ctx-lh-bankssts -> l.bankssts
def fs_name_to_key(fs_name):
    parts = fs_name.split('-')
    if len(parts) >= 3:
        hemisphere = parts[1]  # 'lh' or 'rh'
        region_name = parts[2]
        
        # 转换hemisphere标识
        hemi_short = 'l' if hemisphere == 'lh' else 'r'
        
        return f"{hemi_short}.{region_name}"
    return None

def generate_colormap_based_on_activation(old_df, coordinates_dict, activation_dict, cmap='coolwarm', vmin=None, vmax=None):
    """
    根据激活值生成新的颜色映射
    
    Parameters:
    -----------
    old_df : pandas.DataFrame
        原始的dataframe，包含脑区信息
    coordinates_dict : dict
        脑区坐标字典，key格式为 'l.bankssts' 等
    activation_dict : dict
        激活值字典，key为节点ID（0-67），value为激活值
    cmap : str
        matplotlib colormap名称
    vmin, vmax : float
        颜色映射的最小值和最大值，如果为None则使用激活值的最小最大值
    
    Returns:
    --------
    new_df : pandas.DataFrame
        更新了color列的新dataframe
    """
    
    # 创建新的dataframe副本
    new_df = old_df.copy()
    
    # 创建脑区名称到节点ID的映射
    # 首先提取coordinates_dict中的脑区名称并排序，使其对应节点ID 0-67
    brain_regions = list(coordinates_dict.keys())
    
    # 创建名称到节点ID的映射
    name_to_node_id = {region: idx for idx, region in enumerate(brain_regions)}
    
    # 创建映射：dataframe中的fs_name格式到节点ID
    # 为每一行找到对应的节点ID和激活值
    node_ids = []
    activation_values = []
    
    for idx, row in new_df.iterrows():
        fs_name = row['fs_name']
        
        # 跳过unknown区域
        if 'unknown' in fs_name:
            node_ids.append(-1)
            activation_values.append(np.nan)
            continue
        
        # 获取对应的key格式
        key = fs_name_to_key(fs_name)
        
        if key and key in name_to_node_id:
            node_id = name_to_node_id[key]
            node_ids.append(node_id)
            
            # 获取激活值
            if node_id in activation_dict:
                activation_values.append(activation_dict[node_id])
            else:
                activation_values.append(np.nan)
        else:
            node_ids.append(-1)
            activation_values.append(np.nan)
    
    new_df['node_id'] = node_ids
    new_df['activation'] = activation_values
    
    # 处理颜色映射
    valid_activations = new_df['activation'].dropna()
    
    if len(valid_activations) > 0:
        if vmin is None:
            vmin = valid_activations.min()
        if vmax is None:
            vmax = valid_activations.max()
        
        # 创建颜色归一化对象
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # colormap = plt.cm.get_cmap(cmap)
        colormap = plt.colormaps.get_cmap(cmap)
        
        # 为每行生成新颜色
        new_colors = []
        for act in new_df['activation']:
            if pd.isna(act):
                # 对于unknown区域，采用激活值为0的颜色
                rgba_color = colormap(0)
                # 转换为十六进制颜色代码（包含alpha通道）
                hex_color = mcolors.to_hex(rgba_color, keep_alpha=True)
                new_colors.append(hex_color)
            else:
                # 将激活值映射到颜色
                rgba_color = colormap(norm(act))
                # 转换为十六进制颜色代码（包含alpha通道）
                hex_color = mcolors.to_hex(rgba_color, keep_alpha=True)
                new_colors.append(hex_color)
        
        new_df['color'] = new_colors
    
    return new_df

def plot_brain(lut, view='medial', template="fsaverage", density="10k"):
    mesh = {}
    data = {}
    for hemi in ["left", "right"]:
        mesh[hemi] = tflow.get(
            template,
            extension="surf.gii",
            suffix="pial",
            density=density,
            hemi=hemi[0].upper(),
        )
    
        roi_data = load_surf_data(
            tflow.get(
                template,
                atlas="Desikan2006",
                density=density,
                hemi=hemi[0].upper(),
                extension="label.gii",
            )
        )
    
        data[hemi] = roi_data
    
    desikan = SurfaceImage(mesh=mesh, data=data)
    
    # 创建单个图
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    fs_density = {
        "3k": "fsaverage4",
        "10k": "fsaverage5",
        "41k": "fsaverage6",
        "164k": "fsaverage",
    }
    
    sulcal_depth_map = load_fsaverage_data(
        mesh=fs_density[density], data_type="sulcal"
    )
    
    display = plot_surf_roi(
        roi_map=desikan,
        cmap=lut,
        bg_map=sulcal_depth_map,
        bg_on_data=True,
        # title=f"DK atlas ({density})",
        axes=ax,
        view=view,  # 设置视角：'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'
    )
    ax._colorbars[0].remove()
    
    display.savefig('output.png')

def plot_brain_comparison(lut_dfs, titles, view='medial', path='output.png', template="fsaverage", density="10k"):
    """
    绘制多个脑图的比较
    
    Parameters:
    -----------
    lut_dfs : list
        包含多个dataframe的列表，每个dataframe对应一个比较组
    titles : list
        每个子图的标题列表
    view : str
        视角：'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'
    template : str
        模板名称
    density : str
        密度：'3k', '10k', '41k', '164k'
    """
    
    mesh = {}
    data = {}
    for hemi in ["left", "right"]:
        mesh[hemi] = tflow.get(
            template,
            extension="surf.gii",
            suffix="pial",
            density=density,
            hemi=hemi[0].upper(),
        )
    
        roi_data = load_surf_data(
            tflow.get(
                template,
                atlas="Desikan2006",
                density=density,
                hemi=hemi[0].upper(),
                extension="label.gii",
            )
        )
    
        data[hemi] = roi_data
    
    desikan = SurfaceImage(mesh=mesh, data=data)
    
    # 创建包含3个子图的图
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(18, 6))
    
    fs_density = {
        "3k": "fsaverage4",
        "10k": "fsaverage5",
        "41k": "fsaverage6",
        "164k": "fsaverage",
    }
    
    sulcal_depth_map = load_fsaverage_data(
        mesh=fs_density[density], data_type="sulcal"
    )
    
    # 为每个比较组绘制脑图
    for idx, (lut_df, title) in enumerate(zip(lut_dfs, titles)):
        display = plot_surf_roi(
            roi_map=desikan,
            cmap=lut_df,
            bg_map=sulcal_depth_map,
            bg_on_data=True,
            title=title,
            axes=axes[idx],
            view=view,
        )
        # 移除colorbar，保持画面简洁
        axes[idx]._colorbars[0].remove()
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 运行主分析
    template = "fsaverage"
    lut = tflow.get(
        template,
        atlas="Desikan2006",
        suffix="dseg",
        extension="tsv",
    )
    old_df= pd.read_csv(lut, sep='\t')
    
    # activation_dict_Nor_vs_MCI = get_active_from_csv("../model_2_testset_result/cvib0_Nor_vs_MCI_significant_connections.csv")
    activation_dict_Nor_vs_DSC = get_active_from_csv("../model_2_testset_result/cvib0_Nor_vs_DSC_significant_connections.csv")
    activation_dict_Nor_vs_MCI = get_active_from_csv("../model_2_testset_result/cvib0_Nor_vs_MCI_significant_connections.csv")
    activation_dict_Nor_vs_AD = get_active_from_csv("../model_2_testset_result/cvib0_Nor_vs_AD_significant_connections.csv")
    
    # 生成新的dataframe
    # new_df = generate_colormap_based_on_activation(
    #     old_df=old_df,
    #     coordinates_dict=coordinates_data,
    #     activation_dict=activation_dict_Nor_vs_MCI,
    #     cmap='Reds', # 可以选择 'RdBu_r', 'seismic', 'viridis' 等
    #     vmin=0,  # 可选：设置颜色映射的最小值
    #     vmax=1    # 可选：设置颜色映射的最大值
    # )
    # 生成三个新的dataframe
    new_df_DSC = generate_colormap_based_on_activation(
        old_df=old_df,
        coordinates_dict=coordinates_data,
        activation_dict=activation_dict_Nor_vs_DSC,
        cmap='Reds',
        vmin=0,
        vmax=1
    )
    
    new_df_MCI = generate_colormap_based_on_activation(
        old_df=old_df,
        coordinates_dict=coordinates_data,
        activation_dict=activation_dict_Nor_vs_MCI,
        cmap='Reds',
        vmin=0,
        vmax=1
    )
    
    new_df_AD = generate_colormap_based_on_activation(
        old_df=old_df,
        coordinates_dict=coordinates_data,
        activation_dict=activation_dict_Nor_vs_AD,
        cmap='Reds',
        vmin=0,
        vmax=1
    )

    # plot_brain(new_df, view='medial')
    plot_brain_comparison(
        lut_dfs=[new_df_DSC, new_df_MCI, new_df_AD],
        titles=['Nor vs DSC', 'Nor vs MCI', 'Nor vs AD'],
        view='medial',
        path="output_medial.png"
    )

    plot_brain_comparison(
        lut_dfs=[new_df_DSC, new_df_MCI, new_df_AD],
        titles=['Nor vs DSC', 'Nor vs MCI', 'Nor vs AD'],
        view='dorsal',
        path="output_dorsal.png"
    )

    plot_brain_comparison(
        lut_dfs=[new_df_DSC, new_df_MCI, new_df_AD],
        titles=['Nor vs DSC', 'Nor vs MCI', 'Nor vs AD'],
        view='posterior',
        path="output_posterior.png"
    )


    

    print("finish")