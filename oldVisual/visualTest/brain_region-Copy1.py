import numpy as np
from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
import matplotlib.pyplot as plt

# 获取哈佛-牛津脑图谱
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# 通过脑区名称获取索引
def get_indices_by_names(names, acts):
    indices = []
    oacts = []
    for name, act in zip(names, acts):
        for i, label in enumerate(atlas.labels):
            if name in label:
                indices.append(i)
                oacts.append(act)
                print(f"找到脑区: {label} (索引: {i})")
                break
    return indices, oacts

# 指定要显示的脑区名称
target_regions = ['Middle Frontal Gyrus', 'Temporal Pole', 'Middle Frontal Gyrus', 
                  'Cingulate Gyrus, posterior division', 'Middle Frontal Gyrus', 
                  'Cingulate Gyrus, anterior division', 'Superior Frontal Gyrus', 
                  'Frontal Medial Cortex', 'Cingulate Gyrus, posterior division']

temp = np.array([5.1252, 5.1252, 4.416, 4.416, 4.323, 4.323, 4.0207, 3.9866, 3.9866])
temp = temp / 6
                  
# 获取这些脑区的索引
selected_indices, acts = get_indices_by_names(target_regions, temp)

# 创建激活数据
activation_data = np.zeros(len(atlas.labels))
for idx, act in zip(selected_indices, acts):
    activation_data[idx] = act
    print('{}-{}'.format(idx, act))

# 创建激活图
masker = NiftiLabelsMasker(labels_img=atlas.maps)
masker.fit()
activation_map = masker.inverse_transform(activation_data.reshape(1, -1))

# 使用plot_roi来绘制，这样可以更容易添加标签
display = plotting.plot_roi(activation_map, 
                           title='Brain Activation Map with Region Labels',
                           cmap='hot',
                           colorbar=True)

# 为每个脑区手动添加标签
for i, idx in enumerate(selected_indices):
    # 创建单个脑区的掩码
    single_region_data = np.zeros(len(atlas.labels))
    single_region_data[idx] = 1
    single_region_map = masker.inverse_transform(single_region_data.reshape(1, -1))
    
    # 获取坐标
    coords = plotting.find_xyz_cut_coords(single_region_map)
    
    # 在图中添加文本
    # 由于nilearn的标注方法有问题，我们直接使用matplotlib的text
    fig = plt.gcf()
    
    # 这个方法比较简单，但可能不会在所有切片上都显示
    # 我们只在其中一个视图上显示标签
    ax = display.axes['z']
    ax.text(coords[0], coords[1], f'{idx}', 
            fontsize=8, color='blue', weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

# 保存图片
plt.savefig('brain_activation_map_with_labels.png', dpi=300, bbox_inches='tight')
plt.show()

print("图片已保存为 brain_activation_map_with_labels.png")