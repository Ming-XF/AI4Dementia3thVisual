import numpy as np
from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelsMasker

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
# target_regions = ['Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus']
target_regions = ['Middle Frontal Gyrus', 'Temporal Pole', 'Middle Frontal Gyrus', 'Cingulate Gyrus, posterior division', 'Middle Frontal Gyrus', 'Cingulate Gyrus, anterior division', 'Superior Frontal Gyrus', 'Frontal Medial Cortex', 'Cingulate Gyrus, posterior division']

temp = np.array([5.1252, 5.1252, 4.416, 4.416, 4.323, 4.323, 4.0207, 3.9866, 3.9866])
temp = temp * 10
                  

# 获取这些脑区的索引
selected_indices, acts = get_indices_by_names(target_regions, temp)

# 创建激活数据
activation_data = np.zeros(len(atlas.labels))
for idx, act in zip(selected_indices, acts):
    activation_data[idx] = act
    print(atlas.labels[idx])
    print('{}-{}'.format(idx, act))



# 创建激活图
masker = NiftiLabelsMasker(labels_img=atlas.maps)
masker.fit()
activation_map = masker.inverse_transform(activation_data.reshape(1, -1))

# display = plotting.plot_stat_map(activation_map)

display = plotting.plot_stat_map(activation_map, 
                       title='',
                       display_mode='z',
                       cut_coords=4,
                       cmap='cold_hot',
                       annotate=True,
                       colorbar=True)

# 保存图片
display.savefig('brain_activation_map.png')
display.close()

print("图片已保存为 brain_activation_map.png")





# Temporal Pole-0.8542000000000001
# Middle Frontal Gyrus-0.7205
# Cingulate Gyrus, anterior division-0.7205
# Superior Frontal Gyrus-0.6701166666666666
# Frontal Medial Cortex-0.6644333333333333
# ingulate Gyrus, posterior division-0.6644333333333333


# Middle Frontal Gyrus-0.8542000000000001
# Temporal Pole-0.8542000000000001
# Middle Frontal Gyrus-0.7360000000000001
# Cingulate Gyrus, posterior division-0.7360000000000001
# Middle Frontal Gyrus-0.7205
# Cingulate Gyrus, anterior division-0.7205
# Superior Frontal Gyrus-0.6701166666666666
# Frontal Medial Cortex-0.6644333333333333
# Cingulate Gyrus, posterior division-0.6644333333333333


# Middle Frontal Gyrus-0.8542000000000001
# Temporal Pole-0.8542000000000001
# Middle Frontal Gyrus-0.7360000000000001
# Cingulate Gyrus, posterior division-0.7360000000000001
# Middle Frontal Gyrus-0.7205
# Cingulate Gyrus, anterior division -0.7205
# Superior Frontal Gyrus-0.6701166666666666
# Frontal Medial Cortex-0.6644333333333333
# Cingulate Gyrus, posterior division -0.6644333333333333


# Middle Frontal Gyrus-0.8542000000000001
# Temporal Pole-0.8542000000000001
# Middle Frontal Gyrus-0.7360000000000001
# Cingulate Gyrus, posterior division-0.7360000000000001
# Middle Frontal Gyrus-0.7205
# Cingulate Gyrus, anterior division -0.7205
# Superior Frontal Gyrus-0.6701166666666666
# Frontal Medial Cortex-0.6644333333333333
# Cingulate Gyrus, posterior division -0.6644333333333333


# Temporal Pole
# 8-0.8542000000000001
# Middle Frontal Gyrus
# 4-0.7205
# Cingulate Gyrus, anterior division
# 29-0.7205
# Superior Frontal Gyrus
# 3-0.6701166666666666
# Frontal Medial Cortex
# 25-0.6644333333333333
# Cingulate Gyrus, posterior division
# 30-0.6644333333333333