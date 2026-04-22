import templateflow.api as tflow
from nilearn.surface import SurfaceImage, load_surf_data
import matplotlib.pyplot as plt
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_roi, show
import pandas as pd

template = "fsaverage"

# 只使用 10k 密度
density = "10k"

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

desikan_10k = SurfaceImage(mesh=mesh, data=data)

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

lut = tflow.get(
    template,
    atlas="Desikan2006",
    suffix="dseg",
    extension="tsv",
)
lut= pd.read_csv(lut, sep='\t')

print(lut)

plot_surf_roi(
    roi_map=desikan_10k,
    cmap=lut,
    bg_map=sulcal_depth_map,
    bg_on_data=True,
    title=f"DK atlas ({density})",
    axes=ax,
    view='lateral',  # 设置视角：'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'
)

show()