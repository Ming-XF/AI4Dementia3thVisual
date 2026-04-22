import numpy as np
from nilearn.plotting import view_connectome
from nilearn import plotting

import pdb


def load_and_preprocess_data():
    """
    加载正负样本数据
    """
    # 加载数据
    positive_data = np.load('./data/vae_positive_data.npy')  # 形状: (n_positive, 68, 68)
    negative_data = np.load('./data/vae_negtive_data.npy')   # 形状: (n_negative, 68, 68)
    
    print(f"正样本数量: {positive_data.shape[0]}")
    print(f"负样本数量: {negative_data.shape[0]}")
    print(f"每个FC图的形状: {positive_data.shape[1:]}")
    
    return positive_data, negative_data


def get_coors():
    coordinates_data = {
            'l.bankssts': [-54.343785, -44.539029, 4.163784],
            'l.caudalanteriorcingulate': [-5.030493, 20.087970, 28.999343],
            'l.caudalmiddlefrontal': [-35.521824, 10.809538, 44.190969],
            'l.cuneus': [-7.126394, -79.633054, 18.510034],
            'l.entorhinal': [-22.998997, -7.877750, -35.210045],
            'l.fusiform': [-35.141396, -43.474374, -22.012104],
            'l.inferiorparietal': [-40.934987, -67.711257, 28.124042],
            'l.inferiortemporal': [-49.601517, -34.703196, -25.261257],
            'l.isthmuscingulate': [-6.624240, -47.248045, 16.969356],
            'l.lateraloccipital': [-30.075425, -88.498214, -1.523382],
            'l.lateralorbitofrontal': [-24.788431, 28.715777, -16.968762],
            'l.lingual': [-14.506283, -67.608044, -5.063883],
            'l.medialorbitofrontal': [-5.406928, 36.933371, -18.001864],
            'l.middletemporal': [-57.751067, -30.223900, -13.290262],
            'l.parahippocampal': [-23.907604, -33.142327, -19.249481],
            'l.paracentral': [-7.897553, -29.735224, 56.120347],
            'l.parsopercularis': [-45.746763, 14.555066, 11.852166],
            'l.parsorbitalis': [-42.510831, 38.551743, -14.105952],
            'l.parstriangularis': [-44.017232, 30.266616, 0.805347],
            'l.pericalcarine': [-11.765939, -81.490718, 5.367253],
            'l.postcentral': [-43.230818, -23.574863, 43.947822],
            'l.posteriorcingulate': [-5.701530, -18.390072, 38.473745],
            'l.precentral': [-38.791732, -10.412277, 42.973217],
            'l.precuneus': [-9.690527, -58.233298, 36.662633],
            'l.rostralanteriorcingulate': [-4.385862, 37.523613, -0.212297],
            'l.rostralmiddlefrontal': [-33.210735, 42.715400, 16.838418],
            'l.superiorfrontal': [-11.379796, 24.090999, 43.374715],
            'l.superiorparietal': [-23.409777, -61.788495, 47.827327],
            'l.superiortemporal': [-53.401526, -15.660875, -4.006122],
            'l.supramarginal': [-52.060730, -39.127553, 31.482408],
            'l.frontalpole': [-6.785115, 64.865577, -11.499812],
            'l.temporalpole': [-29.311423, 12.899209, -38.046768],
            'l.transversetemporal': [-44.474530, -22.679342, 7.332593],
            'l.insula': [-37.137130, -3.504331, 1.690092],
            'r.bankssts': [52.975610, -40.553816, 5.303675],
            'r.caudalanteriorcingulate': [5.012041, 22.258100, 27.639678],
            'r.caudalmiddlefrontal': [35.661664, 12.293548, 44.471424],
            'r.cuneus': [7.165208, -80.094411, 19.161360],
            'r.entorhinal': [22.762408, -7.619649, -34.077827],
            'r.fusiform': [35.323007, -43.239303, -21.599235],
            'r.inferiorparietal': [44.345690, -61.781892, 28.633137],
            'r.inferiortemporal': [50.781852, -31.732182, -26.194109],
            'r.isthmuscingulate': [7.091746, -46.163380, 16.740261],
            'r.lateraloccipital': [31.117596, -87.943778, -0.457613],
            'r.lateralorbitofrontal': [24.236422, 29.349355, -17.996568],
            'r.lingual': [14.746014, -66.769025, -4.325692],
            'r.medialorbitofrontal': [5.859795, 37.568028, -16.583859],
            'r.middletemporal': [58.170481, -27.920889, -13.563721],
            'r.parahippocampal': [25.382203, -33.021476, -18.144719],
            'r.paracentral': [7.827022, -28.582297, 55.544456],
            'r.parsopercularis': [46.526088, 14.243186, 13.378662],
            'r.parsorbitalis': [44.123201, 39.162297, -11.985981],
            'r.parstriangularis': [46.623897, 29.459390, 3.342318],
            'r.pericalcarine': [12.515956, -80.239169, 6.009491],
            'r.postcentral': [42.365002, -22.476359, 44.557213],
            'r.posteriorcingulate': [5.685813, -17.196104, 38.859022],
            'r.precentral': [37.897994, -9.813155, 44.547376],
            'r.precuneus': [9.636033, -57.310060, 37.845502],
            'r.rostralanteriorcingulate': [5.367839, 37.109173, 1.676762],
            'r.rostralmiddlefrontal': [33.965750, 42.836306, 17.683447],
            'r.superiorfrontal': [12.193157, 25.698054, 43.075713],
            'r.superiorparietal': [23.192057, -60.478712, 49.691973],
            'r.superiortemporal': [54.388844, -12.260566, -5.118477],
            'r.supramarginal': [52.104994, -33.127594, 31.196859],
            'r.frontalpole': [8.694854, 64.422083, -11.920596],
            'r.temporalpole': [30.573320, 13.776967, -35.773082],
            'r.transversetemporal': [44.650935, -20.795417, 8.168757],
            'r.insula': [38.249302, -3.015601, 1.544895]
    }

    cods = []
    for key, value in coordinates_data.items():
        cods.append(value)
            
    return cods
        

def main():
    
    print("正在加载数据...")
    positive_data, negative_data = load_and_preprocess_data()
    cods = get_coors()
    
    active_data = np.abs(positive_data).meam(axis=-1)

    cnt = 0
    for i in range(len(positive_data)):
        if np.abs(positive_data[i]).max() > 0.1:
            view = view_connectome(positive_data[i], cods, edge_threshold=0.01)
            view.save_as_html("./output/positive{}.html".format(i))
            cnt += 1
            if cnt >= 3:
                break
    cnt = 0
    for i in range(len(negative_data)):
        if np.abs(negative_data[i]).max() > 0.1:
            view = view_connectome(negative_data[i], cods, edge_threshold=0.01)
            view.save_as_html("./output/negative{}.html".format(i))
            cnt += 1
            if cnt >= 3:
                break
    
if __name__ == "__main__":
    main()