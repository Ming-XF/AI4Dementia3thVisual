import pandas as pd
import pdb


def get_active_from_csv(path):
    # 读取 CSV 文件
    df = pd.read_csv(path)  # 请替换为实际文件路径
    
    # 合并 Region1 和 Region2 两列，统计每个区域出现的次数
    region_counts = pd.concat([df['Region1'], df['Region2']]).value_counts().sort_index()

    return region_counts

if __name__ == "__main__":
    # 运行主分析
    pdb.set_trace()
    result = get_active_from_csv("./model_2_testset_result/cvib0_Nor_vs_AD_significant_connections.csv")

    print("finish")