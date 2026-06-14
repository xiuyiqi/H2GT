import numpy as np
import torch

def load_abide_data():
    # 加载数据字典
    data = np.load("path to new_dataset.npy", allow_pickle=True).item()
    
    # 提取字段并保存到不同变量
    feature_matrix_0 = data.get("feature_matrix_0")
    feature_matrix_1 = data.get("feature_matrix_1")
    feature_matrix_2 = data.get("feature_matrix_2")
    feature_matrix_3 = data.get("feature_matrix_3")
    feature_matrix_4 = data.get("feature_matrix_4")
    feature_matrix_5 = data.get("feature_matrix_5")
    feature_matrix_6 = data.get("feature_matrix_6")
    feature_matrix_7 = data.get("feature_matrix_7")
    
    adjacency_matrix_0 = data.get("adjacency_matrix_0")
    adjacency_matrix_1 = data.get("adjacency_matrix_1")
    adjacency_matrix_2 = data.get("adjacency_matrix_2")
    adjacency_matrix_3 = data.get("adjacency_matrix_3")
    adjacency_matrix_4 = data.get("adjacency_matrix_4")
    adjacency_matrix_5 = data.get("adjacency_matrix_5")
    adjacency_matrix_6 = data.get("adjacency_matrix_6")
    adjacency_matrix_7 = data.get("adjacency_matrix_7")
    
    module_adjacency = data.get("module_adjacency")
    labels = data.get("label")
    site = data.get("site")
    
    # 将数据转化为张量
    feature_matrix_0, feature_matrix_1, feature_matrix_2, feature_matrix_3, feature_matrix_4, \
feature_matrix_5, feature_matrix_6, feature_matrix_7, adjacency_matrix_0, adjacency_matrix_1, \
adjacency_matrix_2, adjacency_matrix_3, adjacency_matrix_4, adjacency_matrix_5, adjacency_matrix_6, \
adjacency_matrix_7, module_adjacency, labels = [torch.from_numpy(np.array(data)).float() 
                                              for data in (feature_matrix_0, feature_matrix_1, 
                                                           feature_matrix_2, feature_matrix_3, 
                                                           feature_matrix_4, feature_matrix_5, 
                                                           feature_matrix_6, feature_matrix_7, 
                                                           adjacency_matrix_0, adjacency_matrix_1, 
                                                           adjacency_matrix_2, adjacency_matrix_3, 
                                                           adjacency_matrix_4, adjacency_matrix_5, 
                                                           adjacency_matrix_6, adjacency_matrix_7, 
                                                           module_adjacency, labels)]
    site = np.array(site)
    # 返回每个数据的变量
    return (feature_matrix_0, feature_matrix_1, feature_matrix_2, feature_matrix_3, feature_matrix_4,
            feature_matrix_5, feature_matrix_6, feature_matrix_7, adjacency_matrix_0, adjacency_matrix_1,
            adjacency_matrix_2, adjacency_matrix_3, adjacency_matrix_4, adjacency_matrix_5, adjacency_matrix_6,
            adjacency_matrix_7, module_adjacency, labels, site)

