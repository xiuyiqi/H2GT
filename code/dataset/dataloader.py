import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F

batch_size = 64
test_batch_size = 64
val_batch_size = 64
train_set = 0.7
val_set = 0.1
drop_last = True
percentage = 1

# 普通的随机拆分数据集
def init_dataloader(feature_matrix_0: torch.Tensor, 
                    feature_matrix_1: torch.Tensor, 
                    feature_matrix_2: torch.Tensor, 
                    feature_matrix_3: torch.Tensor, 
                    feature_matrix_4: torch.Tensor,
                    feature_matrix_5: torch.Tensor, 
                    feature_matrix_6: torch.Tensor, 
                    feature_matrix_7: torch.Tensor, 
                    adjacency_matrix_0: torch.Tensor, 
                    adjacency_matrix_1: torch.Tensor,
                    adjacency_matrix_2: torch.Tensor, 
                    adjacency_matrix_3: torch.Tensor, 
                    adjacency_matrix_4: torch.Tensor, 
                    adjacency_matrix_5: torch.Tensor, 
                    adjacency_matrix_6: torch.Tensor,
                    adjacency_matrix_7: torch.Tensor, 
                    module_adjacency: torch.Tensor, 
                    labels: torch.Tensor) -> List[utils.DataLoader]:
    length = labels.shape[0] # 样本数量
    labels = F.one_hot(labels.to(torch.int64)) # 将标签转换为独热编码形式
    train_length = int(length*train_set*percentage) # 训练集的样本数量
    val_length = int(length*val_set) # 验证集的样本数量
    if percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-val_set-train_set)) # 测试集的样本数量

    # 确保所有张量长度一致
    total_length = train_length + val_length + test_length
    dataset = utils.TensorDataset(
        feature_matrix_0[:total_length],
        feature_matrix_1[:total_length],
        feature_matrix_2[:total_length],
        feature_matrix_3[:total_length],
        feature_matrix_4[:total_length],
        feature_matrix_5[:total_length],
        feature_matrix_6[:total_length],
        feature_matrix_7[:total_length],
        adjacency_matrix_0[:total_length],
        adjacency_matrix_1[:total_length],
        adjacency_matrix_2[:total_length],
        adjacency_matrix_3[:total_length],
        adjacency_matrix_4[:total_length],
        adjacency_matrix_5[:total_length],
        adjacency_matrix_6[:total_length],
        adjacency_matrix_7[:total_length],
        module_adjacency[:total_length],
        labels[:total_length]
    ) 

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length]) 
    
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True)
    
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

    return [train_dataloader, val_dataloader, test_dataloader]

# 分层抽样拆分数据集
def init_stratified_dataloader(feature_matrix_0: torch.Tensor, 
                               feature_matrix_1: torch.Tensor, 
                               feature_matrix_2: torch.Tensor, 
                               feature_matrix_3: torch.Tensor, 
                               feature_matrix_4: torch.Tensor,
                               feature_matrix_5: torch.Tensor, 
                               feature_matrix_6: torch.Tensor, 
                               feature_matrix_7: torch.Tensor, 
                               adjacency_matrix_0: torch.Tensor, 
                               adjacency_matrix_1: torch.Tensor,
                               adjacency_matrix_2: torch.Tensor, 
                               adjacency_matrix_3: torch.Tensor, 
                               adjacency_matrix_4: torch.Tensor, 
                               adjacency_matrix_5: torch.Tensor, 
                               adjacency_matrix_6: torch.Tensor,
                               adjacency_matrix_7: torch.Tensor, 
                               module_adjacency: torch.Tensor,
                               labels: torch.Tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    length = labels.shape[0] # 时间序列的样本数量
    labels = F.one_hot(labels.to(torch.int64)) # 将标签转换为独热编码形式
    train_length = int(length*train_set*percentage) # 训练集的样本数量
    val_length = int(length*val_set) # 验证集的样本数量
    if percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-val_set-train_set)) # 测试集的样本数量
    
    # 第一次分层抽样：划分训练集和验证+测试集
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    
    # 修正多行赋值语法错误
    for train_index, test_valid_index in split.split(feature_matrix_0, stratified):
        # 使用正确的方式处理多行赋值
        feature_matrix_0_train = feature_matrix_0[train_index]
        feature_matrix_1_train = feature_matrix_1[train_index]
        feature_matrix_2_train = feature_matrix_2[train_index]
        feature_matrix_3_train = feature_matrix_3[train_index]
        feature_matrix_4_train = feature_matrix_4[train_index]
        feature_matrix_5_train = feature_matrix_5[train_index]
        feature_matrix_6_train = feature_matrix_6[train_index]
        feature_matrix_7_train = feature_matrix_7[train_index]
        
        adjacency_matrix_0_train = adjacency_matrix_0[train_index]
        adjacency_matrix_1_train = adjacency_matrix_1[train_index]
        adjacency_matrix_2_train = adjacency_matrix_2[train_index]
        adjacency_matrix_3_train = adjacency_matrix_3[train_index]
        adjacency_matrix_4_train = adjacency_matrix_4[train_index]
        adjacency_matrix_5_train = adjacency_matrix_5[train_index]
        adjacency_matrix_6_train = adjacency_matrix_6[train_index]
        adjacency_matrix_7_train = adjacency_matrix_7[train_index]
        
        module_adjacency_train = module_adjacency[train_index]
        labels_train = labels[train_index]

        # 验证测试集部分
        feature_matrix_0_val_test = feature_matrix_0[test_valid_index]
        feature_matrix_1_val_test = feature_matrix_1[test_valid_index]
        feature_matrix_2_val_test = feature_matrix_2[test_valid_index]
        feature_matrix_3_val_test = feature_matrix_3[test_valid_index]
        feature_matrix_4_val_test = feature_matrix_4[test_valid_index]
        feature_matrix_5_val_test = feature_matrix_5[test_valid_index]
        feature_matrix_6_val_test = feature_matrix_6[test_valid_index]
        feature_matrix_7_val_test = feature_matrix_7[test_valid_index]
        
        adjacency_matrix_0_val_test = adjacency_matrix_0[test_valid_index]
        adjacency_matrix_1_val_test = adjacency_matrix_1[test_valid_index]
        adjacency_matrix_2_val_test = adjacency_matrix_2[test_valid_index]
        adjacency_matrix_3_val_test = adjacency_matrix_3[test_valid_index]
        adjacency_matrix_4_val_test = adjacency_matrix_4[test_valid_index]
        adjacency_matrix_5_val_test = adjacency_matrix_5[test_valid_index]
        adjacency_matrix_6_val_test = adjacency_matrix_6[test_valid_index]
        adjacency_matrix_7_val_test = adjacency_matrix_7[test_valid_index]
        
        module_adjacency_val_test = module_adjacency[test_valid_index]
        labels_val_test = labels[test_valid_index]
    
    # 修正变量名错误 (test_valid_index -> test_valid_index)
    val_test_stratified = stratified[test_valid_index]
    
    # 第二次分层抽样：划分验证集和测试集
    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    
    for test_index, valid_index in split2.split(feature_matrix_0_val_test, val_test_stratified):
        # 测试集部分
        feature_matrix_0_test = feature_matrix_0_val_test[test_index]
        feature_matrix_1_test = feature_matrix_1_val_test[test_index]
        feature_matrix_2_test = feature_matrix_2_val_test[test_index]
        feature_matrix_3_test = feature_matrix_3_val_test[test_index]
        feature_matrix_4_test = feature_matrix_4_val_test[test_index]
        feature_matrix_5_test = feature_matrix_5_val_test[test_index]
        feature_matrix_6_test = feature_matrix_6_val_test[test_index]
        feature_matrix_7_test = feature_matrix_7_val_test[test_index]
        
        adjacency_matrix_0_test = adjacency_matrix_0_val_test[test_index]
        adjacency_matrix_1_test = adjacency_matrix_1_val_test[test_index]
        adjacency_matrix_2_test = adjacency_matrix_2_val_test[test_index]
        adjacency_matrix_3_test = adjacency_matrix_3_val_test[test_index]
        adjacency_matrix_4_test = adjacency_matrix_4_val_test[test_index]
        adjacency_matrix_5_test = adjacency_matrix_5_val_test[test_index]
        adjacency_matrix_6_test = adjacency_matrix_6_val_test[test_index]
        adjacency_matrix_7_test = adjacency_matrix_7_val_test[test_index]
        
        module_adjacency_test = module_adjacency_val_test[test_index]
        labels_test = labels_val_test[test_index]
        
        # 验证集部分
        feature_matrix_0_val = feature_matrix_0_val_test[valid_index]
        feature_matrix_1_val = feature_matrix_1_val_test[valid_index]
        feature_matrix_2_val = feature_matrix_2_val_test[valid_index]
        feature_matrix_3_val = feature_matrix_3_val_test[valid_index]
        feature_matrix_4_val = feature_matrix_4_val_test[valid_index]
        feature_matrix_5_val = feature_matrix_5_val_test[valid_index]
        feature_matrix_6_val = feature_matrix_6_val_test[valid_index]
        feature_matrix_7_val = feature_matrix_7_val_test[valid_index]
        
        adjacency_matrix_0_val = adjacency_matrix_0_val_test[valid_index]
        adjacency_matrix_1_val = adjacency_matrix_1_val_test[valid_index]
        adjacency_matrix_2_val = adjacency_matrix_2_val_test[valid_index]
        adjacency_matrix_3_val = adjacency_matrix_3_val_test[valid_index]
        adjacency_matrix_4_val = adjacency_matrix_4_val_test[valid_index]
        adjacency_matrix_5_val = adjacency_matrix_5_val_test[valid_index]
        adjacency_matrix_6_val = adjacency_matrix_6_val_test[valid_index]
        adjacency_matrix_7_val = adjacency_matrix_7_val_test[valid_index]
        
        module_adjacency_val = module_adjacency_val_test[valid_index]
        labels_val = labels_val_test[valid_index]

    # 构建训练数据集
    train_dataset = utils.TensorDataset(
        feature_matrix_0_train,
        feature_matrix_1_train,
        feature_matrix_2_train,
        feature_matrix_3_train,
        feature_matrix_4_train,
        feature_matrix_5_train,
        feature_matrix_6_train,
        feature_matrix_7_train,
        adjacency_matrix_0_train,
        adjacency_matrix_1_train,
        adjacency_matrix_2_train,
        adjacency_matrix_3_train,
        adjacency_matrix_4_train,
        adjacency_matrix_5_train,
        adjacency_matrix_6_train,
        adjacency_matrix_7_train,
        module_adjacency_train,
        labels_train
    )
    
    # 构建验证数据集
    val_dataset = utils.TensorDataset(
        feature_matrix_0_val,
        feature_matrix_1_val,
        feature_matrix_2_val,
        feature_matrix_3_val,
        feature_matrix_4_val,
        feature_matrix_5_val,
        feature_matrix_6_val,
        feature_matrix_7_val,
        adjacency_matrix_0_val,
        adjacency_matrix_1_val,
        adjacency_matrix_2_val,
        adjacency_matrix_3_val,
        adjacency_matrix_4_val,
        adjacency_matrix_5_val,
        adjacency_matrix_6_val,
        adjacency_matrix_7_val,
        module_adjacency_val,
        labels_val
    )
    
    # 构建测试数据集
    test_dataset = utils.TensorDataset(
        feature_matrix_0_test,
        feature_matrix_1_test,
        feature_matrix_2_test,
        feature_matrix_3_test,
        feature_matrix_4_test,
        feature_matrix_5_test,
        feature_matrix_6_test,
        feature_matrix_7_test,
        adjacency_matrix_0_test,
        adjacency_matrix_1_test,
        adjacency_matrix_2_test,
        adjacency_matrix_3_test,
        adjacency_matrix_4_test,
        adjacency_matrix_5_test,
        adjacency_matrix_6_test,
        adjacency_matrix_7_test,
        module_adjacency_test,
        labels_test
    )
    
    # 创建数据加载器
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True)
    
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

    return [train_dataloader, val_dataloader, test_dataloader]